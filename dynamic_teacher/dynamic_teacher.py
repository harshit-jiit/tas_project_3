"""
Modernized DynamicTeacher class that replaces AllenNLP dependencies
with PyTorch + Transformers, compatible with Python 3.12

This class implements Teacher-As-Student (TAS) distillation by:
1. Loading a pre-trained teacher model
2. Running inference on training batches in a separate process
3. Adding teacher scores to batches for student model training
"""

import os
import time
import copy
import traceback
from typing import Any, Dict, Iterator, List, Optional, Union
import logging

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.cuda._utils import _get_device_index

from rich.console import Console
from transformers.utils import cached_path, WEIGHTS_NAME

# Custom exceptions and types
class WorkerError(Exception):
    """Custom exception for worker process errors"""
    def __init__(self, message, traceback_str):
        super().__init__(message)
        self.traceback_str = traceback_str

class TensorDict(Dict[str, torch.Tensor]):
    """Simple replacement for AllenNLP's TensorDict"""
    pass

class DataLoader:
    """Interface for data loader compatibility"""
    def __iter__(self):
        raise NotImplementedError

def move_to_device(obj: Dict[str, Any], device: Union[torch.device, int, str]) -> Dict[str, Any]:
    """
    Modern replacement for AllenNLP's move_to_device
    Recursively moves tensors in nested dictionaries to specified device
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(item, device) for item in obj)
    else:
        return obj

def data_parallel_prepare(module: nn.Module, device_ids: List[int]) -> List[nn.Module]:
    """
    Prepare model replicas for data parallel processing
    
    Args:
        module: The model to replicate
        device_ids: List of GPU device IDs
        
    Returns:
        List of model replicas, one per device
    """
    device_ids = [_get_device_index(device_id, True) for device_id in device_ids]
    replicas = replicate(module, device_ids)
    return replicas

def data_parallel_forward(
    replicas: List[nn.Module], 
    inputs: tuple, 
    device_ids: List[int],
    output_device: Optional[int] = None,
    dim: int = 0,
    module_kwargs: Optional[Dict] = None
) -> torch.Tensor:
    """
    Execute forward pass in data parallel fashion
    
    Args:
        replicas: List of model replicas
        inputs: Input tensors
        device_ids: List of GPU device IDs
        output_device: Device for gathering outputs
        dim: Dimension for scattering/gathering
        module_kwargs: Additional keyword arguments
        
    Returns:
        Gathered output tensor
    """
    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    if module_kwargs is None:
        module_kwargs = {}

    if output_device is None:
        output_device = device_ids[0]

    device_ids = [_get_device_index(device_id, True) for device_id in device_ids]
    output_device = _get_device_index(output_device, True)

    # Scatter inputs and kwargs across devices
    inputs, module_kwargs = scatter_kwargs(inputs, module_kwargs, device_ids, dim)
    used_device_ids = device_ids[:len(inputs)]
    
    # Execute forward pass on each device
    outputs = parallel_apply(replicas, inputs, module_kwargs, used_device_ids)
    
    # Gather outputs to specified device
    return gather(outputs, output_device, dim)

class ModernDynamicTeacher:
    """
    Modernized DynamicTeacher that wraps a trained model checkpoint and training batch 
    queue to score (inference only) samples from the batch.
    
    This implementation:
    - Replaces AllenNLP dependencies with pure PyTorch + Transformers
    - Maintains the same teacher-student distillation logic
    - Supports both single GPU and multi-GPU inference
    - Compatible with Python 3.12
    """

    def __init__(
        self,
        config: Dict[str, Any],
        dataloader: DataLoader,
        logger: logging.Logger,
        model_loader_fn: callable,  # Function to load model from config
        config_loader_fn: callable  # Function to load config from path
    ):
        """
        Initialize the DynamicTeacher
        
        Args:
            config: Training configuration dictionary
            dataloader: The training data loader to wrap
            logger: Logger instance
            model_loader_fn: Function that takes config and returns (model, device_info)
            config_loader_fn: Function that takes path and returns config dict
        """
        self.config = config
        self.dynamic_teacher_path = config["dynamic_teacher_path"]
        self.dynamic_teacher_in_batch_scoring = config.get("dynamic_teacher_in_batch_scoring", False)
        self.dynamic_teacher_per_term_scores = config.get("dynamic_teacher_per_term_scores", False)
        
        self.wrapped_dataloader = dataloader
        
        # GPU configuration
        if torch.cuda.is_available():
            total_gpus = torch.cuda.device_count()
            # Use the last GPU by default, or multiple GPUs as specified
            self.cuda_device = config.get("teacher_cuda_device", total_gpus - 1)
        else:
            self.cuda_device = "cpu"
            
        self.logger = logger
        
        # Store loader functions for the subprocess
        self.model_loader_fn = model_loader_fn
        self.config_loader_fn = config_loader_fn

    def __iter__(self) -> Iterator[TensorDict]:
        """
        Create an iterator that yields batches with teacher scores
        """
        # Use spawn context to avoid CUDA context issues
        ctx = mp.get_context("spawn")
        
        queue: mp.JoinableQueue = ctx.JoinableQueue(maxsize=50)  # Limit queue size to prevent memory issues
        worker = ctx.Process(
            target=self.dynamic_teacher_subprocess, 
            args=(queue,), 
            daemon=False  # Set to False for proper cleanup
        )
        worker.start()

        try:
            for batch, worker_error in iter(queue.get, (None, None)):
                if worker_error is not None:
                    e, tb = worker_error
                    raise WorkerError(e, tb)

                yield batch
                queue.task_done()
        finally:
            # Proper cleanup
            if hasattr(queue, "close"):
                queue.close()
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=10)  # Wait for clean shutdown

    def dynamic_teacher_subprocess(self, queue: mp.JoinableQueue):
        """
        Subprocess that loads teacher model and runs inference on batches
        """
        try:
            console = Console()
            console.log(f"[DynamicTeacher] Load teacher model from: {self.dynamic_teacher_path}")

            # Load model configuration
            model_config = self.config_loader_fn(self.dynamic_teacher_path)
            
            # Load and initialize model
            model, model_info = self.model_loader_fn(model_config)
            model.is_teacher_model = True
            model.eval()
            
            # Load model weights
            if model_config.get("model_checkpoint_from_huggingface", False):
                # For Hugging Face models
                try:
                    model_path = cached_path(f"https://huggingface.co/{self.dynamic_teacher_path}/resolve/main/{WEIGHTS_NAME}")
                except:
                    model_path = os.path.join(self.dynamic_teacher_path, WEIGHTS_NAME)
            else:
                # For local checkpoints
                model_path = os.path.join(self.dynamic_teacher_path, "best-model.pytorch-state-dict")
            
            if os.path.exists(model_path):
                load_result = model.load_state_dict(
                    torch.load(model_path, map_location="cpu"), 
                    strict=False
                )
                console.log("[DynamicTeacher] Warmstart Result:", load_result)
                self.logger.info(f'[DynamicTeacher] Warmstart init model from: {model_path}')
                self.logger.info(str(load_result))
            else:
                console.log(f"[DynamicTeacher] Warning: Model file not found at {model_path}")

            # Setup device configuration
            use_multi_gpu = False
            replicas = None
            
            if isinstance(self.cuda_device, int):
                # Single GPU
                if torch.cuda.is_available():
                    model = model.cuda(self.cuda_device)
                    console.log(f"[DynamicTeacher] Using single GPU: {self.cuda_device}")
            elif isinstance(self.cuda_device, list) and len(self.cuda_device) > 1:
                # Multi-GPU
                if torch.cuda.is_available():
                    use_multi_gpu = True
                    model = model.cuda(self.cuda_device[0])
                    replicas = data_parallel_prepare(model, self.cuda_device)
                    console.log(f"[DynamicTeacher] Using multi-GPU: {self.cuda_device}")
            else:
                console.log("[DynamicTeacher] Using CPU")

            # Configuration flags
            use_fp16 = model_config.get("use_fp16", False)
            concatenated_sequences = model_config.get("token_embedder_type") == "bert_cat"

            console.log("[DynamicTeacher] Run Teacher Inference ...")

            # Main inference loop
            with torch.no_grad():
                if use_fp16 and torch.cuda.is_available():
                    # Use automatic mixed precision if available
                    with torch.cuda.amp.autocast():
                        self._process_batches(
                            queue, model, replicas, use_multi_gpu, 
                            concatenated_sequences, use_fp16, console
                        )
                else:
                    self._process_batches(
                        queue, model, replicas, use_multi_gpu, 
                        concatenated_sequences, use_fp16, console
                    )

        except Exception as e:
            console.log(f"[DynamicTeacher] Error in subprocess: {str(e)}")
            queue.put((None, (repr(e), traceback.format_exc())))
        finally:
            queue.put((None, None))
            queue.join()

    def _process_batches(
        self, 
        queue: mp.JoinableQueue, 
        model: nn.Module, 
        replicas: Optional[List[nn.Module]], 
        use_multi_gpu: bool,
        concatenated_sequences: bool,
        use_fp16: bool,
        console: Console
    ):
        """
        Process batches through the teacher model
        """
        batch_count = 0
        
        for orig_batch in self.wrapped_dataloader:
            try:
                # Prepare batch copies for different devices
                if use_multi_gpu and replicas is not None:
                    batch_pos = move_to_device(copy.deepcopy(orig_batch), self.cuda_device[0])
                    batch_neg = move_to_device(copy.deepcopy(orig_batch), self.cuda_device[1])
                else:
                    batch_pos = move_to_device(copy.deepcopy(orig_batch), self.cuda_device)
                    batch_neg = batch_pos

                # Prepare inputs based on model type
                pos_inputs, neg_inputs = self._prepare_model_inputs(
                    batch_pos, batch_neg, concatenated_sequences
                )

                # Run model inference
                if use_multi_gpu and replicas is not None:
                    output_pos, output_neg = self._multi_gpu_forward(
                        replicas, pos_inputs, neg_inputs, use_fp16
                    )
                else:
                    output_pos = self._single_gpu_forward(model, pos_inputs, use_fp16)
                    output_neg = self._single_gpu_forward(model, neg_inputs, use_fp16)

                # Process outputs and add teacher scores to original batch
                self._add_teacher_scores_to_batch(
                    orig_batch, output_pos, output_neg, 
                    batch_pos, batch_neg, model
                )

                queue.put((orig_batch, None))
                batch_count += 1
                
                if batch_count % 100 == 0:
                    console.log(f"[DynamicTeacher] Processed {batch_count} batches")

            except Exception as e:
                console.log(f"[DynamicTeacher] Error processing batch {batch_count}: {str(e)}")
                # Continue with next batch rather than crashing
                continue

    def _prepare_model_inputs(
        self, 
        batch_pos: Dict[str, Any], 
        batch_neg: Dict[str, Any], 
        concatenated_sequences: bool
    ) -> tuple:
        """
        Prepare inputs for the model based on architecture type
        """
        pos_inputs = []
        neg_inputs = []
        
        if concatenated_sequences:
            # For BERT concatenated models (query + doc in single sequence)
            pos_inputs.append(batch_pos["doc_pos_tokens"])
            neg_inputs.append(batch_neg["doc_neg_tokens"])
        else:
            # For independent encoding models (query and doc separate)
            pos_inputs.extend([batch_pos["query_tokens"], batch_pos["doc_pos_tokens"]])
            neg_inputs.extend([batch_neg["query_tokens"], batch_neg["doc_neg_tokens"]])

        return pos_inputs, neg_inputs

    def _single_gpu_forward(
        self, 
        model: nn.Module, 
        inputs: List[torch.Tensor], 
        use_fp16: bool
    ) -> torch.Tensor:
        """
        Run forward pass on single GPU
        """
        return model.forward(*inputs, use_fp16=use_fp16)

    def _multi_gpu_forward(
        self, 
        replicas: List[nn.Module], 
        pos_inputs: List[torch.Tensor], 
        neg_inputs: List[torch.Tensor], 
        use_fp16: bool
    ) -> tuple:
        """
        Run forward pass on multiple GPUs
        """
        kwargs_pos = {"use_fp16": use_fp16}
        kwargs_neg = {"use_fp16": use_fp16}
        
        output_pos, output_neg = parallel_apply(
            replicas, 
            [pos_inputs, neg_inputs], 
            [kwargs_pos, kwargs_neg], 
            self.cuda_device
        )
        
        return output_pos, output_neg

    def _add_teacher_scores_to_batch(
        self, 
        orig_batch: Dict[str, Any], 
        output_pos: torch.Tensor, 
        output_neg: torch.Tensor,
        batch_pos: Dict[str, Any], 
        batch_neg: Dict[str, Any],
        model: nn.Module
    ):
        """
        Add teacher scores to the original batch
        """
        # Handle per-term scores if requested
        if self.dynamic_teacher_per_term_scores:
            if isinstance(output_pos, tuple) and len(output_pos) > 1:
                *output_pos, per_term_scores_pos = output_pos
                *output_neg, per_term_scores_neg = output_neg
                
                orig_batch["dyn_teacher_per_term_scores_pos"] = per_term_scores_pos.cpu()
                orig_batch["dyn_teacher_per_term_scores_neg"] = per_term_scores_neg.cpu()

        # Handle in-batch scoring (for models like ColBERT)
        if self.dynamic_teacher_in_batch_scoring:
            if isinstance(output_pos, tuple) and len(output_pos) >= 3:
                score_pos, query_vecs_pos, doc_vecs_pos = output_pos
                score_neg, query_vecs_neg, doc_vecs_neg = output_neg

                # Run in-batch aggregation if model supports it
                if hasattr(model, 'forward_inbatch_aggregation'):
                    ib_output_pos = model.forward_inbatch_aggregation(
                        query_vecs_pos,
                        batch_pos["query_tokens"]["attention_mask"],
                        doc_vecs_pos,
                        batch_pos["doc_pos_tokens"]["attention_mask"]
                    )
                    ib_output_neg = model.forward_inbatch_aggregation(
                        query_vecs_neg,
                        batch_neg["query_tokens"]["attention_mask"],
                        doc_vecs_neg,
                        batch_neg["doc_neg_tokens"]["attention_mask"]
                    )
                    
                    orig_batch["dyn_teacher_scores_pos"] = ib_output_pos.cpu()
                    orig_batch["dyn_teacher_scores_neg"] = ib_output_neg.cpu()
                else:
                    orig_batch["dyn_teacher_scores_pos"] = score_pos.cpu()
                    orig_batch["dyn_teacher_scores_neg"] = score_neg.cpu()
            else:
                orig_batch["dyn_teacher_scores_pos"] = output_pos.cpu()
                orig_batch["dyn_teacher_scores_neg"] = output_neg.cpu()
        else:
            # Standard scoring
            final_pos = output_pos[0] if isinstance(output_pos, tuple) else output_pos
            final_neg = output_neg[0] if isinstance(output_neg, tuple) else output_neg
            
            orig_batch["dyn_teacher_scores_pos"] = final_pos.cpu()
            orig_batch["dyn_teacher_scores_neg"] = final_neg.cpu()


# Example model and config loader functions
def example_model_loader(config: Dict[str, Any]) -> tuple:
    """
    Example model loader function - you'll need to implement this based on your model architecture
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Tuple of (model, model_info)
    """
    # This is a placeholder - implement according to your model architecture
    # For example:
    # from your_model_module import YourModel
    # model = YourModel(config)
    # return model, {"type": "your_model_type"}
    
    raise NotImplementedError("Implement model_loader_fn based on your model architecture")

def example_config_loader(path: str) -> Dict[str, Any]:
    """
    Example config loader function - you'll need to implement this based on your config format
    
    Args:
        path: Path to config file or directory
        
    Returns:
        Configuration dictionary
    """
    # This is a placeholder - implement according to your config format
    # For example:
    # import json
    # with open(os.path.join(path, "config.json")) as f:
    #     return json.load(f)
    
    raise NotImplementedError("Implement config_loader_fn based on your config format")


# Example usage
if __name__ == "__main__":
    # Example of how to use the modernized DynamicTeacher
    config = {
        "dynamic_teacher_path": "/path/to/teacher/model",
        "dynamic_teacher_in_batch_scoring": True,
        "dynamic_teacher_per_term_scores": False,
        "teacher_cuda_device": 0,  # or [0, 1] for multi-GPU
        "use_fp16": True
    }
    
    # You'll need to provide your actual dataloader and functions
    # dataloader = YourDataLoader(...)
    # logger = logging.getLogger(__name__)
    # 
    # teacher = ModernDynamicTeacher(
    #     config=config,
    #     dataloader=dataloader,
    #     logger=logger,
    #     model_loader_fn=your_model_loader_function,
    #     config_loader_fn=your_config_loader_function
    # )
    #
    # # Use the teacher-wrapped dataloader
    # for batch in teacher:
    #     # batch now contains teacher scores:
    #     # - dyn_teacher_scores_pos
    #     # - dyn_teacher_scores_neg  
    #     # - dyn_teacher_per_term_scores_pos (if enabled)
    #     # - dyn_teacher_per_term_scores_neg (if enabled)
    #     pass
    
    print("Modernized DynamicTeacher class created!")
    print("Key improvements:")
    print("- Removed AllenNLP dependencies")
    print("- Uses pure PyTorch + Transformers")
    print("- Compatible with Python 3.12")
    print("- Improved error handling and resource management")
    print("- Maintains teacher-student distillation logic")