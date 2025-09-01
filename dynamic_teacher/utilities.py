"""
Modern utility functions to replace AllenNLP dependencies
Compatible with Python 3.12 and latest PyTorch/Transformers
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig

# Modern replacements for AllenNLP functions

def move_to_device(
    obj: Union[torch.Tensor, Dict, List, Any], 
    device: Union[torch.device, int, str]
) -> Union[torch.Tensor, Dict, List, Any]:
    """
    Recursively move tensors in nested data structures to specified device
    
    Args:
        obj: Object containing tensors (can be nested dict/list/tuple)
        device: Target device
        
    Returns:
        Object with tensors moved to device
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        moved_items = [move_to_device(item, device) for item in obj]
        return type(obj)(moved_items)
    else:
        return obj

def get_model_device(model: nn.Module) -> torch.device:
    """
    Get the device of a model's parameters
    
    Args:
        model: PyTorch model
        
    Returns:
        Device where model parameters are located
    """
    return next(model.parameters()).device

def load_config_from_path(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file or directory containing config.json
    
    Args:
        config_path: Path to config file or directory
        
    Returns:
        Configuration dictionary
    """
    if os.path.isdir(config_path):
        config_file = os.path.join(config_path, "config.json")
    else:
        config_file = config_path
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_config_to_path(config: Dict[str, Any], config_path: str):
    """
    Save configuration to a JSON file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config file
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

class ModelWrapper:
    """
    Wrapper class to provide AllenNLP-style model interface
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.is_teacher_model = False
    
    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def eval(self):
        self.model.eval()
        return self
    
    def train(self, mode: bool = True):
        self.model.train(mode)
        return self
    
    def to(self, device):
        self.model.to(device)
        return self
    
    def cuda(self, device=None):
        self.model.cuda(device)
        return self
    
    def cpu(self):
        self.model.cpu()
        return self
    
    def state_dict(self):
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)
    
    def parameters(self):
        return self.model.parameters()
    
    def named_parameters(self):
        return self.model.named_parameters()

def get_huggingface_model(model_name: str, config: Optional[Dict] = None) -> Tuple[nn.Module, Dict]:
    """
    Load a Hugging Face model with configuration
    
    Args:
        model_name: Name of the Hugging Face model
        config: Optional additional configuration
        
    Returns:
        Tuple of (model, model_info)
    """
    try:
        hf_config = AutoConfig.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        model_info = {
            "model_type": hf_config.model_type,
            "hidden_size": getattr(hf_config, 'hidden_size', None),
            "num_attention_heads": getattr(hf_config, 'num_attention_heads', None),
            "num_hidden_layers": getattr(hf_config, 'num_hidden_layers', None),
        }
        
        return ModelWrapper(model, config or {}), model_info
        
    except Exception as e:
        raise ValueError(f"Failed to load Hugging Face model '{model_name}': {str(e)}")

def create_model_loader_fn(model_registry: Dict[str, callable]) -> callable:
    """
    Create a model loader function that can load different model types
    
    Args:
        model_registry: Dictionary mapping model types to loader functions
        
    Returns:
        Model loader function
    """
    def model_loader(config: Dict[str, Any]) -> Tuple[nn.Module, Dict]:
        model_type = config.get("model_type", "huggingface")
        
        if model_type in model_registry:
            return model_registry[model_type](config)
        elif model_type == "huggingface":
            model_name = config.get("model_name", config.get("bert_pretrained_model", "bert-base-uncased"))
            return get_huggingface_model(model_name, config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    return model_loader

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("matchmaker")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def calculate_model_size(model: nn.Module) -> Dict[str, int]:
    """
    Calculate model size statistics
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model size information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
    }

def get_device_info() -> Dict[str, Any]:
    """
    Get information about available devices
    
    Returns:
        Dictionary with device information
    """
    info = {
        "cpu_available": True,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": 0,
        "cuda_devices": [],
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    }
    
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_devices"] = [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total": torch.cuda.get_device_properties(i).total_memory,
                "memory_allocated": torch.cuda.memory_allocated(i),
                "memory_cached": torch.cuda.memory_reserved(i),
            }
            for i in range(torch.cuda.device_count())