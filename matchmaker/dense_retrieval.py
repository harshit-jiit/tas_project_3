# Matchmaker Dense Retrieval 
# -------------------------------
# Conduct 3 phases of dense retrieval: encoding, indexing, search
# Only in batch form, not really meant for production use of a search engine
#
# - Needs a trained model (via train.py)
# - Measures efficiency & effectiveness on 1 collection + multiple query sets (start a new experiment for another collection)
# - Allows to start new experiment from each of the 3 steps via modes:
#
#         mode                     config-requirement 
#
#      1) encode+index+search   -> trained_model folder path
#      2) index+search          -> continue_folder folder path pointing to an experiment started with 1)
#      3) search                -> continue_folder folder path pointing to an experiment started with 2)
#
# - We can do a lot of hyperparameter studies starting from each step, or just run through a full pass once

import argparse
import copy
import os
import glob
from timeit import default_timer
import sys
sys.path.append(os.getcwd())
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # needed because of the scann library
from transformers import logging
logging.set_verbosity_warning()
import onnxruntime
from matchmaker.utils.onnx_helper import *
import torch
import numpy
import random
from collections import defaultdict
from typing import Union, Dict, List, Any

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
    

from matchmaker.models.all import get_model, get_word_embedder
from matchmaker.modules.indexing_heads import *

from matchmaker.utils.utils import *
from matchmaker.utils.config import *
from modernized_loader.transformer_tokenizer import FastTransformerTokenizer
from matchmaker.utils.performance_monitor import * 

from matchmaker.eval import *
from matchmaker.utils.core_metrics import *
from matchmaker.retrieval.faiss_indices import *
from torch.utils.data import Dataset, DataLoader
from rich.console import Console
from rich.live import Live
console = Console()

MODE_ALL = "encode+index+search"
MODE_START_INDEX = "index+search"
MODE_START_SEARCH = "search"

class IdSequenceDataset(Dataset):
    """
    PyTorch Dataset for loading sequences from a TSV file: id<tab>text
    """
    def __init__(self, data_path, tokenizer, max_seq_length=-1, min_seq_length=-1, sequence_type="doc"):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.sequence_type = sequence_type
        self.data = self._load_data(data_path)

    def _load_data(self, data_path):
        data = []
        with open(data_path, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    raise ValueError(f"Invalid line format: {line}")
                seq_id, seq_text = parts
                data.append((seq_id, seq_text))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq_id, seq_text = self.data[idx]
        tokenized = self.tokenizer.tokenize(seq_text, max_length=self.max_seq_length)
        # For huggingface, tokenized is a dict with input_ids, attention_mask, etc.
        # Add seq_id
        tokenized["seq_id"] = seq_id
        return tokenized

def collate_fn(batch, max_length, pad_token_id):
    """
    Custom collate function to pad batches dynamically.
    Assumes batch items are dicts from tokenizer.
    """
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item.get("attention_mask", [1] * len(item["input_ids"])) for item in batch]
    seq_ids = [item["seq_id"] for item in batch]

    max_len = min(max(len(ids) for ids in input_ids), max_length)

    padded_input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    padded_attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

    for i, (ids, mask) in enumerate(zip(input_ids, attention_mask)):
        length = min(len(ids), max_len)
        padded_input_ids[i, :length] = torch.as_tensor(ids[:length])
        padded_attention_mask[i, :length] = torch.as_tensor(mask[:length])

    return {
        "seq_tokens": {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            # Add token_type_ids if needed
        },
        "seq_id": seq_ids
    }

def single_sequence_loader(model_config, run_config, input_file, sequence_type, force_exact_batch_size=False):
    """
    Load examples from a .tsv file in the single sequence format: id<tab>text
    Using PyTorch DataLoader.
    """
    if sequence_type == "query":
        max_length = model_config["max_query_length"]
        min_length = model_config["min_query_length"]
        batch_size = run_config["query_batch_size"]
    else:  # doc
        max_length = model_config["max_doc_length"]
        min_length = model_config["min_doc_length"]
        batch_size = run_config["collection_batch_size"]

    tokenizer, _, _ = _get_indexer(model_config)  # Reuse your _get_indexer, but we only need tokenizer

    dataset = IdSequenceDataset(input_file, tokenizer=tokenizer, max_seq_length=max_length, min_seq_length=min_length, sequence_type=sequence_type)

    # For huggingface tokenizer, get pad_token_id
    if isinstance(tokenizer, FastTransformerTokenizer):
        hf_tokenizer = AutoTokenizer.from_pretrained(model_config["bert_pretrained_model"])
        pad_token_id = hf_tokenizer.pad_token_id
    else:
        pad_token_id = 0  # Default, adjust if needed

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Assuming inference, no shuffle
        num_workers=run_config["dataloader_num_workers"],
        collate_fn=lambda b: collate_fn(b, max_length, pad_token_id),
        pin_memory=True
    )
    return loader

if __name__ == "__main__":


    config={
                "continue_folder": "/workspace/2404170001/experiments/neural_ir_experiment/2025-09-02_2011_neural_ir_experiment",
                "expirement_base_path": "/workspace/2404170001/experiments/neural_ir_experiment/2025-09-02_2011_neural_ir_experiment",
                "trained_model": "/workspace/2404170001/experiments/neural_ir_experiment/2025-09-02_2011_neural_ir_experiment",
                "collection_tsv": "/workspace/2404170001/inputs/collection.tsv",
                "collection_batch_size": 256,
                "query_batch_size": 32,
                "onnx_use_inference": False,
                "use_fp16": False,
                "dataloader_num_workers": 0,
                "query_sets": {
                    "msmarco-dev": {
                    "queries_tsv": "/workspace/2404170001/validation_test_split/queries_test.tsv",
                    "qrels": "/workspace/2404170001/validation_test_split/qrels_test.tsv",
                    "binarization_point": 1,
                    "top_n": 1000
                    }
                },
                "token_block_size": 1000000,
                "token_dim": 128,
                "token_dtype": "float32",
                "faiss_index_type": "full",
                "faiss_use_gpu": True,
                "faiss_hnsw_graph_neighbors": 128,
                "faiss_hnsw_efConstruction": 128,
                "faiss_hnsw_efSearch": 128,
                "faiss_ivf_search_probe_count": 500,
                "faiss_ivf_list_count": 20000
                }
    run_folder = r'/workspace/2404170001/experiments/neural_ir_experiment/2025-09-02_2011_neural_ir_experiment'
    # mode = "encode+index+search"
    mode = "index+search"  # Instead of "encode+index+search"

    if mode == MODE_ALL:
        encode_config = config
        index_config = config
        model_config = get_config_single(config["trained_model"])
        print_hello({**model_config, **config}, run_folder, "[Dense Retrieval] Encode & Index & Search",
                    show_settings=["Model","Trained Checkpoint","Index","Collection Batch Size","Query Batch Size","Use ONNX Runtime"])

    elif mode == MODE_START_INDEX:
        if "continue_folder" not in config: raise Exception("continue_folder must be set in config")

        encode_folder = config["continue_folder"]
        encode_config = config
        model_config = get_config_single(encode_config["trained_model"])
        index_config = config
        print_hello({**model_config, **config,**{"trained_model":encode_config["trained_model"]}}, run_folder, "[Dense Retrieval] Index & Search",
                    show_settings=["Model","Trained Checkpoint","Index","Query Batch Size","Use ONNX Runtime"])

    elif mode == MODE_START_SEARCH:
        if "continue_folder" not in config: raise Exception("continue_folder must be set in config")
        
        index_folder = config["continue_folder"]
        index_config  = config
        encode_folder = index_config["continue_folder"] if "continue_folder" in index_config else index_folder
        encode_config= get_config_single(encode_folder)
        model_config = get_config_single(encode_config["trained_model"])
        print_hello({**model_config, **config,**{"trained_model":encode_config["trained_model"]}}, run_folder, "[Dense Retrieval] Search",
                    show_settings=["Model","Trained Checkpoint","Index","Query Batch Size","Use ONNX Runtime"])

    else:
        raise Exception("mode not supported")

    use_onnx = config["onnx_use_inference"]
    if use_onnx: # prevent errors if onnx is not properly installed (see readme for setup ionstructions)
        import onnxruntime
        from matchmaker.utils.onnx_helper import *

    logger = get_logger_to_file(run_folder, "main")
    logger.info("Running: %s", str(sys.argv))

    torch.manual_seed(model_config["random_seed"])
    numpy.random.seed(model_config["random_seed"])
    random.seed(model_config["random_seed"])
    logger.info("Torch seed: %i ",torch.initial_seed())

    # hardcode gpu usage
    cuda_device = 0 # main cuda device
    perf_monitor = PerformanceMonitor.get()
    perf_monitor.start_block("startup")
    
    #
    # create and load model instance
    # -------------------------------
    
    word_embedder, padding_idx = get_word_embedder(model_config)
    model, encoder_type = get_model(model_config,word_embedder,padding_idx)
    # model = build_model(model,encoder_type,word_embedder,model_config)
    
    if model_config.get("model_checkpoint_from_huggingface",False):
        model.from_pretrained(encode_config["trained_model"])
        console.log("[Startup]","Trained model loaded from huggingface")
    else:
        model_path = os.path.join(encode_config["trained_model"], "best-model.pytorch-state-dict")
        load_result = model.load_state_dict(torch.load(model_path),strict=False)
        logger.info('Warmstart init model from:  %s', model_path)
        logger.info(load_result)
        console.log("[Startup]","Trained model loaded locally; result:",load_result)


    #
    # setup heads wrapping the model for indexing & searching
    #
    if mode == MODE_ALL:
        model_indexer = CollectionIndexerHead(model, use_fp16=False if use_onnx else model_config["use_fp16"]).cuda()
        model_indexer.eval()

        if use_onnx:
            console.log("[Startup]","Using ONNX, converting & optimizing indexer ... ")
            convert_and_optimize(model_indexer, os.path.join(run_folder,"indexer-model.onnx"),model_config["use_fp16"])
            del model_indexer
            onnx_indexer = onnxruntime.InferenceSession(os.path.join(run_folder,"indexer-model.onnx"),providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    model_searcher = QuerySearcherHead(model, use_fp16=False if use_onnx else model_config["use_fp16"]).cuda()
    model_searcher.eval()

    if use_onnx:
        console.log("[Startup]","Using ONNX, converting & optimizing searcher ... ")
        convert_and_optimize(model_searcher, os.path.join(run_folder,"searcher-model.onnx"),model_config["use_fp16"])
        del model_searcher

    logger.info('Model %s total parameters: %s', model_config["model"], sum(p.numel() for p in model.parameters() if p.requires_grad))
    logger.info('Network: %s', model)
    del model
   
    #
    # setup multi-gpu, todo fix to forward only calls
    #
    if torch.cuda.device_count() > 1 and not use_onnx:
        console.log("Let's use", torch.cuda.device_count(), "GPUs!")
        if mode == MODE_ALL:
            model_indexer = torch.nn.DataParallel(model_indexer)
            model_indexer.eval()
        if config["query_batch_size"] > 1:
            model_searcher = torch.nn.DataParallel(model_searcher)
            model_searcher.eval()
        else:
            console.log("[Startup] Info: Using single GPU for search, as batch_size is set to 1")
    elif torch.cuda.device_count() > 1 and use_onnx:
        console.log("[Startup] Info: ONNX currently only uses 1 GPU")

    perf_monitor.set_gpu_info(torch.cuda.device_count(),torch.cuda.get_device_name())
    perf_monitor.stop_block("startup")


    def _get_indexer(model):
        _tokenizer = FastTransformerTokenizer(model["bert_pretrained_model"])
        return _tokenizer, None, None

    try:

        #
        # 1) Encode 
        # -------------------------------

        token_base_size = config["token_block_size"]
        token_dimensions = config["token_dim"]

        if mode == MODE_ALL:

            console.log("[Encoding]","Encoding collection from: ",config["collection_tsv"])

            doc_infos = {}
            seq_ids = []

            token_base_number=0
            token_base = np.memmap(os.path.join(run_folder,"token_reps_"+str(token_base_number)+".npy"), dtype=numpy.dtype(config["token_dtype"]),
                                   mode="w+", shape=(token_base_size,token_dimensions))

            current_ids = np.ndarray(shape=(token_base_size), dtype = 'int64') 
            id_mapping = []
            token_insert_index=0
            storage = []
            storage_filled_to_index = []

            input_loader = single_sequence_loader(model_config, config, config["collection_tsv"], sequence_type="doc")
            perf_monitor.start_block("encode")
            start_time = default_timer()
            #import pprofile
            #prof = pprofile.Profile()

            with torch.no_grad(), Live("[bold magenta]           Loading...",console=console,auto_refresh=False) as status:#,prof():
                batch_number=0
                sequence_number=0

                for batch in input_loader:
                    
                    if use_onnx:
                        output = onnx_indexer.run(None, {'input_ids': batch["seq_tokens"]['input_ids'].numpy(),
                                                         'attention_mask': batch["seq_tokens"]['attention_mask'].to(torch.int64).numpy()})[0]
                    else:
                        batch = move_to_device(copy.deepcopy(batch), cuda_device)
                        output = model_indexer.forward(batch["seq_tokens"])
                        output = output.cpu().numpy()  # get the output back to the cpu - in one piece

                    # compare ONNX Runtime and PyTorch results
                    #np.testing.assert_allclose(output, ort_outs[0], rtol=1e-03, atol=1e-05)

                    for sample_i, seq_id in enumerate(batch["seq_id"]):  # operate on cpu memory

                        # assuming either 1 vec or 1-n
                        # if 1-n remove 0 vectors as padding (colbert,tk,tkl models)
                        current_reps = output[sample_i]
                        dim_count = len(current_reps.shape)
                        if dim_count == 2:
                            current_reps = current_reps[np.abs(current_reps).sum(-1) > 0,:] # a bit dicey, but i guess we are fine because what is the prob. of something being 0 without being padding

                        vec_count = 1 if dim_count == 1 else current_reps.shape[0]

                        if token_insert_index + vec_count > token_base_size:
                            storage.append(token_base[:token_insert_index])
                            id_mapping.append(current_ids[:token_insert_index])
                            current_ids = np.ndarray(shape=(token_base_size), dtype = 'int64') 
                            storage_filled_to_index.append(token_insert_index)

                            token_base_number+=1
                            token_insert_index=0
                            token_base = np.memmap(os.path.join(run_folder,"token_reps_"+str(token_base_number)+".npy"), dtype=numpy.dtype(config["token_dtype"]),
                                                   mode="w+", shape=(token_base_size,token_dimensions))

                        start_index = token_insert_index
                        token_insert_index = token_insert_index + vec_count
                        token_base[start_index:token_insert_index] = current_reps
                        current_ids[start_index:token_insert_index] = len(seq_ids)

                        doc_infos[seq_id] = (token_base_number,start_index,token_insert_index)
                        seq_ids.append(seq_id)

                    batch_number += 1
                    sequence_number += len(batch["seq_id"])
                    if batch_number % 10 == 0: status.update("[bold magenta]           Progress ... Batch No.: "+str(batch_number)+" | Sequence No.: "+str(sequence_number) +" | Seq. / second: " + \
                                                  "{:.2f}".format(sequence_number/(default_timer() - start_time)), refresh=True)

            #prof.print_stats()

            # save last token reps
            storage.append(token_base[:token_insert_index])
            id_mapping.append(current_ids[:token_insert_index])
            storage_filled_to_index.append(token_insert_index)

            saveCompressed(os.path.join(run_folder,"doc_infos.npz"),doc_infos=doc_infos,id_mapping=numpy.array(id_mapping, dtype=object),
                                                                    seq_ids=seq_ids,storage_filled_to_index=storage_filled_to_index)
            if not use_onnx:
                perf_monitor.log_unique_value("encoding_gpu_mem",str(torch.cuda.memory_allocated()/float(1e9)) + " GB")
                perf_monitor.log_unique_value("encoding_gpu_mem_max",str(torch.cuda.max_memory_allocated()/float(1e9)) + " GB")
            perf_monitor.log_unique_value("encoded_size_on_disk",str(sum(os.path.getsize(f) for f in glob.glob(os.path.join(run_folder,"token_reps_*")))/float(1e9)) + " GB")

            perf_monitor.stop_block("encode",len(seq_ids))

        #
        # skip encoding 
        #
        else:
            console.log("[Encoding]","Skipping encoding; loading collection vectors & info from: ",encode_folder)
            dfs = numpy.load(os.path.join(encode_folder,"doc_infos.npz"),allow_pickle=True)
            doc_infos=dfs.get("doc_infos")[()]
            id_mapping=dfs.get("id_mapping")[()]
            seq_ids=dfs.get("seq_ids")[()]
            storage_filled_to_index=dfs.get("storage_filled_to_index")[()]

            storage = []
            for f in range(0,len(glob.glob(os.path.join(encode_folder,"token_reps_*")))):
                storage.append(np.memmap(os.path.join(encode_folder,"token_reps_"+str(f)+".npy"), dtype=numpy.dtype(encode_config["token_dtype"]),
                                                   mode="r", shape=(token_base_size,token_dimensions))[:storage_filled_to_index[f]])

        #
        # 2) Nearest neighbor indexing
        # -------------------------

        if index_config["faiss_index_type"] == "ondisk_sharding":
            indexer = FaissShardedOnDiskIdIndexer(index_config)
        elif index_config["faiss_index_type"] == "full":
            indexer = FaissIdIndexer(index_config)
        elif index_config["faiss_index_type"] == "ivf":
            indexer = FaissIVFIndexer(index_config) # todo
        elif index_config["faiss_index_type"] == "hnsw":
            indexer = FaissHNSWIndexer(index_config)
        elif index_config["faiss_index_type"] == "scann":
            from matchmaker.retrieval.scann_index import ScaNNIndexer # import here, because it only works on linux 
            indexer = ScaNNIndexer(index_config)
        else:
            raise Exception("faiss_index_type not supported")
        
        # we don't save the full index, but rebuilt it every time (just loading the vectors basically)
        if mode != MODE_START_SEARCH or index_config["faiss_index_type"] == "full":

            perf_monitor.start_block("indexing")

            indexer.prepare(storage)
            indexer.index(id_mapping, storage)
            
            if index_config["faiss_index_type"] == "scann": # scann needs a folder (already created)
                index_path = os.path.join(run_folder,"scann-index")
                os.makedirs(index_path)
                indexer.save(index_path)
                index_size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(run_folder,"scann-index","*")))
                perf_monitor.log_unique_value("index_size_on_disk",str(index_size/float(1e9)) + " GB")
                perf_monitor.log_unique_value("index_size_on_disk_per_entry",str(index_size/len(seq_ids)) + " bytes")

            elif index_config["faiss_index_type"] != "full":
                indexer.save(os.path.join(run_folder,"faiss.index"))
                perf_monitor.log_unique_value("faiss_index_size_on_disk",str(os.path.getsize(os.path.join(run_folder,"faiss.index"))/float(1e9)) + " GB")
                perf_monitor.log_unique_value("faiss_index_size_on_disk_per_entry",str(os.path.getsize(os.path.join(run_folder,"faiss.index"))/len(seq_ids)) + " bytes")

            perf_monitor.stop_block("indexing")

        else:
            console.log("[Indexing]","loading existing index from: ",index_folder)
            if index_config["faiss_index_type"] == "scann":
                indexer.load(os.path.join(index_folder,"scann-index"))
            else:
                indexer.load(os.path.join(index_folder,"faiss.index"),config)


        #
        # 3) Search
        # -------------------------
        if use_onnx:
            if mode == MODE_ALL: del onnx_indexer
            onnx_searcher = onnxruntime.InferenceSession(os.path.join(run_folder,"searcher-model.onnx"),providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        perf_monitor.start_block("search_total")

        for test_name,test_config in config["query_sets"].items():
            console.log("[Search]","Start retrieval for:", test_config["queries_tsv"])
            
            input_loader = single_sequence_loader(model_config,config,test_config["queries_tsv"], sequence_type="query", force_exact_batch_size=True)
            
            validation_results = defaultdict(list)
            times_query_encode = []
            times_faiss = []
            times_aggregate = []
            with torch.no_grad(), Live("[bold magenta]           Loading...",console=console,auto_refresh=False) as status:
                i=0

                for batch_orig in input_loader:

                    batch_size = len(batch_orig["seq_id"])
            
                    perf_monitor.start_block("search_query_encode")

                    if use_onnx:
                        output = onnx_searcher.run(None, {'input_ids': batch_orig["seq_tokens"]['input_ids'].numpy(),
                                                         'attention_mask': batch_orig["seq_tokens"]['attention_mask'].to(torch.int64).numpy()})[0]
                    else:
                        batch = move_to_device(copy.deepcopy(batch_orig), cuda_device)
                        output = model_searcher.forward(batch["seq_tokens"],search_type="encode")
                        output = output.cpu().numpy()  # get the output back to the cpu - in one piece

                    perf_monitor.stop_block("search_query_encode",batch_size)
                    perf_monitor.start_block("search_nn_lookup")
                    
                    res_scores, ids = indexer.search(output,test_config.get("index_hit_top_n",test_config["top_n"]))

                    perf_monitor.stop_block("search_nn_lookup",batch_size)
                    perf_monitor.start_block("search_aggregation")

                    for sample_i, query_id in enumerate(batch_orig["seq_id"]):  # operate on cpu memory
                        
                        if model_config["model"] == "ColBERT":
                            
                            current_ids_set = set(current_ids.flatten())

                            for t, s_idx in enumerate(current_ids_set):
                                #st_3b1  = time.time()
                                doc_inf = doc_infos[seq_ids[s_idx]]
                                doc_reps = storage[doc_inf[0]][doc_inf[1]:doc_inf[2],:]
                                doc_reps = torch.from_numpy(doc_reps).unsqueeze(0).float()
                                #st_3b_close.append(time.time()-st_3b1)
                                #st_3b1  = time.time()
                                output_scores = model_searcher.forward(curr_q, document_enc=doc_reps, search_type="aggregate", use_fp16=model_config["use_fp16"])
                                #output_scores = output_scores.cpu().numpy() 

                                validation_results[query_id].append((seq_ids[s_idx], float(output_scores)))

                        if model_config['model'] == 'maxP->bert_dot':
                            current_scores = res_scores[sample_i]
                            current_ids = ids[sample_i]

                            unique_ids = set()

                            for t, s_idx in enumerate(current_ids):
                                if s_idx in unique_ids:
                                    continue
                                unique_ids.add(s_idx)
                                validation_results[query_id].append((seq_ids[s_idx], float(current_scores[t])))
                                if len(validation_results[query_id]) == test_config["top_n"]:
                                    break
                                
                            if len(validation_results[query_id]) < test_config["top_n"]:
                                logger.info("too few results for: ",query_id,"got:",len(validation_results[query_id]))

                        else: # simple max. inner product aggregation 
                            # get collection seq ids 
                            current_scores = res_scores[sample_i]
                            current_ids = ids[sample_i]
                            collection_ids = []
                            for t, s_idx in enumerate(current_ids):
                                validation_results[query_id].append((seq_ids[s_idx], float(current_scores[t])))
                    
                    perf_monitor.stop_block("search_aggregation",batch_size)

                    i+=1
                    if i % 10 == 0: status.update("[bold magenta]           Progress ... Batch No.: "+str(i) +" Query No.: "+str(len(batch_orig["seq_id"])*i), refresh=True)
                
            save_sorted_results(validation_results, os.path.join(run_folder,test_name+"-output.txt"),until_rank=test_config["top_n"])

            if "qrels" in test_config:
                ranked_results = unrolled_to_ranked_result(validation_results)
                metrics = calculate_metrics_plain(ranked_results,load_qrels(test_config["qrels"]),test_config["binarization_point"])

                metric_file_path = os.path.join(run_folder, test_name+"-metrics.csv")
                save_fullmetrics_oneN(metric_file_path, metrics, -1, -1)
                console.log("[Search] Results for",test_name)
                print_metric_summary(metrics)

        perf_monitor.stop_block("search_total")
        perf_monitor.print_summary()
        perf_monitor.save_summary(os.path.join(run_folder,"efficiency_metrics.json"))

    except KeyboardInterrupt:
        logger.info('-' * 20)
        logger.info('Manual Stop!')
        console.log("Manual Stop! Bye :)")

    except Exception as e:
        logger.info('-' * 20)
        logger.exception('[train] Got exception: ')
        logger.info('Exiting from training early')
        console.log("[red]Exception! ",str(e))
        console.print_exception()

        exit(1)

    finally:

        # cleanup the onnx model, this will come back to bite me some day, but for now let's save storage space!
        if os.path.exists(os.path.join(run_folder,"indexer-model.onnx")):
            os.remove(os.path.join(run_folder,"indexer-model.onnx"))
        if os.path.exists(os.path.join(run_folder,"searcher-model.onnx")):
            os.remove(os.path.join(run_folder,"searcher-model.onnx"))
