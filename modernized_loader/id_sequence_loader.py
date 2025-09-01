from typing import Dict, List, Union, Optional, Any
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import logging

# Using the same base components from the previous replacement
class ConfigurationError(Exception):
    """Simple replacement for AllenNLP's ConfigurationError"""
    pass

def cached_path(file_path: str) -> str:
    """Simple replacement for AllenNLP's cached_path - just returns the path"""
    return file_path

class Token:
    """Simple token representation"""
    def __init__(self, text: str, text_id: int = None):
        self.text = text
        self.text_id = text_id
        self.idx_start = None
        self.idx_end = None

class Tokenizer:
    """Base tokenizer class"""
    def tokenize(self, text: str) -> List[Token]:
        raise NotImplementedError

class TokenIndexer:
    """Base token indexer class"""
    pass

class TextField:
    """Simple text field implementation"""
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self._token_indexers = None

class MetadataField:
    """Simple metadata field implementation"""
    def __init__(self, metadata: Any):
        self.metadata = metadata

class BlingFireTokenizer(Tokenizer):
    """Basic tokenizer using bling fire library"""
    def __init__(self):
        try:
            from blingfire import text_to_words
            self.text_to_words = text_to_words
        except ImportError:
            # Fallback to simple whitespace tokenization
            self.text_to_words = lambda x: x.lower()
    
    def tokenize(self, sentence: str) -> List[Token]:
        if hasattr(self, 'text_to_words'):
            return [Token(t) for t in self.text_to_words(sentence).split()]
        else:
            return [Token(t) for t in sentence.lower().split()]

class PatchedTransformerTextField:
    """Replacement for transformer text field"""
    def __init__(self, **kwargs):
        self.data = kwargs

class FastTransformerTokenizer:
    """Fast transformer tokenizer wrapper"""
    def __init__(self, model_name: str):
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._max_length = 512
        except ImportError:
            raise ImportError("transformers library required for FastTransformerTokenizer")
    
    def tokenize(self, text: str, max_length: int = None) -> Dict[str, torch.Tensor]:
        if max_length is None:
            max_length = self._max_length
        
        encoded = self._tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Remove batch dimension
        return {k: v.squeeze(0) for k, v in encoded.items()}

class IdSequenceDataset(Dataset):
    """
    PyTorch Dataset for reading single sequence files
    
    Expected format: <sequence_id>\t<sequence_string>
    
    Parameters
    ----------
    file_path : str
        Path to the TSV file
    tokenizer : Union[Tokenizer, FastTransformerTokenizer], optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens
    token_indexers : Dict[str, TokenIndexer], optional
        Indexers used to define input token representations
    max_seq_length : int
        Maximum sequence length
    min_seq_length : int
        Minimum sequence length
    sequence_type : str
        Type of sequence ("doc" or "query")
    """
    
    def __init__(self,
                 file_path: str,
                 tokenizer: Union[Tokenizer, FastTransformerTokenizer] = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_seq_length: int = -1,
                 min_seq_length: int = -1,
                 sequence_type: str = "doc"):
        
        self.file_path = file_path
        self._tokenizer = tokenizer or BlingFireTokenizer()
        self._token_indexers = token_indexers
        
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.sequence_type = sequence_type
        
        # Determine token type
        if isinstance(tokenizer, FastTransformerTokenizer):
            self.token_type = "huggingface"
        else:
            self.token_type = "emb"
            self.padding_value = Token(text="@@PADDING@@", text_id=0)
            self.cls_value = Token(text="@@UNKNOWN@@", text_id=1)
        
        # Load data
        self.data = self._read_file()
    
    def _read_file(self) -> List[Dict[str, str]]:
        """Read and parse the TSV file"""
        data = []
        
        with open(cached_path(self.file_path), "r", encoding="utf8") as data_file:
            for line_num, line in enumerate(data_file, 1):
                line = line.strip("\n")
                
                if not line:
                    continue
                
                line_parts = line.split('\t')
                if len(line_parts) != 2:
                    raise ConfigurationError(
                        f"Invalid line format at line {line_num}, you have too many or too little columns "
                        f"in the line - maybe clean excess tabs from the text? Line: {line}"
                    )
                
                seq_id, seq_text = line_parts
                data.append({
                    'seq_id': seq_id,
                    'seq_text': seq_text
                })
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single item from the dataset"""
        item = self.data[idx]
        return self.text_to_instance(item['seq_id'], item['seq_text'])
    
    def text_to_instance(self, seq_id: str, seq_text: str) -> Dict[str, Any]:
        """Convert text to instance format"""
        
        result = {
            "seq_id": seq_id
        }
        
        if self.token_type == "huggingface":
            seq_tokenized = self._tokenizer.tokenize(seq_text, max_length=self.max_seq_length)
            result["seq_tokens"] = seq_tokenized
        else:
            seq_tokenized = self._tokenizer.tokenize(seq_text)
            
            if self.max_seq_length > -1:
                seq_tokenized = seq_tokenized[:self.max_seq_length]
            if self.min_seq_length > -1 and len(seq_tokenized) < self.min_seq_length:
                seq_tokenized = seq_tokenized + [self.padding_value] * (self.min_seq_length - len(seq_tokenized))
            
            result["seq_tokens"] = seq_tokenized
        
        return result

def pytorch_single_sequence_loader(model_config, run_config, input_file, sequence_type, force_exact_batch_size=False):
    """
    Load examples from a .tsv file in the single sequence format: id<tab>text
    
    Using PyTorch's DataLoader instead of AllenNLP's multiprocess loader
    
    Parameters
    ----------
    model_config : dict
        Model configuration containing tokenizer settings
    run_config : dict
        Run configuration containing batch sizes and other parameters
    input_file : str
        Path to the input TSV file
    sequence_type : str
        Type of sequence ("query" or "doc")
    force_exact_batch_size : bool
        Whether to force exact batch sizes
    """
    
    # Determine sequence-specific parameters
    if sequence_type == "query":
        max_length = model_config["max_query_length"]
        min_length = model_config["min_query_length"]
        batch_size = run_config["query_batch_size"]
    else:  # doc
        max_length = model_config["max_doc_length"]
        min_length = model_config["min_doc_length"]
        batch_size = run_config["collection_batch_size"]
    
    # Get tokenizer and indexers
    _tokenizer, _token_indexers = _get_indexer_pytorch(model_config, max_length)
    
    # Create dataset
    dataset = IdSequenceDataset(
        file_path=input_file,
        tokenizer=_tokenizer,
        token_indexers=_token_indexers,
        max_seq_length=max_length,
        min_seq_length=min_length,
        sequence_type=sequence_type
    )
    
    def collate_fn(batch):
        """Custom collate function to handle batching"""
        if not batch:
            return {}
        
        result = {}
        for key in batch[0].keys():
            if key == "seq_id":
                result[key] = [item[key] for item in batch]
            elif isinstance(batch[0][key], dict):  # Huggingface tokenizer output
                result[key] = {}
                for subkey in batch[0][key].keys():
                    result[key][subkey] = torch.stack([item[key][subkey] for item in batch])
            elif isinstance(batch[0][key], list):  # Token lists
                result[key] = [item[key] for item in batch]
            else:
                result[key] = [item[key] for item in batch]
        
        return result
    
    # Create DataLoader with appropriate settings
    if force_exact_batch_size:
        # Use exact batch size
        loader = DataLoader(
            dataset,
            batch_size=int(batch_size),
            num_workers=run_config.get("dataloader_num_workers", 0),
            collate_fn=collate_fn,
            shuffle=False,
            drop_last=False  # Set to True if you need exact batch sizes
        )
    else:
        # Use dynamic batching (approximate token-based batching)
        # For simplicity, we'll use regular batching but calculate a reasonable batch size
        max_tokens = int(batch_size) * max_length
        estimated_batch_size = max(1, max_tokens // max_length)
        
        loader = DataLoader(
            dataset,
            batch_size=estimated_batch_size,
            num_workers=run_config.get("dataloader_num_workers", 0),
            collate_fn=collate_fn,
            shuffle=False,
            drop_last=False
        )
    
    return loader

def _get_indexer_pytorch(model_config, max_length):
    """Get tokenizer and indexers for PyTorch version"""
    # Default values
    _tokenizer = BlingFireTokenizer()
    _token_indexers = None
    
    if "bert_pretrained_model" in model_config:
        model = model_config["bert_pretrained_model"]
        _tokenizer = FastTransformerTokenizer(model)
        _token_indexers = None
    
    return _tokenizer, _token_indexers

# Advanced batching strategy for better token utilization
class TokenBatchSampler:
    """
    A batch sampler that tries to group sequences by length for more efficient padding
    """
    def __init__(self, dataset, max_tokens, batch_size=None, shuffle=False):
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Get sequence lengths
        self.lengths = []
        for i in range(len(dataset)):
            item = dataset[i]
            if isinstance(item["seq_tokens"], dict):  # Huggingface format
                length = item["seq_tokens"]["attention_mask"].sum().item()
            else:  # Token list format
                length = len(item["seq_tokens"])
            self.lengths.append((i, length))
        
        if shuffle:
            import random
            random.shuffle(self.lengths)
        else:
            # Sort by length for better batching
            self.lengths.sort(key=lambda x: x[1])
    
    def __iter__(self):
        batch = []
        current_tokens = 0
        max_length_in_batch = 0
        
        for idx, length in self.lengths:
            # Check if adding this sample would exceed our limits
            new_max_length = max(max_length_in_batch, length)
            new_total_tokens = new_max_length * (len(batch) + 1)
            
            if (batch and 
                (new_total_tokens > self.max_tokens or 
                 (self.batch_size and len(batch) >= self.batch_size))):
                yield batch
                batch = []
                current_tokens = 0
                max_length_in_batch = 0
            
            batch.append(idx)
            max_length_in_batch = max(max_length_in_batch, length)
        
        if batch:
            yield batch
    
    def __len__(self):
        # Estimate number of batches
        total_tokens = sum(length for _, length in self.lengths)
        return max(1, total_tokens // self.max_tokens)

def pytorch_single_sequence_loader_advanced(model_config, run_config, input_file, sequence_type, force_exact_batch_size=False):
    """
    Advanced version with better token-based batching
    """
    # Determine sequence-specific parameters
    if sequence_type == "query":
        max_length = model_config["max_query_length"]
        min_length = model_config["min_query_length"]
        batch_size = run_config["query_batch_size"]
    else:  # doc
        max_length = model_config["max_doc_length"]
        min_length = model_config["min_doc_length"]
        batch_size = run_config["collection_batch_size"]
    
    # Get tokenizer and indexers
    _tokenizer, _token_indexers = _get_indexer_pytorch(model_config, max_length)
    
    # Create dataset
    dataset = IdSequenceDataset(
        file_path=input_file,
        tokenizer=_tokenizer,
        token_indexers=_token_indexers,
        max_seq_length=max_length,
        min_seq_length=min_length,
        sequence_type=sequence_type
    )
    
    def collate_fn(batch):
        """Custom collate function with efficient padding"""
        if not batch:
            return {}
        
        # Get actual samples
        samples = [dataset[i] for i in batch]
        
        result = {}
        for key in samples[0].keys():
            if key == "seq_id":
                result[key] = [item[key] for item in samples]
            elif isinstance(samples[0][key], dict):  # Huggingface tokenizer output
                result[key] = {}
                for subkey in samples[0][key].keys():
                    result[key][subkey] = torch.stack([item[key][subkey] for item in samples])
            elif isinstance(samples[0][key], list):  # Token lists
                result[key] = [item[key] for item in samples]
            else:
                result[key] = [item[key] for item in samples]
        
        return result
    
    if force_exact_batch_size:
        # Use simple exact batch size
        loader = DataLoader(
            dataset,
            batch_size=int(batch_size),
            num_workers=run_config.get("dataloader_num_workers", 0),
            collate_fn=collate_fn,
            shuffle=False,
            drop_last=False
        )
    else:
        # Use token-based batching
        max_tokens = int(batch_size) * max_length
        batch_sampler = TokenBatchSampler(
            dataset, 
            max_tokens=max_tokens,
            batch_size=int(batch_size) if force_exact_batch_size else None,
            shuffle=False
        )
        
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=run_config.get("dataloader_num_workers", 0),
            collate_fn=collate_fn
        )
    
    return loader

# Example usage:
if __name__ == "__main__":
    # Example configuration
    model_config = {
        "bert_pretrained_model": "bert-base-uncased",
        "max_query_length": 64,
        "min_query_length": -1,
        "max_doc_length": 512,
        "min_doc_length": -1
    }
    
    run_config = {
        "query_batch_size": 32,
        "collection_batch_size": 16,
        "dataloader_num_workers": 4
    }
    
    # Create loader for queries
    query_loader = pytorch_single_sequence_loader(
        model_config, 
        run_config, 
        "queries.tsv",  # Replace with your actual file
        "query"
    )
    
    # Create loader for documents
    doc_loader = pytorch_single_sequence_loader(
        model_config, 
        run_config, 
        "documents.tsv",  # Replace with your actual file
        "doc"
    )
    
    print("Loaders created successfully!")
    
    # Example of iterating through data
    for batch in query_loader:
        print(f"Query batch with {len(batch['seq_id'])} samples")
        if isinstance(batch['seq_tokens'], dict):
            print(f"Token tensor shape: {batch['seq_tokens']['input_ids'].shape}")
        break  # Just show first batch