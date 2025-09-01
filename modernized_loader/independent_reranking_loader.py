from typing import Dict, List, Union, Optional, Any
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import logging

# Simple replacements for AllenNLP components
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

class Instance:
    """Simple instance implementation"""
    def __init__(self, fields: Dict[str, Any]):
        self.fields = fields

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

# For transformer tokenizer compatibility
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

class IndependentReRankingDataset(Dataset):
    """
    PyTorch Dataset for reading re-ranking candidate sequences
    
    Expected format for each input line: <query_id>\t<doc_id>\t<query_sequence_string>\t<doc_sequence_string>
    
    Parameters
    ----------
    file_path : str
        Path to the TSV file
    tokenizer : Tokenizer, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens
    token_indexers : Dict[str, TokenIndexer], optional
        Indexers used to define input token representations
    max_doc_length : int
        Maximum document length
    max_query_length : int
        Maximum query length
    min_doc_length : int
        Minimum document length
    min_query_length : int
        Minimum query length
    query_augment_mask_number : int
        Number of mask tokens to add for query augmentation
    train_qa_spans : bool
        Whether to train QA spans
    """
    
    def __init__(self,
                 file_path: str,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_doc_length: int = -1,
                 max_query_length: int = -1,
                 min_doc_length: int = -1,
                 min_query_length: int = -1,
                 query_augment_mask_number: int = -1,
                 train_qa_spans: bool = False):
        
        self.file_path = file_path
        self._tokenizer = tokenizer or BlingFireTokenizer()
        self._token_indexers = token_indexers
        
        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length
        self.min_doc_length = min_doc_length
        self.min_query_length = min_query_length
        
        # Determine token type
        if hasattr(self._tokenizer, '_tokenizer') and hasattr(self._tokenizer._tokenizer, 'mask_token_id'):
            self.token_type = "huggingface"
        else:
            self.token_type = "emb"
            self.padding_value = Token(text="@@PADDING@@", text_id=0)
            self.cls_value = Token(text="@@UNKNOWN@@", text_id=1)
        
        # Stopwords (keeping from original)
        self.lucene_stopwords = set(["a", "an", "and", "are", "as", "at", "be", "but", "by",
                                   "for", "if", "in", "into", "is", "it", "no", "not", "of", 
                                   "on", "or", "such", "that", "the", "their", "then", "there", 
                                   "these", "they", "this", "to", "was", "will", "with"])
        
        self.use_stopwords = False
        self.query_augment_mask_number = query_augment_mask_number
        
        self.max_title_length = 30
        self.min_title_length = -1
        self.train_qa_spans = train_qa_spans
        
        # Load data
        self.data = self._read_file()
        
    def _read_file(self) -> List[Dict]:
        """Read and parse the TSV file"""
        data = []
        
        with open(cached_path(self.file_path), "r", encoding="utf8") as data_file:
            for line_num, line in enumerate(data_file, 1):
                line = line.strip("\n")
                
                if not line:
                    continue
                
                line_parts = line.split('\t')
                if len(line_parts) == 4:
                    query_id, doc_id, query_sequence, doc_sequence = line_parts
                    doc_title = None
                elif len(line_parts) == 5:
                    query_id, doc_id, query_sequence, doc_title, doc_sequence = line_parts
                else:
                    raise ConfigurationError(f"Invalid line format at line {line_num}: {line}")
                
                data.append({
                    'query_id': query_id,
                    'doc_id': doc_id,
                    'query_sequence': query_sequence,
                    'doc_sequence': doc_sequence,
                    'doc_title': doc_title
                })
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single item from the dataset"""
        item = self.data[idx]
        return self.text_to_instance(
            item['query_id'],
            item['doc_id'],
            item['query_sequence'],
            item['doc_sequence'],
            item['doc_title']
        )
    
    def text_to_instance(self, query_id: str, doc_id: str, query_sequence: str, 
                        doc_sequence: str, doc_title: Optional[str]) -> Dict[str, Any]:
        """Convert text to instance format"""
        
        result = {
            "query_id": query_id,
            "doc_id": doc_id
        }
        
        # Process query
        if self.token_type == "huggingface":
            query_tokenized = self._tokenizer.tokenize(query_sequence, max_length=self.max_query_length)
            
            if self.query_augment_mask_number > -1:
                # Add mask tokens for query augmentation
                mask_token_id = self._tokenizer._tokenizer.mask_token_id
                input_ids = query_tokenized["input_ids"]
                attention_mask = query_tokenized["attention_mask"]
                
                # Pad with mask tokens before the last token (usually [SEP])
                padded_input = torch.cat([
                    torch.nn.functional.pad(input_ids[:-1], (0, self.query_augment_mask_number), value=mask_token_id),
                    input_ids[-1].unsqueeze(0)
                ])
                
                padded_attention = torch.nn.functional.pad(attention_mask, (0, self.query_augment_mask_number), value=1)
                
                query_tokenized["input_ids"] = padded_input
                query_tokenized["attention_mask"] = padded_attention
            
            result["query_tokens"] = query_tokenized
        else:
            query_tokenized = self._tokenizer.tokenize(query_sequence)
            
            if self.max_query_length > -1:
                query_tokenized = query_tokenized[:self.max_query_length]
            if self.min_query_length > -1 and len(query_tokenized) < self.min_query_length:
                query_tokenized = query_tokenized + [self.padding_value] * (self.min_query_length - len(query_tokenized))
            
            result["query_tokens"] = query_tokenized
        
        # Process document
        if self.token_type == "huggingface":
            doc_tokenized = self._tokenizer.tokenize(doc_sequence, max_length=self.max_doc_length)
            result["doc_tokens"] = doc_tokenized
        else:
            doc_tokenized = self._tokenizer.tokenize(doc_sequence)
            if self.max_doc_length > -1:
                doc_tokenized = doc_tokenized[:self.max_doc_length]
            if self.min_doc_length > -1 and len(doc_tokenized) < self.min_doc_length:
                doc_tokenized = doc_tokenized + [self.padding_value] * (self.min_doc_length - len(doc_tokenized))
            
            result["doc_tokens"] = doc_tokenized
        
        # Process title if present
        if doc_title is not None:
            if self.token_type == "huggingface":
                # Temporarily set max length for title
                original_max_length = getattr(self._tokenizer, '_max_length', 512)
                self._tokenizer._max_length = self.max_title_length - 2
                
                title_tokenized = self._tokenizer.tokenize(doc_title)
                result["title_tokens"] = title_tokenized
                
                # Restore original max length
                self._tokenizer._max_length = original_max_length
            else:
                title_tokenized = self._tokenizer.tokenize(doc_title)
                
                if self.max_title_length > -1:
                    title_tokenized = title_tokenized[:self.max_title_length]
                if self.min_title_length > -1 and len(title_tokenized) < self.min_title_length:
                    title_tokenized = title_tokenized + [self.padding_value] * (self.min_title_length - len(title_tokenized))
                
                title_tokenized.insert(0, self.cls_value)
                result["title_tokens"] = title_tokenized
        
        return result

# Replacement for the inference loader function
def pytorch_reranking_inference_loader(model_config, run_config, input_file):
    """
    Load examples from a .tsv file in the reranking candidate file format: q_id<tab>d_id<tab>q_text<tab>d_text
    
    Using PyTorch's DataLoader instead of AllenNLP's multiprocess loader
    """
    from torch.utils.data import DataLoader
    
    # Get tokenizer and indexers
    _tokenizer, _token_indexers = _get_indexer_pytorch(
        model_config, 
        max(run_config["max_doc_length"], run_config["max_query_length"])
    )
    
    # Create dataset
    dataset = IndependentReRankingDataset(
        file_path=input_file,
        tokenizer=_tokenizer,
        token_indexers=_token_indexers,
        max_doc_length=run_config["max_doc_length"],
        max_query_length=run_config["max_query_length"],
        min_doc_length=run_config["min_doc_length"],
        min_query_length=run_config["min_query_length"],
        query_augment_mask_number=run_config["query_augment_mask_number"],
        train_qa_spans=run_config["train_qa_spans"]
    )
    
    # Create DataLoader
    def collate_fn(batch):
        """Custom collate function to handle batching"""
        if not batch:
            return {}
        
        # Group by keys
        result = {}
        for key in batch[0].keys():
            if key in ["query_id", "doc_id"]:
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
    
    loader = DataLoader(
        dataset,
        batch_size=run_config["batch_size_eval"],
        num_workers=run_config.get("dataloader_num_workers", 0),
        collate_fn=collate_fn,
        shuffle=False
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

# Example usage:
if __name__ == "__main__":
    # Example configuration
    model_config = {
        "bert_pretrained_model": "bert-base-uncased"
    }
    
    run_config = {
        "max_doc_length": 512,
        "max_query_length": 64,
        "min_doc_length": -1,
        "min_query_length": -1,
        "query_augment_mask_number": -1,
        "train_qa_spans": False,
        "batch_size_eval": 32,
        "dataloader_num_workers": 4
    }
    
    # Create dataset
    dataset = IndependentReRankingDataset(
        file_path="your_data.tsv",  # Replace with your actual file
        tokenizer=FastTransformerTokenizer("bert-base-uncased"),
        max_doc_length=run_config["max_doc_length"],
        max_query_length=run_config["max_query_length"]
    )
    
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Create data loader
    loader = pytorch_reranking_inference_loader(model_config, run_config, "your_data.tsv")
    print("DataLoader created successfully!")