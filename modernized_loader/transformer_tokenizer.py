"""
Modern replacement for the original transformer_tokenizer.py
This module provides updated tokenizer classes that work with current
Hugging Face Transformers and PyTorch, compatible with Python 3.12
"""

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from typing import Dict, Optional, Union, Any


class ModernTransformerTextField:
    """
    Modern replacement for AllenNLP's TransformerTextField
    Stores tokenized text with input_ids and attention_mask
    """
    
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        
        # Store any additional tokenizer outputs (token_type_ids, etc.)
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __len__(self) -> int:
        """Return the sequence length"""
        return self.input_ids.shape[-1]
    
    def to(self, device: Union[str, torch.device]) -> 'ModernTransformerTextField':
        """Move tensors to specified device"""
        new_kwargs = {}
        for attr_name in dir(self):
            if not attr_name.startswith('_') and attr_name not in ['input_ids', 'attention_mask']:
                attr_value = getattr(self, attr_name)
                if isinstance(attr_value, torch.Tensor):
                    new_kwargs[attr_name] = attr_value.to(device)
                else:
                    new_kwargs[attr_name] = attr_value
        
        return ModernTransformerTextField(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            **new_kwargs
        )


class FastTransformerTokenizer:
    """
    Modern replacement for the original FastTransformerTokenizer
    Provides a clean interface to Hugging Face tokenizers with the same API
    """
    
    def __init__(self, model_name_or_path: str, **tokenizer_kwargs):
        """
        Initialize the tokenizer
        
        Args:
            model_name_or_path: Path to model or model name from Hugging Face Hub
            **tokenizer_kwargs: Additional arguments passed to AutoTokenizer.from_pretrained
        """
        self.model_name_or_path = model_name_or_path
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
        
        # Ensure we have a pad token
        if self._tokenizer.pad_token is None:
            if self._tokenizer.eos_token:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            elif self._tokenizer.unk_token:
                self._tokenizer.pad_token = self._tokenizer.unk_token
            else:
                # Add a pad token if none exists
                self._tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    def tokenize(
        self, 
        text1: str, 
        text2: Optional[str] = None, 
        max_length: int = 512,
        padding: Union[bool, str] = False,
        truncation: bool = True,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize text(s) and return tensors
        
        Args:
            text1: First text to tokenize (query, sentence A, etc.)
            text2: Optional second text (document, sentence B, etc.)
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate sequences
            return_tensors: Type of tensors to return ("pt" for PyTorch)
            
        Returns:
            Dictionary containing input_ids, attention_mask, and potentially other keys
        """
        if text2 is not None:
            # Pair of sequences (e.g., query-document pair)
            encoded = self._tokenizer(
                text1, 
                text2,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_tensors=return_tensors,
                return_attention_mask=True
            )
        else:
            # Single sequence
            encoded = self._tokenizer(
                text1,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_tensors=return_tensors,
                return_attention_mask=True
            )
        
        # Remove batch dimension if present (for single examples)
        if return_tensors == "pt":
            for key in encoded:
                if encoded[key].dim() > 1 and encoded[key].shape[0] == 1:
                    encoded[key] = encoded[key].squeeze(0)
        
        return encoded
    
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text"""
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def batch_decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> list:
        """Decode a batch of token ID sequences"""
        return self._tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self._tokenizer)
    
    @property
    def pad_token_id(self) -> int:
        """Get pad token ID"""
        return self._tokenizer.pad_token_id
    
    @property
    def cls_token_id(self) -> int:
        """Get CLS token ID"""
        return self._tokenizer.cls_token_id
    
    @property
    def sep_token_id(self) -> int:
        """Get SEP token ID"""
        return self._tokenizer.sep_token_id
    
    def __call__(self, *args, **kwargs):
        """Make the tokenizer callable, delegating to the underlying tokenizer"""
        return self._tokenizer(*args, **kwargs)


class PatchedTransformerTextField(ModernTransformerTextField):
    """
    Backward compatibility class that mimics the original PatchedTransformerTextField
    but uses modern PyTorch tensors instead of AllenNLP fields
    """
    
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        super().__init__(input_ids, attention_mask, **kwargs)
    
    @classmethod
    def from_tokenizer_output(cls, tokenizer_output: Dict[str, torch.Tensor]) -> 'PatchedTransformerTextField':
        """
        Create a PatchedTransformerTextField from tokenizer output
        """
        input_ids = tokenizer_output.pop('input_ids')
        attention_mask = tokenizer_output.pop('attention_mask')
        return cls(input_ids=input_ids, attention_mask=attention_mask, **tokenizer_output)


# Utility functions for backward compatibility
def create_tokenized_field(tokenizer_output: Dict[str, torch.Tensor]) -> PatchedTransformerTextField:
    """
    Helper function to create a tokenized field from tokenizer output
    """
    return PatchedTransformerTextField.from_tokenizer_output(tokenizer_output)


# Example usage
if __name__ == "__main__":
    # Example of how to use the modernized tokenizer
    tokenizer = FastTransformerTokenizer("bert-base-uncased")
    
    # Single text tokenization
    query = "What is machine learning?"
    query_tokens = tokenizer.tokenize(query, max_length=64)
    print(f"Query tokens shape: {query_tokens['input_ids'].shape}")
    
    # Text pair tokenization  
    document = "Machine learning is a subset of artificial intelligence."
    pair_tokens = tokenizer.tokenize(query, document, max_length=128)
    print(f"Pair tokens shape: {pair_tokens['input_ids'].shape}")
    
    # Create field objects (for compatibility)
    query_field = PatchedTransformerTextField.from_tokenizer_output(query_tokens.copy())
    print(f"Query field length: {len(query_field)}")
    
    # Decode back to text
    decoded = tokenizer.decode(query_tokens['input_ids'])
    print(f"Decoded text: {decoded}")
    
    print("\nModern tokenizer features:")
    print(f"- Vocab size: {tokenizer.vocab_size}")
    print(f"- Pad token ID: {tokenizer.pad_token_id}")
    print(f"- CLS token ID: {tokenizer.cls_token_id}")
    print(f"- SEP token ID: {tokenizer.sep_token_id}")