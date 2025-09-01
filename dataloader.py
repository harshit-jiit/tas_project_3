import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import random
import json
from collections import defaultdict



class BaseTextDataset(Dataset):
    """Base class for text datasets with common functionality"""
    
    def __init__(self, tokenizer, max_length: int = 512, min_length: int = 1):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        
    def tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text and return tensor dict"""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}
    
    def _load_data(self, file_path: str) -> List:
        """Override this method in subclasses to load specific data formats"""
        raise NotImplementedError("Subclasses must implement _load_data method")
    
    def __len__(self):
        return len(self.data) if hasattr(self, 'data') else 0

class MLMDataset(BaseTextDataset):
    """Dataset for Masked Language Model training"""
    
    def __init__(self, file_path: str, tokenizer, max_length: int = 512, 
                 mask_prob: float = 0.15):
        super().__init__(tokenizer, max_length)
        self.mask_prob = mask_prob
        self.data = self._load_data(file_path)
    
    def _load_data(self, file_path: str) -> List[str]:
        """Load text data for MLM training"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip()
                if self.min_length <= len(text.split()) <= self.max_length:
                    data.append(text)
        return data
    
    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = self.tokenize_text(text)
        
        # Apply MLM masking
        masked_input_ids, labels = self._mask_tokens(tokens['input_ids'])
        
        return {
            'input_ids': masked_input_ids,
            'attention_mask': tokens['attention_mask'],
            'labels': labels
        }
        
    def _mask_tokens(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply MLM masking to tokens"""
        labels = tokens.clone()
        probability_matrix = torch.full(labels.shape, self.mask_prob)
        
        # Don't mask special tokens
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) 
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens
        
        # 80% mask, 10% random, 10% original
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        tokens[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        tokens[indices_random] = random_words[indices_random]
        
        return tokens, labels

class TASBalancedDataset(IterableDataset):
    """
    Dataset for TAS balanced training with teacher scores and query clustering.
    Replicates the functionality of TASBalancedDatasetLoader from AllenNLP.
    """
    
    def __init__(self, query_file: str, collection_file: str, 
                 pairs_with_teacher_scores: str, query_cluster_file: str,
                 tokenizer, batch_size: int, clusters_per_batch: int,
                 max_query_length: int = 64, max_doc_length: int = 512,
                 pair_balancing_strategy: str = "random", random_seed: int = 42):
        
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.clusters_per_batch = clusters_per_batch
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        self.pair_balancing_strategy = pair_balancing_strategy
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Load all required data
        self.queries = self._load_queries(query_file)
        self.documents = self._load_documents(collection_file)
        self.pairs_with_scores = self._load_pairs_with_scores(pairs_with_teacher_scores)
        self.query_clusters = self._load_query_clusters(query_cluster_file)
        
        print(f"Loaded {len(self.queries)} queries, {len(self.documents)} documents")
        print(f"Loaded {len(self.pairs_with_scores)} queries with teacher scores")
        print(f"Loaded {len(self.query_clusters)} query clusters")
        
    def _load_queries(self, file_path: str) -> Dict[str, str]:
        """Load queries from file: query_id<tab>query_text"""
        queries = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    queries[parts[0]] = parts[1]
        return queries
    
    def _load_documents(self, file_path: str) -> Dict[str, str]:
        """Load documents from file: doc_id<tab>doc_text"""
        documents = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    documents[parts[0]] = parts[1]
        return documents
    
    def _load_pairs_with_scores(self, file_path: str) -> Dict[str, List[Tuple[str, float]]]:
        """Load query-document pairs with teacher scores: query_id<tab>doc_id<tab>score"""
        pairs = defaultdict(list)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    q_id, d_id, score = parts[0], parts[1], float(parts[2])
                    pairs[q_id].append((d_id, score))
        
        # Sort pairs by score for each query (descending order)
        for q_id in pairs:
            pairs[q_id].sort(key=lambda x: x[1], reverse=True)
        
        return dict(pairs)
    
    def _load_query_clusters(self, file_path: str) -> Dict[str, List[str]]:
        """Load query clusters: cluster_id<tab>query_id"""
        clusters = defaultdict(list)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    cluster_id, q_id = parts[0], parts[1]
                    clusters[cluster_id].append(q_id)
        return dict(clusters)
    
    def __iter__(self):
        """Iterate over batches of training examples"""
        cluster_ids = list(self.query_clusters.keys())
        
        # Shuffle clusters for each epoch
        random.shuffle(cluster_ids)
        
        for i in range(0, len(cluster_ids), self.clusters_per_batch):
            batch_clusters = cluster_ids[i:i + self.clusters_per_batch]
            
            batch_examples = []
            
            # Process each cluster in the current batch
            for cluster_id in batch_clusters:
                queries_in_cluster = self.query_clusters[cluster_id]
                
                # Sample queries from this cluster
                sampled_queries = random.sample(
                    queries_in_cluster, 
                    min(len(queries_in_cluster), max(1, self.batch_size // len(batch_clusters)))
                )
                
                for q_id in sampled_queries:
                    if q_id in self.pairs_with_scores and q_id in self.queries:
                        # Generate training pairs for this query
                        pairs = self._generate_training_pairs(q_id)
                        batch_examples.extend(pairs)
            
            # Limit batch size and yield
            if batch_examples:
                # Shuffle examples within batch
                random.shuffle(batch_examples)
                
                # Yield batches of the specified size
                for j in range(0, len(batch_examples), self.batch_size):
                    batch = batch_examples[j:j + self.batch_size]
                    if len(batch) > 0:
                        yield self._collate_batch(batch)
    
    def _generate_training_pairs(self, q_id: str) -> List[Dict]:
        """Generate positive and negative training pairs for a query"""
        pairs_with_scores = self.pairs_with_scores[q_id]
        query_text = self.queries[q_id]
        
        if len(pairs_with_scores) < 2:
            return []
        
        training_pairs = []
        
        if self.pair_balancing_strategy == "score_based":
            # Use teacher scores to balance positive and negative pairs
            mid_point = len(pairs_with_scores) // 2
            positive_pairs = pairs_with_scores[:mid_point]
            negative_pairs = pairs_with_scores[mid_point:]
            
            # Create pairs: each positive with each negative (limited)
            for pos_doc_id, pos_score in positive_pairs[:3]:  # Limit to top 3 positives
                for neg_doc_id, neg_score in negative_pairs[-3:]:  # Limit to bottom 3 negatives
                    if (pos_doc_id in self.documents and 
                        neg_doc_id in self.documents and 
                        pos_score > neg_score):
                        
                        training_pairs.append({
                            'query_text': query_text,
                            'pos_doc_text': self.documents[pos_doc_id],
                            'neg_doc_text': self.documents[neg_doc_id],
                            'pos_score': pos_score,
                            'neg_score': neg_score,
                            'query_id': q_id,
                            'pos_doc_id': pos_doc_id,
                            'neg_doc_id': neg_doc_id
                        })
        
        elif self.pair_balancing_strategy == "random":
            # Random sampling of positive and negative pairs
            available_pairs = [(doc_id, score) for doc_id, score in pairs_with_scores 
                             if doc_id in self.documents]
            
            if len(available_pairs) >= 2:
                # Sample pairs randomly
                sampled_pairs = random.sample(available_pairs, min(6, len(available_pairs)))
                sampled_pairs.sort(key=lambda x: x[1], reverse=True)
                
                mid = len(sampled_pairs) // 2
                positives = sampled_pairs[:mid]
                negatives = sampled_pairs[mid:]
                
                for pos_doc_id, pos_score in positives:
                    for neg_doc_id, neg_score in negatives:
                        if pos_score > neg_score:
                            training_pairs.append({
                                'query_text': query_text,
                                'pos_doc_text': self.documents[pos_doc_id],
                                'neg_doc_text': self.documents[neg_doc_id],
                                'pos_score': pos_score,
                                'neg_score': neg_score,
                                'query_id': q_id,
                                'pos_doc_id': pos_doc_id,
                                'neg_doc_id': neg_doc_id
                            })
        
        elif self.pair_balancing_strategy == "top_bottom":
            # Take top scoring as positive, bottom scoring as negative
            if len(pairs_with_scores) >= 2:
                top_pairs = pairs_with_scores[:max(1, len(pairs_with_scores) // 4)]
                bottom_pairs = pairs_with_scores[-max(1, len(pairs_with_scores) // 4):]
                
                for pos_doc_id, pos_score in top_pairs:
                    for neg_doc_id, neg_score in bottom_pairs:
                        if (pos_doc_id in self.documents and 
                            neg_doc_id in self.documents and 
                            pos_score > neg_score):
                            
                            training_pairs.append({
                                'query_text': query_text,
                                'pos_doc_text': self.documents[pos_doc_id],
                                'neg_doc_text': self.documents[neg_doc_id],
                                'pos_score': pos_score,
                                'neg_score': neg_score,
                                'query_id': q_id,
                                'pos_doc_id': pos_doc_id,
                                'neg_doc_id': neg_doc_id
                            })
        
        return training_pairs
    
    def _collate_batch(self, batch_examples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate a batch of examples into tensors"""
        if not batch_examples:
            return {}
        
        # Tokenize all texts in the batch
        query_texts = [ex['query_text'] for ex in batch_examples]
        pos_doc_texts = [ex['pos_doc_text'] for ex in batch_examples]
        neg_doc_texts = [ex['neg_doc_text'] for ex in batch_examples]
        
        # Tokenize queries
        query_tokens = self.tokenizer(
            query_texts,
            truncation=True,
            padding=True,
            max_length=self.max_query_length,
            return_tensors='pt'
        )
        
        # Tokenize positive documents
        pos_doc_tokens = self.tokenizer(
            pos_doc_texts,
            truncation=True,
            padding=True,
            max_length=self.max_doc_length,
            return_tensors='pt'
        )
        
        # Tokenize negative documents
        neg_doc_tokens = self.tokenizer(
            neg_doc_texts,
            truncation=True,
            padding=True,
            max_length=self.max_doc_length,
            return_tensors='pt'
        )
        
        return {
            'query_input_ids': query_tokens['input_ids'],
            'query_attention_mask': query_tokens['attention_mask'],
            'pos_doc_input_ids': pos_doc_tokens['input_ids'],
            'pos_doc_attention_mask': pos_doc_tokens['attention_mask'],
            'neg_doc_input_ids': neg_doc_tokens['input_ids'],
            'neg_doc_attention_mask': neg_doc_tokens['attention_mask'],
            'pos_scores': torch.tensor([ex['pos_score'] for ex in batch_examples], dtype=torch.float),
            'neg_scores': torch.tensor([ex['neg_score'] for ex in batch_examples], dtype=torch.float),
            'query_ids': [ex['query_id'] for ex in batch_examples],
            'pos_doc_ids': [ex['pos_doc_id'] for ex in batch_examples],
            'neg_doc_ids': [ex['neg_doc_id'] for ex in batch_examples],
        }


class ListTrainingDataset(BaseTextDataset):
    """Dataset for list-wise training with multiple candidates"""
    
    def __init__(self, file_path: str, tokenizer, max_query_length: int = 64,
                 max_doc_length: int = 512, max_candidates: int = 10):
        super().__init__(tokenizer)
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        self.max_candidates = max_candidates
        self.data = self._load_data(file_path)
    
    def _load_data(self, file_path: str) -> List[Dict]:
        # Load data in format: query_id, [list of candidate docs with scores]
        data = []
        current_query = None
        current_docs = []
        
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:  # q_id, d_id, query_text, doc_text, [score]
                    q_id, d_id, q_text, d_text = parts[:4]
                    score = float(parts[4]) if len(parts) > 4 else 0.0
                    
                    if current_query != q_id:
                        if current_query is not None and current_docs:
                            data.append({
                                'query_id': current_query,
                                'candidates': current_docs[:self.max_candidates]
                            })
                        current_query = q_id
                        current_docs = []
                    
                    current_docs.append({
                        'doc_id': d_id,
                        'query_text': q_text,
                        'doc_text': d_text,
                        'score': score
                    })
        
        # Add the last query
        if current_query is not None and current_docs:
            data.append({
                'query_id': current_query,
                'candidates': current_docs[:self.max_candidates]
            })
        
        return data
    
    def __getitem__(self, idx):
        item = self.data[idx]
        candidates = item['candidates']
        
        if not candidates:
            return None
        
        query_text = candidates[0]['query_text']
        query_tokens = self.tokenizer(
            query_text, truncation=True, padding='max_length',
            max_length=self.max_query_length, return_tensors='pt'
        )
        
        doc_tokens_list = []
        scores = []
        
        for candidate in candidates:
            doc_tokens = self.tokenizer(
                candidate['doc_text'], truncation=True, padding='max_length',
                max_length=self.max_doc_length, return_tensors='pt'
            )
            doc_tokens_list.append({
                'input_ids': doc_tokens['input_ids'].squeeze(0),
                'attention_mask': doc_tokens['attention_mask'].squeeze(0)
            })
            scores.append(candidate['score'])
        
        # Pad to max_candidates
        while len(doc_tokens_list) < self.max_candidates:
            doc_tokens_list.append({
                'input_ids': torch.zeros(self.max_doc_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_doc_length, dtype=torch.long)
            })
            scores.append(0.0)
        
        return {
            'query_id': item['query_id'],
            'query_input_ids': query_tokens['input_ids'].squeeze(0),
            'query_attention_mask': query_tokens['attention_mask'].squeeze(0),
            'doc_input_ids': torch.stack([doc['input_ids'] for doc in doc_tokens_list]),
            'doc_attention_mask': torch.stack([doc['attention_mask'] for doc in doc_tokens_list]),
            'scores': torch.tensor(scores, dtype=torch.float),
            'num_candidates': len(candidates)
        }

def modern_tas_balanced_loader(model_config: Dict, run_config: Dict) -> DataLoader:
    """
    Modern replacement for TAS balanced training loader.
    
    Expected run_config keys:
    - dynamic_query_file: Path to query file
    - dynamic_collection_file: Path to document collection file
    - dynamic_pairs_with_teacher_scores: Path to query-doc pairs with teacher scores
    - dynamic_query_cluster_file: Path to query cluster assignments
    - batch_size_train: Training batch size
    - dynamic_clusters_per_batch: Number of clusters to sample per batch
    - max_query_length: Maximum query length
    - max_doc_length: Maximum document length
    - tas_balanced_pair_strategy: Strategy for pair sampling ("random", "score_based", "top_bottom")
    - random_seed: Random seed for reproducibility
    """
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.get("bert_pretrained_model", "bert-base-uncased")
    )
    
    dataset = TASBalancedDataset(
        query_file=run_config["dynamic_query_file"],
        collection_file=run_config["dynamic_collection_file"],
        pairs_with_teacher_scores=run_config["dynamic_pairs_with_teacher_scores"],
        query_cluster_file=run_config["dynamic_query_cluster_file"],
        tokenizer=tokenizer,
        batch_size=run_config["batch_size_train"],
        clusters_per_batch=run_config["dynamic_clusters_per_batch"],
        max_query_length=run_config["max_query_length"],
        max_doc_length=run_config["max_doc_length"],
        pair_balancing_strategy=run_config.get("tas_balanced_pair_strategy", "random"),
        random_seed=run_config.get("random_seed", 42)
    )
    
    # Use DataLoader with batch_size=None since TASBalancedDataset yields pre-batched data
    return DataLoader(
        dataset, 
        batch_size=None,  # Dataset handles batching internally
        num_workers=0,    # IterableDataset doesn't work well with multiprocessing
        pin_memory=True
    )

def modern_mlm_loader(model_config: Dict, run_config: Dict, input_file: str) -> DataLoader:
    """Modern replacement for MLM training loader"""
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.get("bert_pretrained_model", "bert-base-uncased")
    )
    
    dataset = MLMDataset(
        input_file, tokenizer,
        max_length=run_config.get("max_sequence_length", 512),
        mask_prob=run_config.get("mlm_probability", 0.15)
    )
    
    return DataLoader(
        dataset,
        batch_size=run_config["batch_size_train"],
        shuffle=True,
        num_workers=run_config.get("dataloader_num_workers", 4),
        pin_memory=True
    )

def modern_list_training_loader(model_config: Dict, run_config: Dict, input_file: str) -> DataLoader:
    """Modern replacement for list training loader"""
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.get("bert_pretrained_model", "bert-base-uncased")
    )
    
    dataset = ListTrainingDataset(
        input_file, tokenizer,
        max_query_length=run_config["max_query_length"],
        max_doc_length=run_config["max_doc_length"],
        max_candidates=run_config.get("max_candidates_per_query", 10)
    )
    
    return DataLoader(
        dataset,
        batch_size=run_config["batch_size_train"],
        shuffle=True,
        num_workers=run_config.get("dataloader_num_workers", 4),
        pin_memory=True,
        collate_fn=lambda x: x  # Custom collation for variable list lengths
    )