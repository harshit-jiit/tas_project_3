# TAS-Balanced Dense Retrieval Training Notebook
# Based on "Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling"

import os
import torch
import torch.nn as nn
import numpy as np
import random
from typing import Dict, List, Tuple
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
import json
from tqdm import tqdm
import math

# Set seeds for reproducibility
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class BERTDot(nn.Module):
    """BERT_DOT model for dense retrieval - student model"""
    
    def __init__(self, bert_model_name: str = "distilbert-base-uncased", 
                 compress_dim: int = -1, return_vecs: bool = False):
        super().__init__()
        self.bert_model = AutoModel.from_pretrained(bert_model_name)
        self.return_vecs = return_vecs
        self.use_compressor = compress_dim > -1
        
        if self.use_compressor:
            self.compressor = nn.Linear(self.bert_model.config.hidden_size, compress_dim)
    
    def forward_representation(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for getting representations"""
        vectors = self.bert_model(**tokens)[0][:, 0, :]  # CLS token
        if self.use_compressor:
            vectors = self.compressor(vectors)
        return vectors
    
    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor], 
                use_fp16: bool = True) -> torch.Tensor:
        """Forward pass for scoring"""
        with torch.cuda.amp.autocast(enabled=use_fp16):
            query_vecs = self.forward_representation(query)
            document_vecs = self.forward_representation(document)
            
            # Dot product scoring
            score = torch.bmm(query_vecs.unsqueeze(1), document_vecs.unsqueeze(2)).squeeze(-1).squeeze(-1)
            
            # Return vectors for in-batch negatives during training
            if self.training and self.return_vecs:
    def forward_inbatch_aggregation(self, query_vecs: torch.Tensor, query_mask: torch.Tensor,
                                   document_vecs: torch.Tensor, document_mask: torch.Tensor) -> torch.Tensor:
        """ColBERT in-batch aggregation for computing all query-document pairs efficiently"""
        
        # Reshape for matrix multiplication: [batch*seq_len, dim]
        query_flat = query_vecs.view(-1, query_vecs.shape[-1])  # [batch*q_len, dim]  
        doc_flat = document_vecs.view(-1, document_vecs.shape[-1])  # [batch*d_len, dim]
        
        # Compute all interactions: [batch*q_len, batch*d_len]
        score = torch.mm(query_flat, doc_flat.transpose(-2, -1))
        
        # Reshape back: [batch, q_len, batch, d_len]
        score = score.view(query_vecs.shape[0], query_vecs.shape[1], 
                          document_vecs.shape[0], document_vecs.shape[1])
        
        # Transpose to get [batch, batch, q_len, d_len] for easier masking
        score = score.transpose(1, 2)
        
        # Apply document mask: mask out padding tokens
        doc_mask_expanded = document_mask.bool().unsqueeze(1).unsqueeze(1).expand(-1, score.shape[1], score.shape[2], -1)
        score[~doc_mask_expanded] = -1000
        
        # Max pooling over document terms
        score = score.max(-1).values  # [batch, batch, q_len]
        
        # Apply query mask: mask out padding tokens  
        query_mask_expanded = query_mask.bool().unsqueeze(1).expand(-1, score.shape[1], -1)
        score[~query_mask_expanded] = 0
        
        # Sum over query terms to get final scores [batch, batch]
        score = score.sum(-1)
        
        return score
            
            return score

class ColBERT(nn.Module):
    """ColBERT model - teacher model for in-batch negatives"""
    
    def __init__(self, bert_model_name: str = "sebastian-hofstaetter/colbert-distilbert-margin_mse-T2-msmarco", 
                 compression_dim: int = 128):
        super().__init__()
        self.bert_model = AutoModel.from_pretrained(bert_model_name)
        self.compressor = nn.Linear(self.bert_model.config.hidden_size, compression_dim)
        self.is_teacher_model = True
        
        # Freeze parameters for teacher
        for param in self.parameters():
            param.requires_grad = False
    
    def forward_representation(self, tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for getting term representations"""
        vecs = self.bert_model(**tokens)[0]
        vecs = self.compressor(vecs)
        return vecs
    
    def forward(self, query: Dict[str, torch.Tensor], document: Dict[str, torch.Tensor], 
                use_fp16: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning score and vectors"""
        with torch.cuda.amp.autocast(enabled=use_fp16):
            query_vecs = self.forward_representation(query)
            document_vecs = self.forward_representation(document)
            
            # ColBERT scoring: max pooling over document terms, sum over query terms
            score_per_term = torch.bmm(query_vecs, document_vecs.transpose(2, 1))
            
            # Mask document padding
            doc_mask = document["attention_mask"].bool().unsqueeze(1).expand(-1, score_per_term.shape[1], -1)
            score_per_term[~doc_mask] = -1000
            
            # Max pool over document terms
            score = score_per_term.max(-1).values
            
            # Mask query padding and sum
            query_mask = query["attention_mask"].bool()
            score[~query_mask] = 0
            score = score.sum(-1)
            
            return score, query_vecs, document_vecs

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class MarginMSELoss(nn.Module):
    """Margin MSE loss for knowledge distillation"""
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, student_pos: torch.Tensor, student_neg: torch.Tensor,
                teacher_pos: torch.Tensor, teacher_neg: torch.Tensor) -> torch.Tensor:
        student_margin = student_pos - student_neg
        teacher_margin = teacher_pos - teacher_neg
        return self.mse(student_margin, teacher_margin)

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_queries(query_file: str) -> Dict[str, str]:
    """Load queries from TSV file"""
    queries = {}
    with open(query_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                queries[parts[0]] = parts[1]
    return queries

def load_collection(collection_file: str) -> Dict[str, str]:
    """Load document collection from TSV file"""
    collection = {}
    with open(collection_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                collection[parts[0]] = parts[1]
    return collection

def load_clusters(cluster_file: str) -> List[List[str]]:
    """Load query clusters from file"""
    clusters = []
    with open(cluster_file, 'r', encoding='utf-8') as f:
        for line in f:
            cluster_ids = line.strip().split()
            if cluster_ids:
                clusters.append(cluster_ids)
    return clusters

def load_teacher_scores(scores_file: str) -> Dict[str, List[Tuple[str, str, float, float]]]:
    """Load teacher scores for pairwise supervision"""
    scores_by_qid = defaultdict(list)
    with open(scores_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                pos_score = float(parts[0])
                neg_score = float(parts[1])
                query_id = parts[2]
                pos_doc_id = parts[3]
                neg_doc_id = parts[4]
                scores_by_qid[query_id].append((pos_doc_id, neg_doc_id, pos_score, neg_score))
    return scores_by_qid

def create_balanced_bins(scores_by_qid: Dict[str, List[Tuple]], num_bins: int = 10):
    """Create balanced margin bins for TAS-Balanced sampling"""
    binned_scores = defaultdict(lambda: defaultdict(list))
    
    for query_id, pairs in scores_by_qid.items():
        if not pairs:
            continue
            
        # Calculate margins
        margins = [pos_score - neg_score for _, _, pos_score, neg_score in pairs]
        
        if not margins:
            continue
            
        min_margin = min(margins)
        max_margin = max(margins)
        
        if min_margin == max_margin:
            # All margins are the same, put everything in bin 0
            binned_scores[query_id][0] = pairs
            continue
            
        bin_size = (max_margin - min_margin) / num_bins
        
        for pair in pairs:
            pos_doc_id, neg_doc_id, pos_score, neg_score = pair
            margin = pos_score - neg_score
            bin_idx = min(int((margin - min_margin) / bin_size), num_bins - 1)
            binned_scores[query_id][bin_idx].append(pair)
    
    return binned_scores

def tokenize_batch(texts: List[str], tokenizer, max_length: int) -> Dict[str, torch.Tensor]:
    """Tokenize a batch of texts"""
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

# ============================================================================
# TAS-BALANCED BATCH SAMPLING
# ============================================================================

class TASBalancedSampler:
    """TAS-Balanced batch sampler"""
    
    def __init__(self, queries: Dict[str, str], collection: Dict[str, str],
                 clusters: List[List[str]], teacher_scores: Dict[str, List[Tuple]],
                 tokenizer, batch_size: int = 32, clusters_per_batch: int = 1,
                 max_query_length: int = 30, max_doc_length: int = 200,
                 num_bins: int = 10, seed: int = 42):
        
        self.queries = queries
        self.collection = collection
        self.clusters = clusters
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.clusters_per_batch = clusters_per_batch
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        self.seed = seed
        
        # Create balanced bins
        self.binned_scores = create_balanced_bins(teacher_scores, num_bins)
        
        # Filter valid queries (intersection of queries, clusters, and teacher scores)
        cluster_qids = set()
        for cluster in clusters:
            cluster_qids.update(cluster)
        
        self.valid_qids = set(queries.keys()) & cluster_qids & set(teacher_scores.keys())
        
        # Clean clusters
        self.clean_clusters = []
        for cluster in clusters:
            clean_cluster = [qid for qid in cluster if qid in self.valid_qids]
            if clean_cluster:
                self.clean_clusters.append(clean_cluster)
        
        random.seed(seed)
    
    def sample_batch(self) -> Dict[str, torch.Tensor]:
        """Sample a TAS-Balanced batch"""
        batch_samples = []
        
        while len(batch_samples) < self.batch_size:
            # Randomly select cluster(s)
            cluster_idx = random.randint(0, len(self.clean_clusters) - 1)
            cluster = self.clean_clusters[cluster_idx]
            
            # Sample queries from cluster
            query_target_count = max(1, self.batch_size // self.clusters_per_batch)
            if query_target_count < len(cluster):
                selected_qids = random.sample(cluster, min(query_target_count, len(cluster)))
            else:
                selected_qids = cluster
            
            for qid in selected_qids:
                if len(batch_samples) >= self.batch_size:
                    break
                
                if qid not in self.binned_scores:
                    continue
                
                # Balanced margin sampling
                available_bins = [bin_idx for bin_idx, pairs in self.binned_scores[qid].items() if pairs]
                if not available_bins:
                    continue
                
                bin_idx = random.choice(available_bins)
                pair = random.choice(self.binned_scores[qid][bin_idx])
                pos_doc_id, neg_doc_id, pos_score, neg_score = pair
                
                if pos_doc_id not in self.collection or neg_doc_id not in self.collection:
                    continue
                
                batch_samples.append({
                    'query_id': qid,
                    'query_text': self.queries[qid],
                    'pos_doc_id': pos_doc_id,
                    'pos_doc_text': self.collection[pos_doc_id],
                    'neg_doc_id': neg_doc_id,
                    'neg_doc_text': self.collection[neg_doc_id],
                    'pos_score': pos_score,
                    'neg_score': neg_score
                })
        
        # Convert to tensors
        return self._create_batch_tensors(batch_samples[:self.batch_size])
    
    def _create_batch_tensors(self, samples: List[Dict]) -> Dict[str, torch.Tensor]:
        """Convert samples to tokenized tensors"""
        queries = [s['query_text'] for s in samples]
        pos_docs = [s['pos_doc_text'] for s in samples]
        neg_docs = [s['neg_doc_text'] for s in samples]
        
        query_tokens = tokenize_batch(queries, self.tokenizer, self.max_query_length)
        pos_doc_tokens = tokenize_batch(pos_docs, self.tokenizer, self.max_doc_length)
        neg_doc_tokens = tokenize_batch(neg_docs, self.tokenizer, self.max_doc_length)
        
        pos_scores = torch.tensor([s['pos_score'] for s in samples], dtype=torch.float)
        neg_scores = torch.tensor([s['neg_score'] for s in samples], dtype=torch.float)
        
        return {
            'query_tokens': query_tokens,
            'pos_doc_tokens': pos_doc_tokens,
            'neg_doc_tokens': neg_doc_tokens,
            'pos_scores': pos_scores,
            'neg_scores': neg_scores
        }

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_step(student_model: BERTDot, teacher_model: ColBERT, batch: Dict[str, torch.Tensor],
               pairwise_loss_fn: MarginMSELoss, inbatch_loss_fn: MarginMSELoss,
               optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler,
               device: torch.device, pairwise_weight: float = 1.0, inbatch_weight: float = 0.75) -> Dict[str, float]:
    """Single training step with dual supervision"""
    
    # Move batch to device
    for key, value in batch.items():
        if isinstance(value, dict):
            batch[key] = {k: v.to(device) for k, v in value.items()}
        elif isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    
    with torch.cuda.amp.autocast():
        # Student forward pass
        student_pos_scores, query_vecs, pos_doc_vecs = student_model(
            batch['query_tokens'], batch['pos_doc_tokens']
        )
        student_neg_scores, _, neg_doc_vecs = student_model(
            batch['query_tokens'], batch['neg_doc_tokens']
        )
        
        # Pairwise loss with static teacher scores
        pairwise_loss = pairwise_loss_fn(
            student_pos_scores, student_neg_scores,
            batch['pos_scores'], batch['neg_scores']
        )
        
        # In-batch negative loss with dynamic ColBERT teacher
        with torch.no_grad():
            teacher_pos_scores, teacher_query_vecs, teacher_pos_doc_vecs = teacher_model(
                batch['query_tokens'], batch['pos_doc_tokens']
            )
            teacher_neg_scores, _, teacher_neg_doc_vecs = teacher_model(
                batch['query_tokens'], batch['neg_doc_tokens']
            )
        
        # In-batch negative loss with dynamic ColBERT teacher using proper aggregation
        with torch.no_grad():
            teacher_pos_scores, teacher_query_vecs, teacher_pos_doc_vecs = teacher_model(
                batch['query_tokens'], batch['pos_doc_tokens']
            )
            teacher_neg_scores, _, teacher_neg_doc_vecs = teacher_model(
                batch['query_tokens'], batch['neg_doc_tokens']
            )
            
            # Use ColBERT's in-batch aggregation method
            # Combine pos and neg doc vectors for all-pairs scoring
            all_teacher_doc_vecs = torch.cat([teacher_pos_doc_vecs, teacher_neg_doc_vecs], dim=0)
            all_doc_masks = torch.cat([batch['pos_doc_tokens']['attention_mask'], 
                                     batch['neg_doc_tokens']['attention_mask']], dim=0)
            
            # Get teacher scores for all query-document pairs in batch
            teacher_inbatch_scores = teacher_model.forward_inbatch_aggregation(
                teacher_query_vecs, batch['query_tokens']['attention_mask'],
                all_teacher_doc_vecs, all_doc_masks
            )  # [batch_size, 2*batch_size]
        
        # Student in-batch scores using same aggregation pattern
        all_student_doc_vecs = torch.cat([pos_doc_vecs, neg_doc_vecs], dim=0)
        
        # Compute student scores for all pairs (mimicking ColBERT aggregation but for BERT_DOT)
        student_inbatch_scores = torch.mm(query_vecs, all_student_doc_vecs.t())  # [batch_size, 2*batch_size]
        
        # Apply in-batch loss following Equation (7)
        inbatch_loss = 0
        for i in range(batch_size):
            student_pos_score = student_inbatch_scores[i, i]  # Query i with its positive (at index i)
            teacher_pos_score = teacher_inbatch_scores[i, i]
            
            # Compare with all other documents in batch
            for j in range(2 * batch_size):
                if j != i:  # Skip self-pairing
                    student_neg_score = student_inbatch_scores[i, j]
                    teacher_neg_score = teacher_inbatch_scores[i, j]
                    
                    inbatch_loss += inbatch_loss_fn(
                        student_pos_score.unsqueeze(0),
                        student_neg_score.unsqueeze(0), 
                        teacher_pos_score.unsqueeze(0),
                        teacher_neg_score.unsqueeze(0)
                    )
        
        # Apply normalization factor from Equation (7): 1/(2*|Q|)
        inbatch_loss = inbatch_loss / (2 * batch_size)
        
        # Combined loss
        total_loss = pairwise_weight * pairwise_loss + inbatch_weight * inbatch_loss
    
    # Backward pass
    optimizer.zero_grad()
    scaler.scale(total_loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    return {
        'total_loss': total_loss.item(),
        'pairwise_loss': pairwise_loss.item(),
        'inbatch_loss': inbatch_loss.item()
    }

def evaluate_model(model: BERTDot, eval_queries: Dict[str, str], eval_collection: Dict[str, str],
                   qrels: Dict[str, Dict[str, int]], tokenizer, device: torch.device,
                   max_query_length: int = 30, max_doc_length: int = 200,
                   batch_size: int = 64) -> Dict[str, float]:
    """Simple evaluation function - compute MRR@10"""
    model.eval()
    
    results = {}
    
    with torch.no_grad():
        for qid, query_text in tqdm(eval_queries.items(), desc="Evaluating"):
            if qid not in qrels:
                continue
            
            # Get relevant documents
            relevant_docs = qrels[qid]
            if not relevant_docs:
                continue
            
            # Score all documents in collection (simplified - in practice you'd use retrieval)
            doc_scores = {}
            
            # Process in batches
            doc_ids = list(eval_collection.keys())
            for i in range(0, len(doc_ids), batch_size):
                batch_doc_ids = doc_ids[i:i+batch_size]
                batch_doc_texts = [eval_collection[doc_id] for doc_id in batch_doc_ids]
                
                query_batch = [query_text] * len(batch_doc_ids)
                
                query_tokens = tokenize_batch(query_batch, tokenizer, max_query_length)
                doc_tokens = tokenize_batch(batch_doc_texts, tokenizer, max_doc_length)
                
                # Move to device
                query_tokens = {k: v.to(device) for k, v in query_tokens.items()}
                doc_tokens = {k: v.to(device) for k, v in doc_tokens.items()}
                
                scores = model(query_tokens, doc_tokens)
                
                for j, doc_id in enumerate(batch_doc_ids):
                    doc_scores[doc_id] = scores[j].item()
            
            # Sort by score and calculate MRR
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            
            for rank, (doc_id, score) in enumerate(sorted_docs[:10], 1):
                if doc_id in relevant_docs and relevant_docs[doc_id] > 0:
                    results[qid] = 1.0 / rank
                    break
            else:
                results[qid] = 0.0
    
    model.train()
    return {'MRR@10': sum(results.values()) / len(results) if results else 0.0}

# ============================================================================
# EARLY STOPPING
# ============================================================================

class EarlyStopper:
    """Simple early stopping implementation"""
    
    def __init__(self, patience: int = 30, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.stop = False
    
    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = val_score
            self.counter = 0
        
        return self.stop

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_tas_balanced_model(
    # Data paths
    query_file: str,
    collection_file: str,
    cluster_file: str,
    teacher_scores_file: str,
    eval_query_file: str = None,
    eval_qrels_file: str = None,
    
    # Model parameters
    student_model_name: str = "distilbert-base-uncased",
    teacher_model_name: str = "sebastian-hofstaetter/colbert-distilbert-margin_mse-T2-msmarco",
    
    # Training parameters
    batch_size: int = 32,
    clusters_per_batch: int = 1,
    max_query_length: int = 30,
    max_doc_length: int = 200,
    learning_rate: float = 2e-5,
    num_epochs: int = 10,
    eval_steps: int = 4000,
    
    # Loss weights
    pairwise_weight: float = 1.0,
    inbatch_weight: float = 0.75,
    
    # Early stopping
    patience: int = 30,
    
    # Output
    save_path: str = "./models/tas_balanced_model",
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Main training function"""
    
    print(f"Using device: {device}")
    device = torch.device(device)
    
    # Load data
    print("Loading data...")
    queries = load_queries(query_file)
    collection = load_collection(collection_file)
    clusters = load_clusters(cluster_file)
    teacher_scores = load_teacher_scores(teacher_scores_file)
    
    print(f"Loaded {len(queries)} queries, {len(collection)} documents, {len(clusters)} clusters")
    
    # Load evaluation data if provided
    eval_queries = None
    eval_qrels = None
    if eval_query_file and eval_qrels_file:
        eval_queries = load_queries(eval_query_file)
        # Load qrels (simplified format: qid doc_id relevance)
        eval_qrels = defaultdict(dict)
        with open(eval_qrels_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    qid, _, doc_id, relevance = parts[:4]
                    eval_qrels[qid][doc_id] = int(relevance)
        print(f"Loaded {len(eval_queries)} evaluation queries")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    
    # Initialize models
    print("Initializing models...")
    student_model = BERTDot(student_model_name, return_vecs=True).to(device)
    teacher_model = ColBERT(teacher_model_name).to(device)
    teacher_model.eval()  # Teacher is always in eval mode
    
    # Initialize sampler
    sampler = TASBalancedSampler(
        queries, collection, clusters, teacher_scores, tokenizer,
        batch_size, clusters_per_batch, max_query_length, max_doc_length
    )
    
    # Initialize training components
    optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()
    pairwise_loss_fn = MarginMSELoss()
    inbatch_loss_fn = MarginMSELoss()
    early_stopper = EarlyStopper(patience=patience)
    
    # Training loop
    print("Starting training...")
    student_model.train()
    
    global_step = 0
    best_score = 0.0
    
    for epoch in range(num_epochs):
        if early_stopper.stop:
            break
            
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_losses = []
        
        # Calculate steps per epoch (estimate)
        steps_per_epoch = 1000  # Adjust based on your data size
        
        with tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}") as pbar:
            for step in pbar:
                # Sample batch
                batch = sampler.sample_batch()
                
                # Training step
                losses = train_step(
                    student_model, teacher_model, batch,
                    pairwise_loss_fn, inbatch_loss_fn, optimizer, scaler, device,
                    pairwise_weight, inbatch_weight
                )
                
                epoch_losses.append(losses['total_loss'])
                global_step += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{losses['total_loss']:.4f}",
                    'pair': f"{losses['pairwise_loss']:.4f}",
                    'inbatch': f"{losses['inbatch_loss']:.4f}"
                })
                
                # Evaluation
                if eval_queries and global_step % eval_steps == 0:
                    print("\nEvaluating...")
                    eval_results = evaluate_model(
                        student_model, eval_queries, collection, eval_qrels,
                        tokenizer, device, max_query_length, max_doc_length
                    )
                    
                    print(f"Step {global_step}: MRR@10 = {eval_results['MRR@10']:.4f}")
                    
                    # Early stopping check
                    if early_stopper(eval_results['MRR@10']):
                        print("Early stopping triggered!")
                        break
                    
                    # Save best model
                    if eval_results['MRR@10'] > best_score:
                        best_score = eval_results['MRR@10']
                        print(f"New best score: {best_score:.4f}")
                        os.makedirs(save_path, exist_ok=True)
                        torch.save(student_model.state_dict(), os.path.join(save_path, "best_model.pt"))
                        
                        # Save config
                        config = {
                            'model_name': student_model_name,
                            'max_query_length': max_query_length,
                            'max_doc_length': max_doc_length,
                            'best_score': best_score,
                            'global_step': global_step
                        }
                        with open(os.path.join(save_path, "config.json"), 'w') as f:
                            json.dump(config, f, indent=2)
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
        
        if early_stopper.stop:
            break
    
    print(f"\nTraining completed! Best MRR@10: {best_score:.4f}")
    return student_model

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example configuration - update paths to your data
    config = {
        "query_file": "/path/to/queries.tsv",
        "collection_file": "/path/to/collection.tsv", 
        "cluster_file": "/path/to/cluster_assignments.tsv",
        "teacher_scores_file": "/path/to/teacher_scores.tsv",
        "eval_query_file": "/path/to/eval_queries.tsv",
        "eval_qrels_file": "/path/to/eval_qrels.tsv",
        "batch_size": 32,
        "learning_rate": 2e-5,
        "num_epochs": 10,
        "save_path": "./models/tas_balanced_model"
    }
    
    # Train the model
    model = train_tas_balanced_model(**config)
    
    print("Training completed!")

# ============================================================================
# INFERENCE FUNCTION
# ============================================================================

def load_trained_model(model_path: str, device: str = "cuda"):
    """Load a trained TAS-Balanced model for inference"""
    
    # Load config
    with open(os.path.join(model_path, "config.json"), 'r') as f:
        config = json.load(f)
    
    # Initialize model
    model = BERTDot(config['model_name'], return_vecs=False)
    model.load_state_dict(torch.load(os.path.join(model_path, "best_model.pt"), map_location=device))
    model.to(device)
    model.eval()
    
    return model, config

def encode_queries_and_docs(model: BERTDot, texts: List[str], tokenizer, 
                           max_length: int, device: torch.device, batch_size: int = 64):
    """Encode a list of texts into dense representations"""
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            tokens = tokenize_batch(batch_texts, tokenizer, max_length)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            
            embeddings = model.forward_representation(tokens)
            all_embeddings.append(embeddings.cpu())
    
    return torch.cat(all_embeddings, dim=0)

def search_with_trained_model(model_path: str, query: str, documents: Dict[str, str], 
                             top_k: int = 10, device: str = "cuda"):
    """Search documents using trained TAS-Balanced model"""
    
    # Load model
    model, config = load_trained_model(model_path, device)
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    device = torch.device(device)
    
    # Encode query
    query_tokens = tokenize_batch([query], tokenizer, config['max_query_length'])
    query_tokens = {k: v.to(device) for k, v in query_tokens.items()}
    
    with torch.no_grad():
        query_embedding = model.forward_representation(query_tokens)
    
    # Encode documents and compute scores
    doc_ids = list(documents.keys())
    doc_texts = list(documents.values())
    
    doc_embeddings = encode_queries_and_docs(
        model, doc_texts, tokenizer, config['max_doc_length'], device
    ).to(device)
    
    # Compute similarities
    scores = torch.mm(query_embedding, doc_embeddings.t()).squeeze(0)
    
    # Get top-k results
    top_scores, top_indices = torch.topk(scores, min(top_k, len(doc_ids)))
    
    results = []
    for score, idx in zip(top_scores.cpu().tolist(), top_indices.cpu().tolist()):
        results.append({
            'doc_id': doc_ids[idx],
            'score': score,
            'text': doc_texts[idx]
        })
    
    return results

# ============================================================================
# SIMPLIFIED TRAINING SCRIPT FOR QUICK TESTING
# ============================================================================

def quick_train_example():
    """Quick training example with synthetic data for testing"""
    
    print("Creating synthetic data for testing...")
    
    # Create synthetic queries
    synthetic_queries = {
        f"q{i}": f"What is topic {i % 5} about?" 
        for i in range(100)
    }
    
    # Create synthetic documents
    synthetic_docs = {
        f"d{i}": f"This document discusses topic {i % 5} in great detail. " * 10
        for i in range(500)
    }
    
    # Create synthetic clusters (5 clusters of 20 queries each)
    synthetic_clusters = []
    for cluster_id in range(5):
        cluster = [f"q{i}" for i in range(cluster_id * 20, (cluster_id + 1) * 20)]
        synthetic_clusters.append(cluster)
    
    # Create synthetic teacher scores
    synthetic_teacher_scores = defaultdict(list)
    for qid in synthetic_queries.keys():
        topic_id = int(qid[1:]) % 5
        # Relevant docs have higher scores for same topic
        for _ in range(10):  # 10 pairs per query
            pos_doc_id = f"d{topic_id * 100 + random.randint(0, 99)}"  # Same topic
            neg_doc_id = f"d{random.randint(0, 499)}"  # Random doc
            
            pos_score = random.uniform(0.7, 1.0)  # High score for relevant
            neg_score = random.uniform(0.0, 0.3)  # Low score for non-relevant
            
            synthetic_teacher_scores[qid].append((pos_doc_id, neg_doc_id, pos_score, neg_score))
    
    # Initialize components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Initialize models
    student_model = BERTDot("distilbert-base-uncased", return_vecs=True).to(device)
    # For testing, we'll use the student model as teacher too (simplified)
    teacher_model = BERTDot("distilbert-base-uncased", return_vecs=True).to(device)
    teacher_model.eval()
    
    # Initialize sampler
    sampler = TASBalancedSampler(
        synthetic_queries, synthetic_docs, synthetic_clusters, synthetic_teacher_scores, 
        tokenizer, batch_size=8, clusters_per_batch=1
    )
    
    # Training setup
    optimizer = torch.optim.Adam(student_model.parameters(), lr=2e-5)
    scaler = torch.cuda.amp.GradScaler()
    pairwise_loss_fn = MarginMSELoss()
    inbatch_loss_fn = MarginMSELoss()
    
    print("Starting quick training test...")
    student_model.train()
    
    # Quick training loop (just a few steps)
    for step in range(10):
        batch = sampler.sample_batch()
        
        # Move batch to device
        for key, value in batch.items():
            if isinstance(value, dict):
                batch[key] = {k: v.to(device) for k, v in value.items()}
            elif isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
        
        # Simple training step (without in-batch negatives for simplicity)
        with torch.cuda.amp.autocast():
            student_pos_scores = student_model(batch['query_tokens'], batch['pos_doc_tokens'])
            student_neg_scores = student_model(batch['query_tokens'], batch['neg_doc_tokens'])
            
            loss = pairwise_loss_fn(
                student_pos_scores, student_neg_scores,
                batch['pos_scores'], batch['neg_scores']
            )
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print(f"Step {step + 1}: Loss = {loss.item():.4f}")
    
    print("Quick training test completed!")
    return student_model

# ============================================================================
# DATA PREPROCESSING UTILITIES
# ============================================================================

def create_qrels_from_trec_format(trec_file: str, output_file: str):
    """Convert TREC format qrels to simple format"""
    with open(trec_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, doc_id, relevance = parts[:4]
                outfile.write(f"{qid}\t{doc_id}\t{relevance}\n")

def create_candidate_file_from_run(run_file: str, qrels_file: str, output_file: str, top_k: int = 1000):
    """Create candidate file from TREC run file"""
    
    # Load qrels
    qrels = defaultdict(dict)
    with open(qrels_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, doc_id, relevance = parts[:4]
                qrels[qid][doc_id] = int(relevance)
    
    # Process run file
    candidates = defaultdict(list)
    with open(run_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                qid, _, doc_id, rank, score, _ = parts[:6]
                candidates[qid].append((doc_id, float(score), int(rank)))
    
    # Write candidates with relevance labels
    with open(output_file, 'w') as f:
        for qid, docs in candidates.items():
            # Sort by rank and take top-k
            docs.sort(key=lambda x: x[2])  # Sort by rank
            for doc_id, score, rank in docs[:top_k]:
                relevance = qrels.get(qid, {}).get(doc_id, 0)
                f.write(f"{qid}\t{doc_id}\t{relevance}\t{rank}\t{score}\n")

def validate_data_files(query_file: str, collection_file: str, cluster_file: str, 
                       teacher_scores_file: str):
    """Validate that all required data files exist and have correct format"""
    
    files_to_check = [
        (query_file, "Query file"),
        (collection_file, "Collection file"), 
        (cluster_file, "Cluster file"),
        (teacher_scores_file, "Teacher scores file")
    ]
    
    for file_path, file_type in files_to_check:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_type} not found: {file_path}")
        
        print(f"✓ {file_type} found: {file_path}")
        
        # Check first few lines for format
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 3:  # Check first 3 lines
                    break
                print(f"  Line {i+1}: {line.strip()[:100]}...")
        print()

# ============================================================================
# CONFIGURATION TEMPLATES
# ============================================================================

MSMARCO_CONFIG_TEMPLATE = {
    "query_file": "/path/to/msmarco-passage/queries.train.tsv",
    "collection_file": "/path/to/msmarco-passage/collection.tsv",
    "cluster_file": "/path/to/clustering_output/cluster-assignment-ids.tsv", 
    "teacher_scores_file": "/path/to/teacher_scores/pairwise_teacher_scores.tsv",
    "eval_query_file": "/path/to/msmarco-passage/queries.dev.small.tsv",
    "eval_qrels_file": "/path/to/msmarco-passage/qrels.dev.small.tsv",
    
    "student_model_name": "distilbert-base-uncased",
    "teacher_model_name": "sebastian-hofstaetter/colbert-distilbert-margin_mse-T2-msmarco",
    
    "batch_size": 32,
    "clusters_per_batch": 1,
    "max_query_length": 30,
    "max_doc_length": 200,
    "learning_rate": 2e-5,
    "num_epochs": 10,
    "eval_steps": 4000,
    
    "pairwise_weight": 1.0,
    "inbatch_weight": 0.75,
    "patience": 30,
    
    "save_path": "./models/tas_balanced_msmarco"
}

print("TAS-Balanced Dense Retrieval Training Notebook loaded successfully!")
print("\nTo get started:")
print("1. Update the paths in MSMARCO_CONFIG_TEMPLATE to point to your data files")
print("2. Run: model = train_tas_balanced_model(**MSMARCO_CONFIG_TEMPLATE)")
print("3. For quick testing with synthetic data, run: quick_train_example()")
print("\nMake sure you have the following data files ready:")
print("- queries.tsv (query_id \\t query_text)")  
print("- collection.tsv (doc_id \\t doc_text)")
print("- cluster_assignments.tsv (space-separated query_ids per line)")
print("- teacher_scores.tsv (pos_score neg_score query_id pos_doc_id neg_doc_id)")