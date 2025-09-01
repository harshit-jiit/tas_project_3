from rich.console import Console
import random
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import traceback
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Tuple, Optional
import logging

class ModernTokenizer:
    """
    Modern wrapper for Hugging Face AutoTokenizer to replace FastTransformerTokenizer
    """
    def __init__(self, model_name: str):
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Ensure we have a pad token
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def tokenize(self, text: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """
        Tokenize text and return tensors
        """
        encoded = self._tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding=False,  # We'll handle padding in collate_fn
            return_tensors="pt",
            return_attention_mask=True
        )
        
        # Squeeze to remove batch dimension for individual samples
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }

class TensorDict(Dict[str, torch.Tensor]):
    """
    Simple replacement for AllenNLP's TensorDict
    """
    pass

class WorkerError(Exception):
    """
    Custom exception for worker process errors
    """
    def __init__(self, message, traceback_str):
        super().__init__(message)
        self.traceback_str = traceback_str

class TASBalancedDatasetLoader:
    """
    Modernized version of TASBalancedDatasetLoader that dynamically samples queries 
    from given cluster information for a batch.
    
    Key improvements:
    - Replaced AllenNLP components with PyTorch + Transformers
    - Uses modern AutoTokenizer from Hugging Face
    - Maintains the same TAS (Teacher-As-Student) balanced sampling logic
    - Compatible with Python 3.12 and latest packages
    """

    def __init__(
        self,
        query_file: str,
        collection_file: str,
        pairs_with_teacher_scores: str,
        query_cluster_file: str,
        batch_size: int,
        clusters_per_batch: int,
        tokenizer_name: str,  # Changed from tokenizer object to model name
        max_doc_length: int = 512,
        max_query_length: int = 64,
        pair_balancing_strategy: str = "bins",  # "bins", "random", or "hard-margin"
        random_seed: int = 42,
    ):
        self.query_file = query_file
        self.collection_file = collection_file
        self.pairs_with_teacher_scores = pairs_with_teacher_scores
        self.query_cluster_file = query_cluster_file
        self.batch_size = batch_size
        self.clusters_per_batch = clusters_per_batch

        # Initialize modern tokenizer
        self._tokenizer = ModernTokenizer(tokenizer_name)

        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length

        self.read_with_scores = True
        self.pair_balancing_strategy = pair_balancing_strategy
        self.uniform_percentile_sampling = pair_balancing_strategy == "bins"
        self.uniform_percentile_sampling_bins = 10

        self.seed = random_seed
        
        # Data storage
        self.collection = {}
        self.collection_ids = []
        self.queries = {}
        self.pairs_with_teacher_scores_by_qid = defaultdict(list)
        self.query_clusters = []
        self.query_ids = set()

    def __iter__(self) -> Iterator[TensorDict]:
        """
        Create an iterator that yields batches of training data
        """
        ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
        queue: mp.JoinableQueue = ctx.JoinableQueue(2000)
        worker = ctx.Process(
            target=self.data_loader_subprocess, args=(queue,), daemon=True
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
            if hasattr(queue, "close"):
                queue.close()
            if worker.is_alive():
                worker.terminate()

    def load_data(self):
        """
        Load all necessary data files: collection, queries, pairs, and clusters
        """
        console = Console()

        # Load collection (documents)
        console.log("[TASBalanced] Loading collection from:", self.collection_file)
        with open(self.collection_file, "r", encoding="utf8") as cf:
            for line in cf:
                ls = line.split("\t")
                if len(ls) >= 2:
                    doc_id = ls[0]
                    # Truncate very long documents to prevent memory issues
                    doc_text = ls[1].rstrip()[:100_000]
                    self.collection[doc_id] = doc_text
                    self.collection_ids.append(doc_id)

        # Load queries
        console.log("[TASBalanced] Loading queries from:", self.query_file)
        with open(self.query_file, "r", encoding="utf8") as qf:
            for line in qf:
                ls = line.split("\t")
                if len(ls) >= 2:
                    query_id = ls[0]
                    query_text = ls[1].rstrip()
                    self.queries[query_id] = query_text

        # Load pairs with teacher scores
        console.log("[TASBalanced] Loading pairs from:", self.pairs_with_teacher_scores)
        with open(self.pairs_with_teacher_scores, "r", encoding="utf8") as qf:
            for line in qf:
                ls = line.split()
                if len(ls) >= 5:
                    pos_score = float(ls[0])
                    neg_score = float(ls[1])
                    query_id = ls[2]
                    pos_doc_id = ls[3]
                    neg_doc_id = ls[4].strip()
                    
                    # Store as (pos_doc_id, neg_doc_id, pos_score, neg_score)
                    self.pairs_with_teacher_scores_by_qid[query_id].append(
                        (pos_doc_id, neg_doc_id, pos_score, neg_score)
                    )

        # Create balanced bins if using uniform percentile sampling
        if self.uniform_percentile_sampling:
            console.log("[TASBalanced] Creating balanced bins")
            self._create_balanced_bins()

        # Load query clusters
        console.log("[TASBalanced] Loading cluster assignments from:", self.query_cluster_file)
        with open(self.query_cluster_file, "r", encoding="utf8") as qf:
            for line in qf:
                cluster_query_ids = line.split()
                if cluster_query_ids:  # Only add non-empty clusters
                    self.query_clusters.append(cluster_query_ids)

        # Find valid query IDs (intersection of pairs and cluster files)
        all_cluster_ids = [qid for cluster in self.query_clusters for qid in cluster]
        self.query_ids = set(self.pairs_with_teacher_scores_by_qid.keys()).intersection(
            set(all_cluster_ids)
        )

        # Clean clusters to only include valid query IDs
        for i, cluster in enumerate(self.query_clusters):
            self.query_clusters[i] = list(set(cluster).intersection(self.query_ids))
        
        # Remove empty clusters
        self.query_clusters = [c for c in self.query_clusters if len(c) > 0]

        console.log(
            f"[TASBalanced] Done loading! Using {len(self.query_ids)} queries from "
            f"{len(self.query_clusters)} clusters for seed: {self.seed} with "
            f"pair_balancing_strategy: {self.pair_balancing_strategy}"
        )

    def _create_balanced_bins(self):
        """
        Create balanced bins for uniform percentile sampling
        """
        pairs_with_teacher_scores_by_qid_binned = defaultdict(list)
        avg_bin_lengths = [[] for _ in range(self.uniform_percentile_sampling_bins)]
        
        for q_id, pair_list in self.pairs_with_teacher_scores_by_qid.items():
            if len(pair_list) >= 2:
                # Calculate margins (positive_score - negative_score)
                margins = np.array([pair[2] - pair[3] for pair in pair_list])
                
                # Create bins based on margin percentiles
                min_margin, max_margin = np.min(margins), np.max(margins)
                if min_margin == max_margin:
                    # If all margins are the same, put everything in first bin
                    bins = [pair_list] + [[] for _ in range(self.uniform_percentile_sampling_bins - 1)]
                else:
                    bin_edges = np.linspace(min_margin, max_margin, self.uniform_percentile_sampling_bins + 1)
                    indices = np.digitize(margins, bin_edges[1:-1])  # Exclude first and last edges
                    
                    bins = [[] for _ in range(self.uniform_percentile_sampling_bins)]
                    for i, pair in enumerate(pair_list):
                        bin_idx = min(indices[i], self.uniform_percentile_sampling_bins - 1)
                        bins[bin_idx].append(pair)
                
                # Record bin lengths for statistics
                for i, bin_content in enumerate(bins):
                    avg_bin_lengths[i].append(len(bin_content))
                
                pairs_with_teacher_scores_by_qid_binned[q_id] = bins
        
        self.pairs_with_teacher_scores_by_qid = pairs_with_teacher_scores_by_qid_binned

    def data_loader_subprocess(self, queue):
        """
        Subprocess that generates batches and puts them in the queue
        """
        # Set random seeds for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        try:
            self.load_data()
            query_target_count = max(1, int(self.batch_size / self.clusters_per_batch))

            while True:
                batch_samples = []

                while len(batch_samples) < self.batch_size:
                    # Randomly select a cluster
                    cluster_idx = random.randint(0, len(self.query_clusters) - 1)
                    cluster = self.query_clusters[cluster_idx]

                    # Sample queries from the selected cluster
                    if query_target_count < len(cluster):
                        selected_query_ids = random.sample(cluster, query_target_count)
                    else:
                        selected_query_ids = cluster

                    # Process each selected query
                    for query_id in selected_query_ids:
                        if len(batch_samples) >= self.batch_size:
                            break

                        if query_id not in self.queries:
                            continue

                        # Get tokenized query
                        query_tokens = self.get_tokenized_query(self.queries[query_id])

                        # Select a pair based on the balancing strategy
                        pair = self._select_pair(query_id)
                        if pair is None:
                            continue

                        pos_doc_id, neg_doc_id, pos_score, neg_score = pair

                        # Check if documents exist
                        if pos_doc_id not in self.collection or neg_doc_id not in self.collection:
                            continue

                        # Get tokenized documents
                        pos_doc_tokens = self.get_tokenized_document(self.collection[pos_doc_id])
                        neg_doc_tokens = self.get_tokenized_document(self.collection[neg_doc_id])

                        # Create sample
                        sample = {
                            "query_tokens": query_tokens,
                            "doc_pos_tokens": pos_doc_tokens,
                            "doc_neg_tokens": neg_doc_tokens,
                            "pos_score": torch.tensor(pos_score, dtype=torch.float32),
                            "neg_score": torch.tensor(neg_score, dtype=torch.float32),
                        }

                        batch_samples.append(sample)

                # Create batch by collating samples
                batch = self._collate_samples(batch_samples)
                queue.put((batch, None))

        except Exception as e:
            queue.put((None, (repr(e), traceback.format_exc())))
        
        queue.put((None, None))
        queue.join()

    def _select_pair(self, query_id: str) -> Optional[Tuple[str, str, float, float]]:
        """
        Select a positive-negative document pair for the given query
        """
        if query_id not in self.pairs_with_teacher_scores_by_qid:
            return None

        pairs = self.pairs_with_teacher_scores_by_qid[query_id]

        if self.uniform_percentile_sampling:
            # Select from bins
            attempts = 0
            while attempts < 10:  # Prevent infinite loops
                bin_idx = random.randint(0, len(pairs) - 1)
                if len(pairs[bin_idx]) > 0:
                    return random.choice(pairs[bin_idx])
                attempts += 1
            
            # Fallback: select from any non-empty bin
            non_empty_bins = [bin_content for bin_content in pairs if len(bin_content) > 0]
            if non_empty_bins:
                selected_bin = random.choice(non_empty_bins)
                return random.choice(selected_bin)
            return None
        else:
            # Random sampling
            if len(pairs) == 0:
                return None
            return random.choice(pairs)

    def get_tokenized_query(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize query text
        """
        return self._tokenizer.tokenize(text, max_length=self.max_query_length)

    def get_tokenized_document(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize document text
        """
        return self._tokenizer.tokenize(text, max_length=self.max_doc_length)

    def _collate_samples(self, samples: List[Dict[str, Any]]) -> TensorDict:
        """
        Collate individual samples into a batch with proper padding
        """
        if not samples:
            return TensorDict()

        # Extract sequences for padding
        query_input_ids = [sample["query_tokens"]["input_ids"] for sample in samples]
        query_attention_masks = [sample["query_tokens"]["attention_mask"] for sample in samples]
        
        pos_doc_input_ids = [sample["doc_pos_tokens"]["input_ids"] for sample in samples]
        pos_doc_attention_masks = [sample["doc_pos_tokens"]["attention_mask"] for sample in samples]
        
        neg_doc_input_ids = [sample["doc_neg_tokens"]["input_ids"] for sample in samples]
        neg_doc_attention_masks = [sample["doc_neg_tokens"]["attention_mask"] for sample in samples]

        # Pad sequences
        batch = TensorDict({
            "query_tokens": {
                "input_ids": self._pad_sequences(query_input_ids),
                "attention_mask": self._pad_sequences(query_attention_masks, pad_value=0)
            },
            "doc_pos_tokens": {
                "input_ids": self._pad_sequences(pos_doc_input_ids),
                "attention_mask": self._pad_sequences(pos_doc_attention_masks, pad_value=0)
            },
            "doc_neg_tokens": {
                "input_ids": self._pad_sequences(neg_doc_input_ids),
                "attention_mask": self._pad_sequences(neg_doc_attention_masks, pad_value=0)
            },
            "pos_score": torch.stack([sample["pos_score"] for sample in samples]),
            "neg_score": torch.stack([sample["neg_score"] for sample in samples])
        })

        return batch

    def _pad_sequences(self, sequences: List[torch.Tensor], pad_value: Optional[int] = None) -> torch.Tensor:
        """
        Pad sequences to the same length
        """
        if pad_value is None:
            pad_value = self._tokenizer._tokenizer.pad_token_id

        max_len = max(len(seq) for seq in sequences)
        padded = []
        
        for seq in sequences:
            pad_len = max_len - len(seq)
            if pad_len > 0:
                padding = torch.full((pad_len,), pad_value, dtype=seq.dtype)
                padded_seq = torch.cat([seq, padding])
            else:
                padded_seq = seq
            padded.append(padded_seq)
        
        return torch.stack(padded)


# Example usage and testing
if __name__ == "__main__":
    # Example of how to use the modernized loader
    loader = TASBalancedDatasetLoader(
        query_file="path/to/queries.tsv",
        collection_file="path/to/collection.tsv", 
        pairs_with_teacher_scores="path/to/pairs.tsv",
        query_cluster_file="path/to/clusters.txt",
        batch_size=32,
        clusters_per_batch=4,
        tokenizer_name="bert-base-uncased",  # Use any HuggingFace model
        max_doc_length=512,
        max_query_length=64,
        pair_balancing_strategy="bins",
        random_seed=42
    )
    
    print("Modernized TAS Balanced Dataset Loader initialized!")
    print("Key improvements:")
    print("- Replaced AllenNLP with PyTorch + Transformers")
    print("- Uses AutoTokenizer from Hugging Face")
    print("- Compatible with Python 3.12")
    print("- Maintains original TAS balanced sampling logic")
    
    # To iterate through batches:
    # for batch in loader:
    #     # batch contains tokenized queries, positive docs, negative docs, and scores
    #     print(f"Batch keys: {batch.keys()}")
    #     break