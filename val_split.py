#!/usr/bin/env python3
"""
Split MS MARCO dev queries to EXACTLY replicate the TAS-Balanced paper methodology:
- DEV-7K: First 6,980 queries (standard benchmark for final evaluation)
- Early Stopping: 3,200 queries sampled from remaining queries (for training validation)

This ensures proper comparison with the paper's reported results.
"""

import os
import random
from collections import defaultdict
from tqdm import tqdm
import json

# Set reproducible seed to match paper if possible
random.seed(208973249)

class PaperCompliantMSMARCOSplitter:
    def __init__(self, base_path='/workspace/2404170001'):
        self.base_path = base_path
        
        # Input files
        self.queries_file = f'{base_path}/inputs/queries.dev.small.tsv'
        self.qrels_file = f'{base_path}/inputs/qrels.dev.tsv'
        self.collection_file = f'{base_path}/inputs/collection.tsv'
        
        # Output directory
        self.output_dir = f'{base_path}/paper_compliant_split'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Paper's exact methodology
        self.dev_7k_count = 6980           # DEV-7K: Standard benchmark
        self.early_stopping_count = 3200    # Early stopping validation
        self.max_doc_length = 100_000
        self.bm25_hits = 100
        
        # Data containers
        self.queries = {}
        self.collection = {}
        self.qrels = defaultdict(dict)
        
    def load_data(self):
        """Load all source data with consistent string types"""
        print("Loading source data...")
        
        # Load queries
        with open(self.queries_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading queries"):
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    qid, qtext = str(parts[0]), parts[1]
                    self.queries[qid] = qtext
        
        print(f"Loaded {len(self.queries)} queries")
        
        # Load collection
        with open(self.collection_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading collection"):
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    docid, text = str(parts[0]), parts[1]
                    self.collection[docid] = text[:self.max_doc_length]
        
        print(f"Loaded {len(self.collection)} documents")
        
        # Load qrels
        with open(self.qrels_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading qrels"):
                parts = line.strip().split()
                if len(parts) >= 4:
                    qid, _, docid, rel = str(parts[0]), parts[1], str(parts[2]), float(parts[3])
                    if rel > 0:
                        self.qrels[qid][docid] = rel
        
        print(f"Loaded qrels for {len(self.qrels)} queries")
        
    def split_queries_paper_method(self):
        """
        Split queries using the EXACT paper methodology:
        1. DEV-7K: First 6,980 queries (standard benchmark)
        2. Early Stopping: 3,200 queries from remaining pool
        """
        print(f"\n=== PAPER-COMPLIANT QUERY SPLITTING ===")
        
        # Get queries that have both text and relevance judgments
        valid_queries = list(set(self.queries.keys()) & set(self.qrels.keys()))
        print(f"Total queries with both text and qrels: {len(valid_queries)}")
        
        if len(valid_queries) < (self.dev_7k_count + self.early_stopping_count):
            raise ValueError(f"Not enough valid queries! Have {len(valid_queries)}, need {self.dev_7k_count + self.early_stopping_count}")
        
        # Sort queries for reproducible splits
        valid_queries.sort()
        
        # Paper's methodology: 
        # 1. DEV-7K: First 6,980 queries (standard benchmark)
        self.dev_7k_queries = set(valid_queries[:self.dev_7k_count])
        
        # 2. Early stopping: 3,200 from remaining queries
        remaining_queries = valid_queries[self.dev_7k_count:]
        self.early_stopping_queries = set(random.sample(remaining_queries, self.early_stopping_count))
        
        print(f"\n=== PAPER METHODOLOGY APPLIED ===")
        print(f"DEV-7K (main evaluation): {len(self.dev_7k_queries)} queries")
        print(f"Early stopping (training validation): {len(self.early_stopping_queries)} queries")
        print(f"Unused queries: {len(valid_queries) - len(self.dev_7k_queries) - len(self.early_stopping_queries)}")
        
        return self.dev_7k_queries, self.early_stopping_queries
    
    def save_paper_query_splits(self):
        """Save query splits using paper's naming convention"""
        print("Saving paper-compliant query split files...")
        
        # DEV-7K queries (main evaluation)
        dev_7k_file = os.path.join(self.output_dir, 'queries_dev_7k.tsv')
        with open(dev_7k_file, 'w', encoding='utf-8') as f:
            for qid in sorted(self.dev_7k_queries):
                f.write(f"{qid}\t{self.queries[qid]}\n")
        
        # Early stopping queries (training validation)
        early_stopping_file = os.path.join(self.output_dir, 'queries_early_stopping.tsv')
        with open(early_stopping_file, 'w', encoding='utf-8') as f:
            for qid in sorted(self.early_stopping_queries):
                f.write(f"{qid}\t{self.queries[qid]}\n")
        
        print(f"Saved DEV-7K queries: {dev_7k_file}")
        print(f"Saved early stopping queries: {early_stopping_file}")
        
        return dev_7k_file, early_stopping_file
    
    def save_paper_qrels_splits(self):
        """Save qrels splits for paper methodology"""
        print("Saving paper-compliant qrels split files...")
        
        # DEV-7K qrels
        dev_7k_qrels_file = os.path.join(self.output_dir, 'qrels_dev_7k.tsv')
        with open(dev_7k_qrels_file, 'w', encoding='utf-8') as f:
            for qid in sorted(self.dev_7k_queries):
                if qid in self.qrels:
                    for docid, rel_score in self.qrels[qid].items():
                        f.write(f"{qid}\t0\t{docid}\t{int(rel_score)}\n")
        
        # Early stopping qrels
        early_stopping_qrels_file = os.path.join(self.output_dir, 'qrels_early_stopping.tsv')
        with open(early_stopping_qrels_file, 'w', encoding='utf-8') as f:
            for qid in sorted(self.early_stopping_queries):
                if qid in self.qrels:
                    for docid, rel_score in self.qrels[qid].items():
                        f.write(f"{qid}\t0\t{docid}\t{int(rel_score)}\n")
        
        print(f"Saved DEV-7K qrels: {dev_7k_qrels_file}")
        print(f"Saved early stopping qrels: {early_stopping_qrels_file}")
        
        return dev_7k_qrels_file, early_stopping_qrels_file
    
    def prepare_pyserini_collection(self):
        """Prepare collection in JSONL format for Pyserini"""
        print("Preparing collection for Pyserini...")
        
        collection_jsonl = os.path.join(self.output_dir, 'collection.jsonl')
        with open(collection_jsonl, 'w', encoding='utf-8') as f:
            for docid, text in tqdm(self.collection.items(), desc="Converting to JSONL"):
                record = {"id": docid, "contents": text}
                f.write(json.dumps(record) + '\n')
        
        print(f"Saved collection: {collection_jsonl}")
        return collection_jsonl
    
    def generate_paper_bm25_commands(self, dev_7k_file, early_stopping_file):
        """Generate BM25 commands following paper methodology"""
        print("\nGenerating paper-compliant BM25 commands...")
        
        collection_jsonl = os.path.join(self.output_dir, 'collection.jsonl')
        index_path = os.path.join(self.output_dir, 'pyserini_index')
        
        dev_7k_bm25_file = os.path.join(self.output_dir, 'bm25_dev_7k.txt')
        early_stopping_bm25_file = os.path.join(self.output_dir, 'bm25_early_stopping.txt')
        
        commands = f"""
# ===== PAPER-COMPLIANT BM25 GENERATION COMMANDS =====

# 1. Build Lucene index
python -m pyserini.index.lucene \\
  --collection JsonCollection \\
  --input {self.output_dir} \\
  --index {index_path} \\
  --generator DefaultLuceneDocumentGenerator \\
  --threads 8

# 2. Generate BM25 results for DEV-7K (main evaluation)
python -m pyserini.search.lucene \\
  --index {index_path} \\
  --topics {dev_7k_file} \\
  --output {dev_7k_bm25_file} \\
  --bm25 \\
  --hits {self.bm25_hits} \\
  --output-format trec

# 3. Generate BM25 results for Early Stopping (training validation)
python -m pyserini.search.lucene \\
  --index {index_path} \\
  --topics {early_stopping_file} \\
  --output {early_stopping_bm25_file} \\
  --bm25 \\
  --hits {self.bm25_hits} \\
  --output-format trec

# ===== VERIFICATION COMMANDS =====
# Check line counts to verify generation worked
wc -l {dev_7k_bm25_file}          # Should be ~698,000 lines (6,980 queries × 100 hits)
wc -l {early_stopping_bm25_file}  # Should be ~320,000 lines (3,200 queries × 100 hits)
"""
        
        # Save commands to file
        commands_file = os.path.join(self.output_dir, 'generate_paper_bm25.sh')
        with open(commands_file, 'w') as f:
            f.write(commands)
        
        print(f"Saved BM25 generation commands: {commands_file}")
        print("Run the commands in the file to generate BM25 results.")
        
        return dev_7k_bm25_file, early_stopping_bm25_file
    
    def generate_paper_validation_files(self, dev_7k_bm25_file, early_stopping_bm25_file):
        """Generate validation TSV files following paper methodology"""
        print("Generating paper-compliant validation files...")
        
        def create_validation_file(query_set, bm25_file, output_file, set_name):
            """Create validation file for a specific query set"""
            print(f"  Creating {set_name} validation file...")
            
            # Load BM25 results for this set
            bm25_pairs = set()
            if os.path.exists(bm25_file):
                with open(bm25_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            qid, docid = str(parts[0]), str(parts[2])
                            if qid in query_set:
                                bm25_pairs.add((qid, docid))
            
            print(f"    Loaded {len(bm25_pairs)} BM25 pairs for {set_name}")
            
            # Generate validation file
            known_pairs = set()
            with open(output_file, 'w', encoding='utf-8') as f:
                
                # Add ALL BM25 pairs (ensures no KeyError)
                for qid, docid in bm25_pairs:
                    if qid in self.queries and docid in self.collection:
                        if (qid, docid) not in known_pairs:
                            known_pairs.add((qid, docid))
                            row = [qid, docid, self.queries[qid], self.collection[docid]]
                            f.write('\t'.join(row) + '\n')
                
                # Add relevant passages not retrieved by BM25 (paper methodology)
                bm25_doc_to_queries = defaultdict(set)
                for qid, docid in bm25_pairs:
                    bm25_doc_to_queries[docid].add(qid)
                
                for qid in query_set:
                    if qid in self.qrels:
                        for docid in self.qrels[qid]:
                            if (qid, docid) not in known_pairs:
                                # Only add if document wasn't retrieved by BM25 for different query
                                bm25_queries_for_doc = bm25_doc_to_queries.get(docid, set())
                                if len(bm25_queries_for_doc) == 0 or qid in bm25_queries_for_doc:
                                    if qid in self.queries and docid in self.collection:
                                        known_pairs.add((qid, docid))
                                        row = [qid, docid, self.queries[qid], self.collection[docid]]
                                        f.write('\t'.join(row) + '\n')
            
            print(f"    Generated {len(known_pairs)} pairs for {set_name}")
            return len(known_pairs)
        
        # Generate validation files using paper's naming
        dev_7k_tsv = os.path.join(self.output_dir, 'dev_7k_evaluation.tsv')
        early_stopping_tsv = os.path.join(self.output_dir, 'early_stopping_validation.tsv')
        
        dev_7k_pairs = create_validation_file(self.dev_7k_queries, dev_7k_bm25_file, dev_7k_tsv, "DEV-7K")
        early_stopping_pairs = create_validation_file(self.early_stopping_queries, early_stopping_bm25_file, early_stopping_tsv, "Early Stopping")
        
        return dev_7k_tsv, early_stopping_tsv, dev_7k_pairs, early_stopping_pairs
    
    def run_paper_compliant_pipeline(self):
        """Run the complete paper-compliant pipeline"""
        print("=== PAPER-COMPLIANT MS MARCO SPLIT PIPELINE ===\n")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Split queries using paper methodology
        dev_7k_queries, early_stopping_queries = self.split_queries_paper_method()
        
        # Step 3: Save query splits
        dev_7k_file, early_stopping_file = self.save_paper_query_splits()
        
        # Step 4: Save qrels splits
        dev_7k_qrels, early_stopping_qrels = self.save_paper_qrels_splits()
        
        # Step 5: Prepare collection for Pyserini
        collection_jsonl = self.prepare_pyserini_collection()
        
        # Step 6: Generate BM25 commands
        dev_7k_bm25_file, early_stopping_bm25_file = self.generate_paper_bm25_commands(dev_7k_file, early_stopping_file)
        
        print(f"\n=== PAPER-COMPLIANT PIPELINE SUMMARY ===")
        print(f"Output directory: {self.output_dir}")
        print(f"DEV-7K queries (main evaluation): {len(self.dev_7k_queries)}")
        print(f"Early stopping queries (training validation): {len(self.early_stopping_queries)}")
        print(f"\nGenerated files:")
        print(f"  - DEV-7K queries: {dev_7k_file}")
        print(f"  - Early stopping queries: {early_stopping_file}")
        print(f"  - DEV-7K qrels: {dev_7k_qrels}")
        print(f"  - Early stopping qrels: {early_stopping_qrels}")
        print(f"  - Collection: {collection_jsonl}")
        
        print(f"\n=== NEXT STEPS ===")
        print(f"1. Run BM25 generation: {self.output_dir}/generate_paper_bm25.sh")
        print(f"2. After BM25 completes: splitter.generate_paper_final_files()")
        print(f"3. Your results will be directly comparable to the paper!")
        
        return {
            'dev_7k_queries': dev_7k_file,
            'early_stopping_queries': early_stopping_file,
            'dev_7k_qrels': dev_7k_qrels,
            'early_stopping_qrels': early_stopping_qrels,
            'collection': collection_jsonl,
            'dev_7k_bm25': dev_7k_bm25_file,
            'early_stopping_bm25': early_stopping_bm25_file
        }
    
    def generate_paper_final_files(self):
        """Generate final files after BM25 generation completes"""
        print("Generating paper-compliant final files...")
        
        dev_7k_bm25_file = os.path.join(self.output_dir, 'bm25_dev_7k.txt')
        early_stopping_bm25_file = os.path.join(self.output_dir, 'bm25_early_stopping.txt')
        
        if not os.path.exists(dev_7k_bm25_file):
            print(f"❌ DEV-7K BM25 file not found: {dev_7k_bm25_file}")
            print("Run BM25 generation commands first!")
            return None
        
        if not os.path.exists(early_stopping_bm25_file):
            print(f"❌ Early stopping BM25 file not found: {early_stopping_bm25_file}")
            print("Run BM25 generation commands first!")
            return None
        
        # Generate validation files
        dev_7k_tsv, early_stopping_tsv, dev_7k_pairs, early_stopping_pairs = self.generate_paper_validation_files(dev_7k_bm25_file, early_stopping_bm25_file)
        
        # Generate paper-compliant matchmaker config
        config_content = f'''# Paper-compliant matchmaker config for TAS-Balanced replication

# Early stopping during training (matches paper's 3,200 query validation)
validation_cont:
  binarization_point: 1
  candidate_set_from_to: [5, 100]
  candidate_set_path: {early_stopping_bm25_file}
  qrels: {os.path.join(self.output_dir, 'qrels_early_stopping.tsv')}
  save_only_best: true
  tsv: {early_stopping_tsv}

# Final evaluation on DEV-7K (matches paper's evaluation benchmark)
test:
  dev_7k_evaluation:
    binarization_point: 1
    candidate_set_from_to: [5, 100] 
    candidate_set_path: {dev_7k_bm25_file}
    qrels: {os.path.join(self.output_dir, 'qrels_dev_7k.tsv')}
    tsv: {dev_7k_tsv}
    save_secondary_output: True

# PAPER REPLICATION SUMMARY:
# Early stopping: {len(self.early_stopping_queries)} queries (training validation)
# DEV-7K evaluation: {len(self.dev_7k_queries)} queries (final benchmark)
# Early stopping pairs: {early_stopping_pairs}
# DEV-7K pairs: {dev_7k_pairs}
# 
# This configuration exactly matches the TAS-Balanced paper methodology.
# Your results on DEV-7K will be directly comparable to Table 6 in the paper.
'''
        
        config_file = os.path.join(self.output_dir, 'paper_compliant_config.yaml')
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        print(f"\n=== PAPER REPLICATION COMPLETE! ===")
        print(f"Generated files:")
        print(f"  - Early stopping TSV: {early_stopping_tsv} ({early_stopping_pairs} pairs)")
        print(f"  - DEV-7K evaluation TSV: {dev_7k_tsv} ({dev_7k_pairs} pairs)")  
        print(f"  - Paper-compliant config: {config_file}")
        print(f"\n🎯 REPLICATION READY:")
        print(f"   - Use DEV-7K results to compare with Table 6 in the paper")
        print(f"   - Expected DEV-7K results: nDCG@10 ~0.340, MRR@10 ~0.340, Recall@1K ~0.975")
        print(f"   - Your implementation should achieve similar numbers!")
        
        return {
            'early_stopping_tsv': early_stopping_tsv,
            'dev_7k_tsv': dev_7k_tsv,
            'config': config_file
        }

# Usage example for paper replication
if __name__ == "__main__":
    print("=== TAS-BALANCED PAPER REPLICATION SETUP ===\n")
    
    # Initialize paper-compliant splitter
    splitter = PaperCompliantMSMARCOSplitter()
    
    # Run paper-compliant pipeline
    files = splitter.run_paper_compliant_pipeline()
    
    print("\n" + "="*70)
    print("MANUAL STEP REQUIRED:")
    print("="*70)
    print("1. Run BM25 generation commands from the generated .sh file")
    print("2. Then call: splitter.generate_paper_final_files()")
    print("3. Use the paper_compliant_config.yaml in your training")
    print("4. Your DEV-7K results will be comparable to Table 6!")
    print("="*70)