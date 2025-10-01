#!/usr/bin/env python3
"""
Split MS MARCO dev queries into validation (3200) and test (remaining) sets
Generate proper qrels, validation files, and BM25 candidates that are fully aligned
to prevent KeyError in calculate_metrics_along_candidate_depth
"""

import os
import random
from collections import defaultdict
from tqdm import tqdm
import json

# Set reproducible seed
random.seed(208973249)

class MSMARCODevSplitter:
    def __init__(self, base_path='/workspace/2404170001'):
        self.base_path = base_path
        
        # Input files
        self.queries_file = f'{base_path}/inputs/queries.dev.small.tsv'
        self.qrels_file = f'{base_path}/inputs/qrels.dev.tsv'
        self.collection_file = f'{base_path}/inputs/collection.tsv'
        
        # Output directory
        self.output_dir = f'{base_path}/validation_test_split'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Parameters
        self.validation_query_count = 3200
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
        
    def split_queries(self):
        """Split queries into validation and test sets"""
        print(f"\nSplitting queries into validation ({self.validation_query_count}) and test sets...")
        
        # Get queries that have both text and relevance judgments
        valid_queries = list(set(self.queries.keys()) & set(self.qrels.keys()))
        print(f"Queries with both text and qrels: {len(valid_queries)}")
        
        if len(valid_queries) < self.validation_query_count:
            raise ValueError(f"Not enough valid queries! Have {len(valid_queries)}, need {self.validation_query_count}")
        
        # Sample validation queries
        self.validation_queries = set(random.sample(valid_queries, self.validation_query_count))
        self.test_queries = set(valid_queries) - self.validation_queries
        
        print(f"Validation queries: {len(self.validation_queries)}")
        print(f"Test queries: {len(self.test_queries)}")
        
        return self.validation_queries, self.test_queries
    
    def save_query_splits(self):
        """Save query splits to files for BM25 generation"""
        print("Saving query split files...")
        
        # Validation queries
        val_query_file = os.path.join(self.output_dir, 'queries_validation.tsv')
        with open(val_query_file, 'w', encoding='utf-8') as f:
            for qid in sorted(self.validation_queries):
                f.write(f"{qid}\t{self.queries[qid]}\n")
        
        # Test queries  
        test_query_file = os.path.join(self.output_dir, 'queries_test.tsv')
        with open(test_query_file, 'w', encoding='utf-8') as f:
            for qid in sorted(self.test_queries):
                f.write(f"{qid}\t{self.queries[qid]}\n")
        
        print(f"Saved validation queries: {val_query_file}")
        print(f"Saved test queries: {test_query_file}")
        
        return val_query_file, test_query_file
    
    def save_qrels_splits(self):
        """Save qrels splits for validation and test"""
        print("Saving qrels split files...")
        
        # Validation qrels
        val_qrels_file = os.path.join(self.output_dir, 'qrels_validation.tsv')
        with open(val_qrels_file, 'w', encoding='utf-8') as f:
            for qid in sorted(self.validation_queries):
                if qid in self.qrels:
                    for docid, rel_score in self.qrels[qid].items():
                        f.write(f"{qid}\t0\t{docid}\t{int(rel_score)}\n")
        
        # Test qrels
        test_qrels_file = os.path.join(self.output_dir, 'qrels_test.tsv')
        with open(test_qrels_file, 'w', encoding='utf-8') as f:
            for qid in sorted(self.test_queries):
                if qid in self.qrels:
                    for docid, rel_score in self.qrels[qid].items():
                        f.write(f"{qid}\t0\t{docid}\t{int(rel_score)}\n")
        
        print(f"Saved validation qrels: {val_qrels_file}")
        print(f"Saved test qrels: {test_qrels_file}")
        
        return val_qrels_file, test_qrels_file
    
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
    
    def generate_bm25_commands(self, val_query_file, test_query_file):
        """Generate BM25 indexing and search commands"""
        print("\nGenerating BM25 commands...")
        
        collection_jsonl = os.path.join(self.output_dir, 'collection.jsonl')
        index_path = os.path.join(self.output_dir, 'pyserini_index')
        
        val_bm25_file = os.path.join(self.output_dir, 'bm25_validation.txt')
        test_bm25_file = os.path.join(self.output_dir, 'bm25_test.txt')
        
        commands = f"""
# ===== PYSERINI BM25 GENERATION COMMANDS =====

# 1. Build Lucene index
python -m pyserini.index.lucene \\
  --collection JsonCollection \\
  --input {self.output_dir} \\
  --index {index_path} \\
  --generator DefaultLuceneDocumentGenerator \\
  --threads 8

# 2. Generate BM25 results for VALIDATION queries
python -m pyserini.search.lucene \\
  --index {index_path} \\
  --topics {val_query_file} \\
  --output {val_bm25_file} \\
  --bm25 \\
  --hits {self.bm25_hits} \\
  --output-format trec

# 3. Generate BM25 results for TEST queries  
python -m pyserini.search.lucene \\
  --index {index_path} \\
  --topics {test_query_file} \\
  --output {test_bm25_file} \\
  --bm25 \\
  --hits {self.bm25_hits} \\
  --output-format trec

# ===== VERIFICATION COMMANDS =====
# Check line counts to verify generation worked
wc -l {val_bm25_file}   # Should be ~320,000 lines (3200 queries × 100 hits)
wc -l {test_bm25_file}  # Should be ~(remaining_queries × 100) lines
"""
        
        # Save commands to file
        commands_file = os.path.join(self.output_dir, 'generate_bm25.sh')
        with open(commands_file, 'w') as f:
            f.write(commands)
        
        print(f"Saved BM25 generation commands: {commands_file}")
        print("Run the commands in the file to generate BM25 results.")
        
        return val_bm25_file, test_bm25_file
    
    def generate_validation_files(self, bm25_val_file, bm25_test_file):
        """Generate validation TSV files that are guaranteed to align with BM25 candidates"""
        print("Generating aligned validation files...")
        
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
                
                # Add ALL BM25 pairs (this ensures no KeyError)
                for qid, docid in bm25_pairs:
                    if qid in self.queries and docid in self.collection:
                        if (qid, docid) not in known_pairs:
                            known_pairs.add((qid, docid))
                            row = [qid, docid, self.queries[qid], self.collection[docid]]
                            f.write('\t'.join(row) + '\n')
                
                # Add ONLY relevant documents that DON'T conflict with BM25
                bm25_doc_to_queries = defaultdict(set)
                for qid, docid in bm25_pairs:
                    bm25_doc_to_queries[docid].add(qid)
                
                # Add qrels documents that either:
                # 1. Weren't retrieved by BM25 at all, OR  
                # 2. Were retrieved by BM25 for the SAME query
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
        
        # Generate validation files
        validation_tsv = os.path.join(self.output_dir, 'validation.tsv')
        test_tsv = os.path.join(self.output_dir, 'test.tsv')
        
        val_pairs = create_validation_file(self.validation_queries, bm25_val_file, validation_tsv, "validation")
        test_pairs = create_validation_file(self.test_queries, bm25_test_file, test_tsv, "test")
        
        return validation_tsv, test_tsv, val_pairs, test_pairs
    
    def generate_candidate_set_parser(self):
        """Generate candidate set parsing function that matches validate_model expectations"""
        
        parser_code = '''
def parse_candidate_set(candidate_file_path, max_candidates_per_query):
    """
    Parse candidate set file in TREC format for use with calculate_metrics_along_candidate_depth
    Returns: dict mapping query_id -> {doc_id: rank_position}
    
    This matches the expected format in validate_model where:
    candidate_positions = np.array([candidates[d_id] for d_id in ranked_doc_ids])
    """
    
    candidate_ranking = {}
    
    with open(candidate_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid = str(parts[0])  # Ensure string type
                docid = str(parts[2])  # Ensure string type  
                rank = int(parts[3])
                
                if rank <= max_candidates_per_query:
                    if qid not in candidate_ranking:
                        candidate_ranking[qid] = {}
                    
                    # Store rank position (1-based as per TREC format)
                    candidate_ranking[qid][docid] = rank
    
    print(f"Parsed candidate set: {len(candidate_ranking)} queries, "
          f"{sum(len(docs) for docs in candidate_ranking.values())} total candidates")
    
    return candidate_ranking
        '''
        
        parser_file = os.path.join(self.output_dir, 'parse_candidate_set.py')
        with open(parser_file, 'w') as f:
            f.write(parser_code)
        
        print(f"Generated candidate set parser: {parser_file}")
        return parser_file
    
    def run_full_pipeline(self):
        """Run the complete pipeline"""
        print("=== MS MARCO DEV SPLIT PIPELINE ===\n")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Split queries
        val_queries, test_queries = self.split_queries()
        
        # Step 3: Save query splits
        val_query_file, test_query_file = self.save_query_splits()
        
        # Step 4: Save qrels splits
        val_qrels_file, test_qrels_file = self.save_qrels_splits()
        
        # Step 5: Prepare collection for Pyserini
        collection_jsonl = self.prepare_pyserini_collection()
        
        # Step 6: Generate BM25 commands
        val_bm25_file, test_bm25_file = self.generate_bm25_commands(val_query_file, test_query_file)
        
        # Step 7: Generate candidate set parser
        parser_file = self.generate_candidate_set_parser()
        
        print(f"\n=== PIPELINE SUMMARY ===")
        print(f"Output directory: {self.output_dir}")
        print(f"Validation queries: {len(self.validation_queries)}")
        print(f"Test queries: {len(self.test_queries)}")
        print(f"\nGenerated files:")
        print(f"  - {val_query_file}")
        print(f"  - {test_query_file}")
        print(f"  - {val_qrels_file}")
        print(f"  - {test_qrels_file}")
        print(f"  - {collection_jsonl}")
        print(f"  - {parser_file}")
        
        print(f"\n=== NEXT STEPS ===")
        print(f"1. Run BM25 generation commands from: {self.output_dir}/generate_bm25.sh")
        print(f"2. After BM25 generation completes, run: splitter.generate_final_validation_files()")
        print(f"3. Use the generated files in your matchmaker config")
        
        return {
            'validation_queries': val_query_file,
            'test_queries': test_query_file,
            'validation_qrels': val_qrels_file,
            'test_qrels': test_qrels_file,
            'collection': collection_jsonl,
            'bm25_validation': val_bm25_file,
            'bm25_test': test_bm25_file
        }
    
    def generate_final_validation_files(self):
        """Generate final validation files after BM25 generation is complete"""
        print("Generating final validation files...")
        
        val_bm25_file = os.path.join(self.output_dir, 'bm25_validation.txt')
        test_bm25_file = os.path.join(self.output_dir, 'bm25_test.txt')
        
        if not os.path.exists(val_bm25_file):
            print(f"❌ BM25 validation file not found: {val_bm25_file}")
            print("Run BM25 generation commands first!")
            return None
        
        if not os.path.exists(test_bm25_file):
            print(f"❌ BM25 test file not found: {test_bm25_file}")
            print("Run BM25 generation commands first!")
            return None
        
        validation_tsv, test_tsv, val_pairs, test_pairs = self.generate_validation_files(val_bm25_file, test_bm25_file)
        
        # Generate matchmaker config
        config_content = f'''# Matchmaker config for MS MARCO dev split

validation_cont:
  binarization_point: 1
  candidate_set_from_to: [5, 100]
  candidate_set_path: {val_bm25_file}
  qrels: {os.path.join(self.output_dir, 'qrels_validation.tsv')}
  save_only_best: true
  tsv: {validation_tsv}

test:
  dev_test_split:
    binarization_point: 1
    candidate_set_from_to: [5, 100] 
    candidate_set_path: {test_bm25_file}
    qrels: {os.path.join(self.output_dir, 'qrels_test.tsv')}
    tsv: {test_tsv}
    save_secondary_output: True

# Data split summary:
# Validation queries: {len(self.validation_queries)}
# Test queries: {len(self.test_queries)}
# Validation pairs: {val_pairs}
# Test pairs: {test_pairs}
'''
        
        config_file = os.path.join(self.output_dir, 'matchmaker_config.yaml')
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        print(f"\n=== COMPLETE! ===")
        print(f"Generated files:")
        print(f"  - Validation TSV: {validation_tsv} ({val_pairs} pairs)")
        print(f"  - Test TSV: {test_tsv} ({test_pairs} pairs)")  
        print(f"  - Matchmaker config: {config_file}")
        print(f"\n✅ All files are aligned - no KeyError should occur!")
        
        # Verification
        self.verify_final_alignment(val_bm25_file, validation_tsv, "validation")
        self.verify_final_alignment(test_bm25_file, test_tsv, "test")
        
        return {
            'validation_tsv': validation_tsv,
            'test_tsv': test_tsv,
            'config': config_file
        }
    
    def verify_final_alignment(self, bm25_file, validation_file, set_name):
        """Verify that validation file perfectly aligns with BM25 candidates"""
        print(f"\nVerifying {set_name} alignment...")
        
        # Load BM25 candidates
        bm25_candidates = set()
        with open(bm25_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    qid, docid = str(parts[0]), str(parts[2])
                    bm25_candidates.add((qid, docid))
        
        # Load validation pairs
        validation_pairs = set()
        with open(validation_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    qid, docid = str(parts[0]), str(parts[1])
                    validation_pairs.add((qid, docid))
        
        # Check alignment
        missing_from_validation = bm25_candidates - validation_pairs
        extra_in_validation = validation_pairs - bm25_candidates
        
        print(f"  BM25 candidates: {len(bm25_candidates)}")
        print(f"  Validation pairs: {len(validation_pairs)}")
        print(f"  Missing BM25 in validation: {len(missing_from_validation)}")
        print(f"  Extra in validation: {len(extra_in_validation)}")
        
        if len(missing_from_validation) == 0:
            print(f"  ✅ Perfect alignment - all BM25 candidates in validation")
        else:
            print(f"  ❌ Alignment issue - some BM25 candidates missing")
        
        return len(missing_from_validation) == 0

# Usage example
if __name__ == "__main__":
    # Initialize splitter
    splitter = MSMARCODevSplitter()
    
    # Run initial pipeline (before BM25 generation)
    files = splitter.run_full_pipeline()
    
    print("\n" + "="*60)
    print("MANUAL STEP REQUIRED:")
    print("="*60)
    print("Now run the BM25 generation commands, then call:")
    print("splitter.generate_final_validation_files()")
    print("="*60)