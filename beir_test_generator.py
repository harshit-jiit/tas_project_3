#!/usr/bin/env python3
"""
BEIR Test File Generator - Python 3.12 Compatible
Generates proper test files with BM25 candidates for any BEIR dataset
Avoids deprecated allennlp and ensures perfect alignment to prevent KeyErrors
"""

import os
import json
import zipfile
import requests
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

class BEIRTestGenerator:
    def __init__(self, dataset_name, base_path='/workspace/2404170001/beir_evaluation'):
        """
        Initialize BEIR test generator
        
        Args:
            dataset_name: BEIR dataset name (e.g., 'scifact', 'nfcorpus', 'trec-covid')
            base_path: Base directory for outputs
        """
        self.dataset_name = dataset_name.lower()
        self.base_path = Path(base_path)
        
        # Create directories
        self.dataset_dir = self.base_path / self.dataset_name
        self.output_dir = self.dataset_dir / 'test_files'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parameters
        self.bm25_hits = 1000  # Standard for BEIR evaluation1
        self.max_doc_length = 100_000
        
        # Data containers
        self.corpus = {}
        self.queries = {}
        self.qrels = defaultdict(dict)
        
        # Available BEIR datasets
        self.available_datasets = [
            'msmarco', 'trec-covid', 'nfcorpus', 'nq', 'hotpotqa', 'fiqa',
            'arguana', 'touche-2020', 'cqadupstack', 'quora', 'dbpedia-entity',
            'scidocs', 'fever', 'climate-fever', 'scifact', 'robust04', 'signal1m'
        ]
        
        if self.dataset_name not in self.available_datasets:
            print(f"⚠️  Warning: {self.dataset_name} not in standard BEIR list")
            print(f"Available datasets: {', '.join(self.available_datasets)}")
    
    def download_dataset(self):
        """Download and extract BEIR dataset"""
        print(f"📥 Downloading BEIR {self.dataset_name} dataset...")
        
        # BEIR download URL
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.dataset_name}.zip"
        zip_path = self.dataset_dir / f"{self.dataset_name}.zip"
        extract_path = self.dataset_dir / "raw_data"
        
        # Check if already downloaded
        if extract_path.exists() and any(extract_path.iterdir()):
            print(f"✅ Dataset already exists at {extract_path}")
            return extract_path
        
        # Download
        try:
            print(f"Downloading from: {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(zip_path, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)
            
            print(f"✅ Downloaded: {zip_path}")
            
        except requests.RequestException as e:
            print(f"❌ Download failed: {e}")
            return None
        
        # Extract
        try:
            print("📂 Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.dataset_dir)
            
            # Find extracted folder
            for item in self.dataset_dir.iterdir():
                if item.is_dir() and item.name != 'test_files':
                    extract_path = item
                    break
            
            os.remove(zip_path)  # Clean up
            print(f"✅ Extracted to: {extract_path}")
            return extract_path
            
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return None
    
    def inspect_qrels_format(self, data_path):
        """Inspect qrels file format for debugging"""
        qrels_file = data_path / "qrels" / "test.tsv"
        if not qrels_file.exists():
            print(f"❌ Qrels file not found: {qrels_file}")
            return
        
        print(f"🔍 Inspecting qrels format: {qrels_file}")
        
        with open(qrels_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:10]  # Read first 10 lines
        
        print(f"Total lines in file: {len(lines)} (showing first 10)")
        print("Format analysis:")
        
        for i, line in enumerate(lines, 1):
            line_clean = line.strip()
            if not line_clean:
                continue
                
            parts = line_clean.split('\t')
            print(f"Line {i}: {len(parts)} columns")
            for j, part in enumerate(parts):
                print(f"  Column {j}: '{part}'")
            
            # Try to detect format
            if len(parts) == 3:
                print(f"  -> Detected BEIR format (3 columns)")
            elif len(parts) == 4:
                print(f"  -> Detected TREC format (4 columns)")
            else:
                print(f"  -> Unknown format ({len(parts)} columns)")
            print()
        
        return lines
    
    def load_beir_data(self, data_path):
        """Load BEIR dataset files"""
        print(f"📊 Loading BEIR data from {data_path}...")
        
        # First inspect qrels format for debugging
        self.inspect_qrels_format(data_path)
        
        # Load corpus
        corpus_file = data_path / "corpus.jsonl"
        if corpus_file.exists():
            with open(corpus_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(tqdm(f, desc="Loading corpus"), 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    
                    try:
                        doc = json.loads(line)
                        doc_id = str(doc['_id'])
                        title = doc.get('title', '')
                        text = doc.get('text', '')
                        # Combine title and text
                        full_text = f"{title} {text}".strip()
                        self.corpus[doc_id] = full_text[:self.max_doc_length]
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"   Warning: Skipping invalid JSON line {line_num}: {e}")
                        continue
            
            print(f"✅ Loaded {len(self.corpus)} documents")
        else:
            print(f"❌ Corpus file not found: {corpus_file}")
            return False
        
        # Load queries
        queries_file = data_path / "queries.jsonl"
        if queries_file.exists():
            with open(queries_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(tqdm(f, desc="Loading queries"), 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    
                    try:
                        query = json.loads(line)
                        query_id = str(query['_id'])
                        query_text = query['text']
                        self.queries[query_id] = query_text
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"   Warning: Skipping invalid JSON line {line_num}: {e}")
                        continue
            
            print(f"✅ Loaded {len(self.queries)} queries")
        else:
            print(f"❌ Queries file not found: {queries_file}")
            return False
        
        # Load qrels (test split) - Handle both BEIR (3-col) and TREC (4-col) formats
        qrels_file = data_path / "qrels" / "test.tsv"
        if qrels_file.exists():
            with open(qrels_file, 'r', encoding='utf-8') as f:
                qrels_loaded = 0
                for line_num, line in enumerate(tqdm(f, desc="Loading qrels"), 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    
                    parts = line.split('\t')
                    if len(parts) < 3:
                        continue
                    
                    # Extract components based on number of columns
                    if len(parts) == 3:
                        # BEIR format: query_id \t doc_id \t relevance
                        query_id_str, doc_id_str, relevance_str = parts[0], parts[1], parts[2]
                    elif len(parts) >= 4:
                        # TREC format: query_id \t iteration \t doc_id \t relevance
                        query_id_str, doc_id_str, relevance_str = parts[0], parts[2], parts[3]
                    else:
                        continue
                    
                    # Clean strings
                    query_id_str = query_id_str.strip()
                    doc_id_str = doc_id_str.strip()
                    relevance_str = relevance_str.strip()
                    
                    # Skip header rows
                    if (query_id_str.lower() in ['query_id', 'query-id', 'qid', 'query'] or
                        doc_id_str.lower() in ['doc_id', 'doc-id', 'docid', 'document_id', 'document'] or
                        relevance_str.lower() in ['relevance', 'score', 'label', 'rel', 'rating']):
                        print(f"   Skipping header row: {line}")
                        continue
                    
                    try:
                        query_id = str(query_id_str)
                        doc_id = str(doc_id_str)
                        relevance = float(relevance_str)
                        if relevance > 0:  # Only store relevant documents
                            self.qrels[query_id][doc_id] = relevance
                            qrels_loaded += 1
                    except ValueError as e:
                        print(f"   Warning: Skipping invalid line {line_num}: {line}")
                        print(f"   Error: Cannot convert '{relevance_str}' to float")
                        continue
                
                print(f"✅ Loaded {qrels_loaded} relevant qrels pairs for {len(self.qrels)} queries")
            
            print(f"✅ Loaded qrels for {len(self.qrels)} queries")
        else:
            print(f"❌ Qrels file not found: {qrels_file}")
            return False
        
        return True
    
    def prepare_pyserini_files(self):
        """Prepare files for Pyserini BM25 generation"""
        print("🔧 Preparing files for Pyserini...")
        
        # 1. Create collection.jsonl for Pyserini
        collection_jsonl = self.output_dir / 'collection.jsonl'
        with open(collection_jsonl, 'w', encoding='utf-8') as f:
            for doc_id, text in tqdm(self.corpus.items(), desc="Converting corpus"):
                record = {"id": doc_id, "contents": text}
                f.write(json.dumps(record) + '\n')
        
        print(f"✅ Created collection: {collection_jsonl}")
        
        # 2. Create queries.tsv for Pyserini
        queries_tsv = self.output_dir / 'queries_test.tsv'
        with open(queries_tsv, 'w', encoding='utf-8') as f:
            for query_id, query_text in self.queries.items():
                f.write(f"{query_id}\t{query_text}\n")
        
        print(f"✅ Created queries: {queries_tsv}")
        
        # 3. Create qrels.tsv (BEIR format: 3 columns)
        qrels_tsv = self.output_dir / 'qrels_test.tsv'
        with open(qrels_tsv, 'w', encoding='utf-8') as f:
            for query_id in sorted(self.qrels.keys()):
                for doc_id, relevance in self.qrels[query_id].items():
                    f.write(f"{query_id}\t0\t{doc_id}\t{int(relevance)}\n")
        
        print(f"✅ Created qrels: {qrels_tsv}")
        
        return collection_jsonl, queries_tsv, qrels_tsv
    
    def generate_bm25_commands(self, queries_tsv):
        """Generate Pyserini BM25 commands"""
        print("📝 Generating BM25 commands...")
        
        index_path = self.output_dir / 'pyserini_index'
        bm25_results = self.output_dir / 'bm25_test.txt'
        
        # Create BM25 generation script
        commands = f'''#!/bin/bash
# ===== BEIR {self.dataset_name.upper()} BM25 GENERATION =====

echo "Building Lucene index..."
python -m pyserini.index.lucene \\
  --collection JsonCollection \\
  --input {self.output_dir} \\
  --index {index_path} \\
  --generator DefaultLuceneDocumentGenerator \\
  --threads 8

echo "Generating BM25 results..."
python -m pyserini.search.lucene \\
  --index {index_path} \\
  --topics {queries_tsv} \\
  --output {bm25_results} \\
  --bm25 \\
  --hits {self.bm25_hits} \\
  --output-format trec

echo "Verifying results..."
echo "Expected lines: $(($(wc -l < {queries_tsv}) * {self.bm25_hits}))"
echo "Actual lines: $(wc -l < {bm25_results})"

echo "✅ BM25 generation complete!"
echo "Results saved to: {bm25_results}"
'''
        
        # Save commands to executable script
        script_file = self.output_dir / 'generate_bm25.sh'
        with open(script_file, 'w') as f:
            f.write(commands)
        
        # Make executable
        os.chmod(script_file, 0o755)
        
        print(f"✅ Created BM25 script: {script_file}")
        print(f"Expected BM25 results: {bm25_results}")
        
        return str(bm25_results), str(script_file)
    
    def create_evaluation_tsv(self, bm25_file):
        """Create evaluation TSV file aligned with BM25 candidates"""
        print("📄 Creating evaluation TSV file...")
        
        if not os.path.exists(bm25_file):
            print(f"❌ BM25 results not found: {bm25_file}")
            print("Please run BM25 generation first!")
            return None
        
        # Load BM25 candidates
        bm25_candidates = set()
        bm25_ranking = defaultdict(dict)
        
        with open(bm25_file, 'r') as f:
            for line in tqdm(f, desc="Loading BM25 results"):
                parts = line.strip().split()
                if len(parts) >= 4:
                    query_id = str(parts[0])
                    doc_id = str(parts[2])
                    rank = int(parts[3])
                    score = float(parts[4])
                    
                    bm25_candidates.add((query_id, doc_id))
                    bm25_ranking[query_id][doc_id] = rank
        
        print(f"✅ Loaded {len(bm25_candidates)} BM25 candidate pairs")
        
        # Debug: Check ID format matching
        print("🔍 Checking ID format compatibility...")
        sample_bm25_queries = set(list(bm25_candidates)[:5]) if bm25_candidates else set()
        sample_bm25_query_ids = {qid for qid, _ in sample_bm25_queries}
        sample_corpus_ids = set(list(self.corpus.keys())[:5])
        sample_query_ids = set(list(self.queries.keys())[:5])
        
        print(f"   Sample BM25 query IDs: {sample_bm25_query_ids}")
        print(f"   Sample loaded query IDs: {sample_query_ids}")
        print(f"   Sample BM25 doc IDs: {set(doc_id for _, doc_id in sample_bm25_queries)}")
        print(f"   Sample loaded doc IDs: {sample_corpus_ids}")
        
        # Check for matches
        bm25_query_ids = {qid for qid, _ in bm25_candidates}
        bm25_doc_ids = {doc_id for _, doc_id in bm25_candidates}
        
        query_matches = len(bm25_query_ids & set(self.queries.keys()))
        doc_matches = len(bm25_doc_ids & set(self.corpus.keys()))
        
        print(f"   Query ID matches: {query_matches}/{len(bm25_query_ids)}")
        print(f"   Doc ID matches: {doc_matches}/{len(bm25_doc_ids)}")
        
        if query_matches == 0:
            print("   ❌ No query ID matches found!")
        if doc_matches == 0:
            print("   ❌ No document ID matches found!")
        
        # Create evaluation TSV
        test_tsv = self.output_dir / 'test_evaluation.tsv'
        written_pairs = set()
        
        with open(test_tsv, 'w', encoding='utf-8') as f:
            # Add ALL BM25 candidates (ensures perfect alignment)
            for query_id, doc_id in tqdm(bm25_candidates, desc="Writing BM25 candidates"):
                if query_id in self.queries and doc_id in self.corpus:
                    if (query_id, doc_id) not in written_pairs:
                        written_pairs.add((query_id, doc_id))
                        row = [
                            query_id,
                            doc_id, 
                            self.queries[query_id],
                            self.corpus[doc_id]
                        ]
                        f.write('\t'.join(row) + '\n')
            
            # Add relevant documents not retrieved by BM25
            qrels_not_in_bm25 = 0
            for query_id in self.qrels:
                for doc_id in self.qrels[query_id]:
                    if (query_id, doc_id) not in written_pairs:
                        if query_id in self.queries and doc_id in self.corpus:
                            written_pairs.add((query_id, doc_id))
                            qrels_not_in_bm25 += 1
                            row = [
                                query_id,
                                doc_id,
                                self.queries[query_id], 
                                self.corpus[doc_id]
                            ]
                            f.write('\t'.join(row) + '\n')
        
        print(f"✅ Created evaluation TSV: {test_tsv}")
        print(f"   Total pairs: {len(written_pairs)}")
        print(f"   BM25 candidates: {len(bm25_candidates)}")
        print(f"   Qrels not in BM25: {qrels_not_in_bm25}")
        
        return str(test_tsv), bm25_ranking
    
    def create_matchmaker_config(self, test_tsv, qrels_tsv, bm25_file):
        """Create matchmaker configuration file"""
        print("⚙️  Creating matchmaker config...")
        
        config_content = f'''# Matchmaker config for BEIR {self.dataset_name}
# Generated by BEIRTestGenerator

# Dataset info
dataset_name: {self.dataset_name}
corpus_size: {len(self.corpus)}
query_count: {len(self.queries)}
qrels_count: {len(self.qrels)}

# Test configuration
test:
  {self.dataset_name}_test:
    tsv: "{test_tsv}"
    qrels: "{qrels_tsv}"
    binarization_point: 1
    candidate_set_path: "{bm25_file}"
    candidate_set_from_to: [5, {self.bm25_hits}]
    save_secondary_output: True
    top_n: {self.bm25_hits}

# Important settings for BEIR evaluation
max_doc_length: 2000
max_query_length: 30
batch_size_eval: 64
use_fp16: True

# Expected BEIR metrics for {self.dataset_name}:
# - MRR@10: 0.3-0.8 (depending on dataset)
# - Recall@10: 0.3-0.7
# - NDCG@10: 0.4-0.8
# Perfect 1.0 scores indicate evaluation issues!
'''
        
        config_file = self.output_dir / 'matchmaker_config.yaml'
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        print(f"✅ Created config: {config_file}")
        
        # Add format note
        print(f"\n📝 Format Note:")
        print(f"   BEIR qrels format: query_id \\t doc_id \\t relevance (3 columns)")
        print(f"   TREC qrels format: query_id \\t iteration \\t doc_id \\t relevance (4 columns)")
        
        return str(config_file)
    
    def create_candidate_parser(self):
        """Create candidate set parser for matchmaker"""
        
        parser_code = '''
def parse_candidate_set(candidate_file_path, max_candidates_per_query):
    """
    Parse TREC-format BM25 results for matchmaker validate_model
    Returns: dict mapping query_id -> {doc_id: rank_position}
    
    Note: BM25 results are in TREC format (5 columns), not BEIR format (3 columns)
    TREC format: query_id Q0 doc_id rank score run_name
    """
    
    candidate_ranking = {}
    
    with open(candidate_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid = str(parts[0])
                docid = str(parts[2])
                rank = int(parts[3])
                
                if rank <= max_candidates_per_query:
                    if qid not in candidate_ranking:
                        candidate_ranking[qid] = {}
                    
                    candidate_ranking[qid][docid] = rank
    
    print(f"Parsed {len(candidate_ranking)} queries with "
          f"{sum(len(docs) for docs in candidate_ranking.values())} candidates")
    
    return candidate_ranking


def calculate_beir_metrics(rankings, qrels, k_values=[1, 3, 5, 10, 20, 100]):
    """
    Calculate BEIR-style metrics
    
    Args:
        rankings: dict {query_id: [(doc_id, score), ...]} sorted by score desc
        qrels: dict {query_id: {doc_id: relevance}}
        k_values: list of k values for metrics
    
    Returns:
        dict with metrics
    """
    import numpy as np
    
    def dcg_at_k(relevances, k):
        relevances = relevances[:k]
        if len(relevances) == 0:
            return 0.0
        dcg = relevances[0]
        for i in range(1, len(relevances)):
            dcg += relevances[i] / np.log2(i + 1)
        return dcg
    
    def ndcg_at_k(relevances, k):
        dcg = dcg_at_k(relevances, k)
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = dcg_at_k(ideal_relevances, k)
        return dcg / idcg if idcg > 0 else 0.0
    
    metrics = {}
    
    for k in k_values:
        recall_scores = []
        mrr_scores = []
        ndcg_scores = []
        
        for query_id in rankings:
            if query_id not in qrels:
                continue
            
            ranked_docs = [doc_id for doc_id, score in rankings[query_id][:k]]
            relevant_docs = set(doc_id for doc_id, rel in qrels[query_id].items() if rel > 0)
            
            if len(relevant_docs) == 0:
                continue
            
            # Recall@k
            retrieved_relevant = set(ranked_docs) & relevant_docs
            recall = len(retrieved_relevant) / len(relevant_docs)
            recall_scores.append(recall)
            
            # MRR@k
            mrr = 0.0
            for i, doc_id in enumerate(ranked_docs):
                if doc_id in relevant_docs:
                    mrr = 1.0 / (i + 1)
                    break
            mrr_scores.append(mrr)
            
            # NDCG@k
            relevances = [qrels[query_id].get(doc_id, 0) for doc_id in ranked_docs]
            ndcg = ndcg_at_k(relevances, k)
            ndcg_scores.append(ndcg)
        
        # Average metrics
        metrics[f'Recall@{k}'] = np.mean(recall_scores) if recall_scores else 0.0
        metrics[f'MRR@{k}'] = np.mean(mrr_scores) if mrr_scores else 0.0
        metrics[f'NDCG@{k}'] = np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    return metrics


# Usage example:
# candidate_ranking = parse_candidate_set("bm25_test.txt", 1000)
# metrics = calculate_beir_metrics(model_rankings, qrels)
        '''
        
        parser_file = self.output_dir / 'beir_utils.py'
        with open(parser_file, 'w') as f:
            f.write(parser_code)
        
        print(f"✅ Created utilities: {parser_file}")
        return str(parser_file)
    
    def verify_alignment(self, test_tsv, bm25_file):
        """Verify perfect alignment between test TSV and BM25 candidates"""
        print("🔍 Verifying alignment...")
        
        # Load BM25 pairs
        bm25_pairs = set()
        with open(bm25_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    qid, docid = str(parts[0]), str(parts[2])
                    bm25_pairs.add((qid, docid))
        
        # Load TSV pairs
        tsv_pairs = set()
        with open(test_tsv, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    qid, docid = str(parts[0]), str(parts[1])
                    tsv_pairs.add((qid, docid))
        
        # Check alignment
        missing_in_tsv = bm25_pairs - tsv_pairs
        extra_in_tsv = tsv_pairs - bm25_pairs
        
        print(f"  BM25 pairs: {len(bm25_pairs)}")
        print(f"  TSV pairs: {len(tsv_pairs)}")
        print(f"  Missing in TSV: {len(missing_in_tsv)}")
        print(f"  Extra in TSV: {len(extra_in_tsv)}")
        
        if len(missing_in_tsv) == 0:
            print("  ✅ Perfect alignment!")
            return True
        else:
            print("  ❌ Alignment issues detected")
            if len(missing_in_tsv) < 10:
                print(f"  Missing pairs: {list(missing_in_tsv)}")
            return False
    
    def run_pipeline(self):
        """Run the complete BEIR test generation pipeline"""
        print(f"🚀 BEIR Test Generator for {self.dataset_name}")
        print("="*60)
        
        # Step 1: Download dataset
        data_path = self.download_dataset()
        if not data_path:
            return None
        
        # Step 2: Load BEIR data
        if not self.load_beir_data(data_path):
            return None
        
        # Step 3: Prepare Pyserini files
        collection_jsonl, queries_tsv, qrels_tsv = self.prepare_pyserini_files()
        
        # Step 4: Generate BM25 commands
        bm25_file, bm25_script = self.generate_bm25_commands(queries_tsv)
        
        # Step 5: Create utilities
        utils_file = self.create_candidate_parser()
        
        print(f"\n🎯 PIPELINE PHASE 1 COMPLETE")
        print(f"="*60)
        print(f"Dataset: {self.dataset_name}")
        print(f"Corpus: {len(self.corpus)} documents")
        print(f"Queries: {len(self.queries)} queries")
        print(f"Qrels: {len(self.qrels)} queries with relevance judgments")
        print(f"Output directory: {self.output_dir}")
        
        print(f"\n📋 NEXT STEPS:")
        print(f"1. Install Pyserini: pip install pyserini")
        print(f"2. Run BM25 generation: bash {bm25_script}")
        print(f"3. Generate final files: generator.create_final_files()")
        
        return {
            'dataset_name': self.dataset_name,
            'output_dir': str(self.output_dir),
            'bm25_script': bm25_script,
            'bm25_file': bm25_file,
            'queries_tsv': str(queries_tsv),
            'qrels_tsv': str(qrels_tsv),
            'utils_file': utils_file
        }
    
    def create_final_files(self):
        """Create final evaluation files after BM25 generation"""
        print("🏁 Creating final evaluation files...")
        
        bm25_file = self.output_dir / 'bm25_test.txt'
        qrels_tsv = self.output_dir / 'qrels_test.tsv'
        
        if not bm25_file.exists():
            print(f"❌ BM25 results not found: {bm25_file}")
            print("Please run BM25 generation first!")
            return None
        
        # Load BEIR data if not already loaded
        if len(self.corpus) == 0 or len(self.queries) == 0:
            print("📊 Loading BEIR data...")
            data_path = self.dataset_dir / "raw_data"
            if not data_path.exists():
                # Try to find the extracted data directory
                for item in self.dataset_dir.iterdir():
                    if item.is_dir() and item.name != 'test_files':
                        data_path = item
                        break
            
            if not data_path.exists():
                print(f"❌ BEIR data not found. Please run generator.run_pipeline() first!")
                return None
            
            if not self.load_beir_data(data_path):
                print(f"❌ Failed to load BEIR data from {data_path}")
                return None
            
            print(f"✅ Loaded {len(self.corpus)} docs, {len(self.queries)} queries, {len(self.qrels)} qrels")
        
        # Create evaluation TSV
        test_tsv, bm25_ranking = self.create_evaluation_tsv(str(bm25_file))
        if not test_tsv:
            return None
        
        # Create matchmaker config
        config_file = self.create_matchmaker_config(test_tsv, str(qrels_tsv), str(bm25_file))
        
        # Verify alignment
        alignment_ok = self.verify_alignment(test_tsv, str(bm25_file))
        
        print(f"\n🎉 BEIR TEST GENERATION COMPLETE!")
        print(f"="*60)
        print(f"Dataset: {self.dataset_name}")
        print(f"✅ Test evaluation TSV: {test_tsv}")
        print(f"✅ Qrels: {qrels_tsv}")
        print(f"✅ BM25 candidates: {bm25_file}")
        print(f"✅ Matchmaker config: {config_file}")
        print(f"✅ Alignment verified: {alignment_ok}")
        
        print(f"\n📊 Expected Results:")
        print(f"- Should get realistic scores (0.3-0.8 range)")
        print(f"- Perfect 1.0 scores indicate evaluation problems")
        print(f"- Use this test file in your matchmaker config")
        
        print(f"\n🔧 Usage in matchmaker:")
        print(f"test:")
        print(f"  {self.dataset_name}_test:")
        print(f"    tsv: \"{test_tsv}\"")
        print(f"    qrels: \"{qrels_tsv}\"")
        print(f"    binarization_point: 1")
        
        return {
            'test_tsv': test_tsv,
            'qrels_tsv': str(qrels_tsv),
            'bm25_file': str(bm25_file),
            'config_file': config_file,
            'alignment_ok': alignment_ok
        }


# Usage examples for different BEIR datasets
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    else:
        dataset_name = 'scifact'  # Default
    
    print(f"🎯 Generating test files for BEIR {dataset_name}")
    
    # Initialize generator
    generator = BEIRTestGenerator(dataset_name)
    
    # Run pipeline
    result = generator.run_pipeline()
    # generator.create_final_files()
    if result:
        print("\n" + "="*60)
        print("🔥 READY FOR BM25 GENERATION")
        print("="*60)
        print(f"Run: bash {result['bm25_script']}")
        print("Then call: generator.create_final_files()")
    else:
        print("❌ Pipeline failed!")


# Quick commands for common datasets:
# python beir_test_generator.py scifact
# python beir_test_generator.py nfcorpus  
# python beir_test_generator.py trec-covid
# python beir_test_generator.py hotpotqa