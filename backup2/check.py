# import pandas as pd
# from collections import defaultdict

# def debug_query_document_mismatch():
#     """
#     Debug script to identify mismatches between BM25 results, qrels, and validation files
#     """
    
#     # File paths
#     bm25_file = '/workspace/2404170001/validation_2/pyserini_data_fixed/bm25_run.txt'
#     qrels_file = '/workspace/2404170001/inputs/qrels.dev.tsv'
#     queries_file = '/workspace/2404170001/inputs/queries.dev.small.tsv'
#     validation_file = '/workspace/2404170001/validation_2/pyserini_data_fixed/validation_bm25_aligned.tsv'
    
#     print("=== DEBUGGING QUERY-DOCUMENT MISMATCH ===\n")
    
#     # 1. Load all data structures
#     print("1. Loading data files...")
    
#     # Load queries
#     queries = {}
#     with open(queries_file, 'r') as f:
#         for line in f:
#             parts = line.strip().split('\t')
#             if len(parts) == 2:
#                 qid, qtext = parts
#                 queries[qid] = qtext
#     print(f"   Loaded {len(queries)} queries")
    
#     # Load qrels
#     qrels = defaultdict(dict)
#     with open(qrels_file, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) >= 4:
#                 qid, _, docid, rel = parts[0], parts[1], parts[2], float(parts[3])
#                 if rel > 0:
#                     qrels[qid][docid] = rel
#     print(f"   Loaded qrels for {len(qrels)} queries")
    
#     # Load BM25 results
#     bm25_results = defaultdict(list)
#     with open(bm25_file, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) >= 6:
#                 qid, _, docid, rank, score = parts[0], parts[1], parts[2], int(parts[3]), float(parts[4])
#                 bm25_results[qid].append((docid, rank, score))
#     print(f"   Loaded BM25 results for {len(bm25_results)} queries")
    
#     # Load validation file
#     validation_pairs = []
#     with open(validation_file, 'r') as f:
#         for line in f:
#             parts = line.strip().split('\t')
#             if len(parts) >= 2:
#                 qid, docid = parts[0], parts[1]
#                 validation_pairs.append((qid, docid))
#     print(f"   Loaded {len(validation_pairs)} validation pairs\n")
    
#     # 2. Build comprehensive document-to-query mappings
#     print("2. Building document-to-query mappings...")
    
#     # Map documents to queries in each source
#     bm25_doc_to_queries = defaultdict(list)
#     qrels_doc_to_queries = defaultdict(list)
#     validation_doc_to_queries = defaultdict(list)
    
#     # BM25 mapping
#     for qid, docs in bm25_results.items():
#         for docid, rank, score in docs:
#             bm25_doc_to_queries[docid].append((qid, rank, score))
    
#     # Qrels mapping
#     for qid, docs in qrels.items():
#         for docid, rel_score in docs.items():
#             qrels_doc_to_queries[docid].append((qid, rel_score))
    
#     # Validation mapping
#     for qid, docid in validation_pairs:
#         validation_doc_to_queries[docid].append(qid)
    
#     print(f"   Documents in BM25: {len(bm25_doc_to_queries)}")
#     print(f"   Documents in qrels: {len(qrels_doc_to_queries)}")
#     print(f"   Documents in validation: {len(validation_doc_to_queries)}")
    
#     # 3. Find all documents with query mismatches
#     print("\n3. Analyzing all documents for query mismatches...")
    
#     all_docs = set(bm25_doc_to_queries.keys()) | set(qrels_doc_to_queries.keys()) | set(validation_doc_to_queries.keys())
    
#     mismatch_docs = []
#     multiple_query_docs = []
#     missing_candidate_docs = []
    
#     for docid in all_docs:
#         bm25_queries = set(q[0] for q in bm25_doc_to_queries.get(docid, []))
#         qrels_queries = set(q[0] for q in qrels_doc_to_queries.get(docid, []))
#         validation_queries = set(validation_doc_to_queries.get(docid, []))
        
#         # Check for mismatches between qrels and BM25
#         if qrels_queries and bm25_queries:
#             if not qrels_queries.intersection(bm25_queries):
#                 mismatch_docs.append({
#                     'docid': docid,
#                     'qrels_queries': list(qrels_queries),
#                     'bm25_queries': list(bm25_queries),
#                     'validation_queries': list(validation_queries)
#                 })
        
#         # Check for documents relevant to multiple queries
#         if len(qrels_queries) > 1:
#             multiple_query_docs.append({
#                 'docid': docid,
#                 'qrels_queries': list(qrels_queries),
#                 'count': len(qrels_queries)
#             })
        
#         # Check for documents in validation but not in BM25 candidates
#         if validation_queries and not bm25_queries:
#             missing_candidate_docs.append({
#                 'docid': docid,
#                 'validation_queries': list(validation_queries),
#                 'in_qrels': bool(qrels_queries)
#             })
    
#     # 4. Report findings
#     print(f"\n4. ANALYSIS RESULTS:")
#     print(f"   Total documents analyzed: {len(all_docs)}")
#     print(f"   Documents with query mismatches: {len(mismatch_docs)}")
#     print(f"   Documents relevant to multiple queries: {len(multiple_query_docs)}")
#     print(f"   Documents in validation but not BM25: {len(missing_candidate_docs)}")
    
#     # 5. Detailed reporting
#     if mismatch_docs:
#         print(f"\n5. DOCUMENTS WITH QUERY MISMATCHES:")
#         print(f"   (Document appears in BM25 for different query than in qrels)")
#         for i, doc_info in enumerate(mismatch_docs[:10]):  # Show first 10
#             docid = doc_info['docid']
#             print(f"\n   Document {docid}:")
#             print(f"     Qrels queries: {doc_info['qrels_queries']}")
#             print(f"     BM25 queries: {doc_info['bm25_queries']}")
#             print(f"     Validation queries: {doc_info['validation_queries']}")
            
#             # Show query texts for context
#             for qid in (doc_info['qrels_queries'] + doc_info['bm25_queries'])[:4]:
#                 if qid in queries:
#                     print(f"       Query {qid}: {queries[qid][:100]}...")
        
#         if len(mismatch_docs) > 10:
#             print(f"   ... and {len(mismatch_docs) - 10} more documents with mismatches")
    
#     if multiple_query_docs:
#         print(f"\n6. DOCUMENTS RELEVANT TO MULTIPLE QUERIES:")
#         multiple_query_docs.sort(key=lambda x: x['count'], reverse=True)
#         for i, doc_info in enumerate(multiple_query_docs[:5]):  # Show top 5
#             docid = doc_info['docid']
#             print(f"\n   Document {docid} (relevant to {doc_info['count']} queries):")
#             for qid in doc_info['qrels_queries'][:3]:  # Show first 3 queries
#                 if qid in queries:
#                     print(f"     Query {qid}: {queries[qid][:80]}...")
        
#         if len(multiple_query_docs) > 5:
#             print(f"   ... and {len(multiple_query_docs) - 5} more documents")
    
#     if missing_candidate_docs:
#         print(f"\n7. DOCUMENTS IN VALIDATION BUT NOT BM25:")
#         for i, doc_info in enumerate(missing_candidate_docs[:10]):  # Show first 10
#             docid = doc_info['docid']
#             print(f"\n   Document {docid}:")
#             print(f"     Validation queries: {doc_info['validation_queries']}")
#             print(f"     In qrels: {doc_info['in_qrels']}")
            
#             for qid in doc_info['validation_queries'][:2]:
#                 if qid in queries:
#                     print(f"       Query {qid}: {queries[qid][:80]}...")
        
#         if len(missing_candidate_docs) > 10:
#             print(f"   ... and {len(missing_candidate_docs) - 10} more documents")
    
#     # 6. Statistics and recommendations
#     print(f"\n8. STATISTICS:")
    
#     # Count total mismatched pairs
#     total_mismatch_pairs = sum(len(doc['qrels_queries']) * len(doc['bm25_queries']) 
#                               for doc in mismatch_docs)
    
#     # Count documents that would cause KeyError
#     problematic_docs_for_eval = set()
#     for doc_info in mismatch_docs:
#         for val_qid in doc_info['validation_queries']:
#             for bm25_qid in doc_info['bm25_queries']:
#                 if val_qid != bm25_qid:
#                     problematic_docs_for_eval.add(doc_info['docid'])
    
#     print(f"   Documents causing evaluation KeyError: {len(problematic_docs_for_eval)}")
#     print(f"   Total query-document mismatches: {total_mismatch_pairs}")
    
#     # Calculate percentage of validation pairs affected
#     affected_validation_pairs = 0
#     for qid, docid in validation_pairs:
#         if docid in [d['docid'] for d in mismatch_docs]:
#             affected_validation_pairs += 1
    
#     print(f"   Affected validation pairs: {affected_validation_pairs}/{len(validation_pairs)} ({affected_validation_pairs/len(validation_pairs)*100:.1f}%)")
    
#     print(f"\n9. RECOMMENDATIONS:")
    
#     if len(mismatch_docs) > len(all_docs) * 0.1:  # More than 10% of documents
#         print(f"   🚨 CRITICAL: {len(mismatch_docs)} documents have query mismatches!")
#         print(f"      This suggests a systematic issue with query ID mapping.")
#         print(f"      Recommendation: Regenerate BM25 results with correct query file.")
#     elif len(mismatch_docs) > 0:
#         print(f"   ⚠️  WARNING: {len(mismatch_docs)} documents have query mismatches.")
#         print(f"      This could be due to:")
#         print(f"      1. Cross-query relevance (document relevant to multiple queries)")
#         print(f"      2. BM25 retrieval noise (irrelevant documents retrieved)")
#         print(f"      3. Query ID mapping issues")
    
#     if len(missing_candidate_docs) > 0:
#         print(f"   ⚠️  {len(missing_candidate_docs)} relevant documents missing from BM25 candidates.")
#         print(f"      This will limit the maximum possible recall.")
    
#     # 7. Generate fix suggestions
#     print(f"\n10. IMMEDIATE FIXES:")
    
#     print(f"   A. For evaluation script - add defensive checks:")
#     print(f"      - Skip documents not in candidate set")
#     print(f"      - Log warnings for missing documents")
    
#     print(f"   B. For data consistency - consider:")
#     print(f"      - Using only BM25-retrieved documents for validation")
#     print(f"      - Adding missing relevant documents to candidate set")
#     print(f"      - Regenerating BM25 with consistent query IDs")
    
#     return {
#         'mismatch_docs': mismatch_docs,
#         'multiple_query_docs': multiple_query_docs,
#         'missing_candidate_docs': missing_candidate_docs,
#         'total_affected_pairs': affected_validation_pairs
#     }

# if __name__ == "__main__":
#     results = debug_query_document_mismatch()


import pandas as pd
from collections import defaultdict

def debug_query_document_mismatch():
    """
    Debug script to identify mismatches between BM25 results, qrels, and validation files
    """
    
    # File paths
    bm25_file = '/workspace/2404170001/validation_2/pyserini_data_fixed/bm25_run.txt'
    qrels_file = '/workspace/2404170001/inputs/qrels.dev.tsv'
    queries_file = '/workspace/2404170001/inputs/queries.dev.small.tsv'
    validation_file = '/workspace/2404170001/validation_2/pyserini_data_fixed/validation_fixed.tsv'
    
    print("=== DEBUGGING QUERY-DOCUMENT MISMATCH ===\n")
    
    # 1. Load all data structures
    print("1. Loading data files...")
    
    # Load queries - FORCE STRING TYPE
    queries = {}
    with open(queries_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                qid, qtext = str(parts[0]), parts[1]  # Force string
                queries[qid] = qtext
    print(f"   Loaded {len(queries)} queries")
    
    # Load qrels - FORCE STRING TYPE
    qrels = defaultdict(dict)
    with open(qrels_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, docid, rel = str(parts[0]), parts[1], str(parts[2]), float(parts[3])  # Force string
                if rel > 0:
                    qrels[qid][docid] = rel
    print(f"   Loaded qrels for {len(qrels)} queries")
    
    # Load BM25 results - FORCE STRING TYPE
    bm25_results = defaultdict(list)
    bm25_query_set = set()
    with open(bm25_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                qid, _, docid, rank, score = str(parts[0]), parts[1], str(parts[2]), int(parts[3]), float(parts[4])  # Force string
                bm25_results[qid].append((docid, rank, score))
                bm25_query_set.add(qid)
    print(f"   Loaded BM25 results for {len(bm25_results)} queries")
    print(f"   BM25 query set size: {len(bm25_query_set)}")
    
    # Load validation file - FORCE STRING TYPE
    validation_pairs = []
    validation_query_set = set()
    with open(validation_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                qid, docid = str(parts[0]), str(parts[1])  # Force string
                validation_pairs.append((qid, docid))
                validation_query_set.add(qid)
    print(f"   Loaded {len(validation_pairs)} validation pairs")
    print(f"   Validation query set size: {len(validation_query_set)}")
    
    # CRITICAL CHECK: Are the query sets the same?
    print(f"\n1.5. CRITICAL QUERY SET COMPARISON:")
    print(f"   BM25 queries: {len(bm25_query_set)}")
    print(f"   Validation queries: {len(validation_query_set)}")
    
    query_overlap = bm25_query_set & validation_query_set
    only_bm25 = bm25_query_set - validation_query_set
    only_validation = validation_query_set - bm25_query_set
    
    print(f"   Query overlap: {len(query_overlap)}")
    print(f"   Only in BM25: {len(only_bm25)}")
    print(f"   Only in validation: {len(only_validation)}")
    
    if len(only_bm25) > 0:
        print(f"   Sample BM25-only queries: {list(only_bm25)[:5]}")
    if len(only_validation) > 0:
        print(f"   Sample validation-only queries: {list(only_validation)[:5]}")
    
    if len(query_overlap) != len(bm25_query_set) or len(query_overlap) != len(validation_query_set):
        print(f"   ❌ QUERY SETS DON'T MATCH - This is the root problem!")
        print(f"   ❌ You need to regenerate validation file using EXACT BM25 queries")
        return None
    else:
        print(f"   ✅ Query sets match perfectly!")
    
    # 2. Build comprehensive document-to-query mappings
    print("\n2. Building document-to-query mappings...")
    
    # Map documents to queries in each source
    bm25_doc_to_queries = defaultdict(list)
    qrels_doc_to_queries = defaultdict(list)
    validation_doc_to_queries = defaultdict(list)
    
    # BM25 mapping
    for qid, docs in bm25_results.items():
        for docid, rank, score in docs:
            bm25_doc_to_queries[docid].append((qid, rank, score))
    
    # Qrels mapping
    for qid, docs in qrels.items():
        for docid, rel_score in docs.items():
            qrels_doc_to_queries[docid].append((qid, rel_score))
    
    # Validation mapping
    for qid, docid in validation_pairs:
        validation_doc_to_queries[docid].append(qid)
    
    print(f"   Documents in BM25: {len(bm25_doc_to_queries)}")
    print(f"   Documents in qrels: {len(qrels_doc_to_queries)}")
    print(f"   Documents in validation: {len(validation_doc_to_queries)}")
    
    # 3. Find problematic documents
    print("\n3. Analyzing documents for issues...")
    
    all_docs = set(bm25_doc_to_queries.keys()) | set(qrels_doc_to_queries.keys()) | set(validation_doc_to_queries.keys())
    
    # Documents that appear for different queries in different sources
    cross_query_docs = []
    # Documents relevant to multiple queries (normal but worth noting)
    multi_relevant_docs = []
    # Documents in validation but not retrieved by BM25
    qrels_only_docs = []
    
    for docid in all_docs:
        bm25_queries = set(q[0] for q in bm25_doc_to_queries.get(docid, []))
        qrels_queries = set(q[0] for q in qrels_doc_to_queries.get(docid, []))
        validation_queries = set(validation_doc_to_queries.get(docid, []))
        
        # Case 1: Document appears in BM25 and qrels but for DIFFERENT queries
        if bm25_queries and qrels_queries:
            if not bm25_queries.intersection(qrels_queries):
                cross_query_docs.append({
                    'docid': docid,
                    'bm25_queries': list(bm25_queries),
                    'qrels_queries': list(qrels_queries),
                    'validation_queries': list(validation_queries)
                })
        
        # Case 2: Document relevant to multiple queries (this is normal)
        if len(qrels_queries) > 1:
            multi_relevant_docs.append({
                'docid': docid,
                'queries': list(qrels_queries),
                'count': len(qrels_queries)
            })
        
        # Case 3: Document in qrels but not retrieved by BM25 (recall limitation)
        if qrels_queries and not bm25_queries:
            qrels_only_docs.append({
                'docid': docid,
                'qrels_queries': list(qrels_queries)
            })
    
    # 4. Report findings
    print(f"\n4. ANALYSIS RESULTS:")
    print(f"   Total documents: {len(all_docs)}")
    print(f"   Cross-query documents: {len(cross_query_docs)} ❌")
    print(f"   Multi-relevant documents: {len(multi_relevant_docs)} ℹ️")
    print(f"   Qrels-only documents: {len(qrels_only_docs)} ⚠️")
    
    # 5. Show examples of cross-query documents (the main problem)
    if cross_query_docs:
        print(f"\n5. CROSS-QUERY DOCUMENTS (Main Issue):")
        for i, doc_info in enumerate(cross_query_docs[:5]):
            docid = doc_info['docid']
            print(f"\n   Document {docid}:")
            print(f"     BM25 retrieved for queries: {doc_info['bm25_queries']}")
            print(f"     Relevant according to qrels for: {doc_info['qrels_queries']}")
            
            # Show the actual query texts
            print(f"     BM25 query texts:")
            for qid in doc_info['bm25_queries'][:2]:
                if qid in queries:
                    print(f"       {qid}: {queries[qid]}")
            
            print(f"     Qrels query texts:")
            for qid in doc_info['qrels_queries'][:2]:
                if qid in queries:
                    print(f"       {qid}: {queries[qid]}")
    
    # 6. Calculate impact on evaluation
    problematic_validation_pairs = 0
    for qid, docid in validation_pairs:
        # This pair will cause issues if document was retrieved by BM25 for a different query
        for doc_info in cross_query_docs:
            if doc_info['docid'] == docid:
                if qid in doc_info['qrels_queries'] and qid not in doc_info['bm25_queries']:
                    problematic_validation_pairs += 1
                    break
    
    print(f"\n6. EVALUATION IMPACT:")
    print(f"   Validation pairs that will cause KeyError: {problematic_validation_pairs}")
    print(f"   Percentage of problematic pairs: {problematic_validation_pairs/len(validation_pairs)*100:.2f}%")
    
    # 7. Solutions
    print(f"\n7. SOLUTIONS:")
    
    if len(cross_query_docs) == 0:
        print(f"   ✅ No cross-query documents found!")
        print(f"   ✅ Your files should work correctly for evaluation")
    else:
        print(f"   Solution 1: Filter validation file to remove problematic pairs")
        print(f"   Solution 2: Regenerate BM25 ensuring query consistency") 
        print(f"   Solution 3: Modify evaluation script to handle missing candidates gracefully")
        
        # Generate a clean validation file
        clean_validation_file = '/workspace/2404170001/validation_2/pyserini_data_fixed/validation_clean.tsv'
        clean_pairs = []
        
        for qid, docid in validation_pairs:
            # Only include if document was retrieved by BM25 for this query OR if it's a qrels-only relevant doc
            doc_bm25_queries = set(q[0] for q in bm25_doc_to_queries.get(docid, []))
            doc_qrels_queries = set(q[0] for q in qrels_doc_to_queries.get(docid, []))
            
            # Include if: 
            # 1. Document was retrieved by BM25 for this query, OR
            # 2. Document is relevant in qrels for this query AND wasn't retrieved by BM25 for any other query
            if (qid in doc_bm25_queries) or (qid in doc_qrels_queries and len(doc_bm25_queries) == 0):
                clean_pairs.append((qid, docid))
        
        # Save clean validation file
        with open(clean_validation_file, 'w') as f:
            for qid, docid in clean_pairs:
                if qid in queries:
                    # You'll need to load collection here or modify as needed
                    f.write(f"{qid}\t{docid}\tplaceholder_query\tplaceholder_doc\n")
        
        print(f"   Generated clean validation file: {clean_validation_file}")
        print(f"   Clean pairs: {len(clean_pairs)} (reduced from {len(validation_pairs)})")
    
    return {
        'cross_query_docs': len(cross_query_docs),
        'multi_relevant_docs': len(multi_relevant_docs), 
        'qrels_only_docs': len(qrels_only_docs),
        'problematic_pairs': problematic_validation_pairs,
        'total_pairs': len(validation_pairs)
    }

# Additional helper function to regenerate validation correctly
def regenerate_validation_with_bm25_queries():
    """
    Regenerate validation file using EXACTLY the queries from BM25 results
    """
    print("=== REGENERATING VALIDATION WITH BM25 QUERIES ===")
    
    # File paths
    bm25_file = '/workspace/2404170001/validation_2/pyserini_data_fixed/bm25_run.txt'
    qrels_file = '/workspace/2404170001/inputs/qrels.dev.tsv'
    queries_file = '/workspace/2404170001/inputs/queries.dev.small.tsv'
    collection_file = '/workspace/2404170001/inputs/collection.tsv'
    
    # Load source data
    print("Loading source data...")
    
    # Queries
    queries = {}
    with open(queries_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                qid, qtext = str(parts[0]), parts[1]
                queries[qid] = qtext
    
    # Collection
    collection = {}
    with open(collection_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                docid, text = str(parts[0]), parts[1]
                collection[docid] = text[:100000]
    
    # Qrels
    qrels = defaultdict(dict)
    with open(qrels_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, docid, rel = str(parts[0]), parts[1], str(parts[2]), float(parts[3])
                if rel > 0:
                    qrels[qid][docid] = rel
    
    # Get EXACT query set from BM25
    bm25_queries = set()
    bm25_pairs = set()
    with open(bm25_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                qid, docid = str(parts[0]), str(parts[2])
                bm25_queries.add(qid)
                bm25_pairs.add((qid, docid))
    
    print(f"BM25 queries: {len(bm25_queries)}")
    print(f"BM25 pairs: {len(bm25_pairs)}")
    
    # Generate new validation file
    new_validation_file = '/workspace/2404170001/validation_2/pyserini_data_fixed/validation_regenerated.tsv'
    known_pairs = set()
    
    print("Generating new validation file...")
    with open(new_validation_file, 'w') as f:
        
        # Add ALL BM25 pairs first
        for qid, docid in bm25_pairs:
            if qid in queries and docid in collection:
                if (qid, docid) not in known_pairs:
                    known_pairs.add((qid, docid))
                    row = [qid, docid, queries[qid], collection[docid]]
                    f.write('\t'.join(row) + '\n')
        
        # Add relevant documents from qrels for BM25 queries ONLY
        for qid in bm25_queries:  # Only process BM25 queries
            if qid in qrels:
                for docid in qrels[qid]:
                    if (qid, docid) not in known_pairs:
                        if qid in queries and docid in collection:
                            known_pairs.add((qid, docid))
                            row = [qid, docid, queries[qid], collection[docid]]
                            f.write('\t'.join(row) + '\n')
    
    print(f"Generated validation file: {new_validation_file}")
    print(f"Total pairs: {len(known_pairs)}")
    
    # Verify the new file
    print("\nVerifying new file...")
    new_df = pd.read_csv(new_validation_file, sep='\t', header=None,
                         names=['qid', 'doc_id', 'query', 'doc'],
                         dtype=str)
    
    new_validation_queries = set(new_df['qid'])
    new_validation_pairs = set(zip(new_df['qid'], new_df['doc_id']))
    
    print(f"New validation queries: {len(new_validation_queries)}")
    print(f"New validation pairs: {len(new_validation_pairs)}")
    
    # Final check
    query_match = new_validation_queries == bm25_queries
    all_bm25_included = len(bm25_pairs - new_validation_pairs) == 0
    
    print(f"Query sets match: {query_match}")
    print(f"All BM25 pairs included: {all_bm25_included}")
    
    if query_match and all_bm25_included:
        print("🎉 SUCCESS! New validation file should work perfectly!")
        return new_validation_file
    else:
        print("❌ Still have issues...")
        return None

if __name__ == "__main__":
    # Run the diagnostic first
    print("Running diagnostic...")
    results = debug_query_document_mismatch()
    
    if results is None:
        print("\n" + "="*50)
        print("QUERY SET MISMATCH DETECTED")
        print("="*50)
        print("Running regeneration...")
        new_file = regenerate_validation_with_bm25_queries()
        
        if new_file:
            print(f"\n✅ SUCCESS! Use this validation file: {new_file}")
        else:
            print("\n❌ Regeneration failed - need manual intervention")
    else:
        print(f"\nDiagnostic completed. Found {results['cross_query_docs']} problematic documents.")