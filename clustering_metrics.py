import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import os

def compute_clustering_metrics(run_folder):
    """
    Compute clustering evaluation metrics from saved cluster files
    
    Args:
        run_folder: Path to the experiment folder containing saved cluster files
    
    Returns:
        dict: Dictionary containing all clustering metrics
    """
    
    # Load the saved query vectors and IDs
    print("Loading query vectors...")
    data_file = os.path.join(run_folder, "query_vectors_n_ids.npz")
    data = np.load(data_file)
    
    # Extract components
    query_embeddings = data["storage"]  # The actual query embeddings/vectors
    id_mapping = data["id_mapping"]     # ID mapping
    seq_ids = data["seq_ids"]           # Sequence IDs
    
    print(f"Loaded {len(query_embeddings)} query embeddings with dimension {query_embeddings.shape[1]}")
    
    # Load cluster assignments
    print("Loading cluster assignments...")
    cluster_file = os.path.join(run_folder, "cluster-assignment-ids.tsv")
    
    # Create a mapping from query ID to cluster label
    query_to_cluster = {}
    cluster_labels = []
    
    with open(cluster_file, 'r', encoding='utf-8') as f:
        for cluster_id, line in enumerate(f):
            query_ids = line.strip().split('\t')
            for query_id in query_ids:
                if query_id:  # Skip empty strings
                    query_to_cluster[query_id] = cluster_id
    
    # Create cluster labels array matching the order of embeddings
    cluster_labels = []
    valid_indices = []
    
    for i, seq_id in enumerate(seq_ids):
        if seq_id in query_to_cluster:
            cluster_labels.append(query_to_cluster[seq_id])
            valid_indices.append(i)
        else:
            print(f"Warning: Query ID {seq_id} not found in cluster assignments")
    
    # Filter embeddings to only include those with cluster assignments
    filtered_embeddings = query_embeddings[valid_indices]
    cluster_labels = np.array(cluster_labels)
    
    print(f"Computing metrics for {len(filtered_embeddings)} queries across {len(np.unique(cluster_labels))} clusters")
    
    # Compute clustering metrics
    metrics = {}
    
    try:
        # Silhouette Score: Measures cluster separation and cohesion (-1 to 1, higher is better)
        sil_score = silhouette_score(filtered_embeddings, cluster_labels)
        metrics['silhouette_score'] = float(sil_score)
        print(f"Silhouette Score: {sil_score:.4f}")
        
        # Calinski-Harabasz Index: Evaluates cluster definition quality (higher is better)
        ch_score = calinski_harabasz_score(filtered_embeddings, cluster_labels)
        metrics['calinski_harabasz_score'] = float(ch_score)
        print(f"Calinski-Harabasz Index: {ch_score:.4f}")
        
        # Davies-Bouldin Index: Assesses cluster similarity and separation (lower is better)
        db_score = davies_bouldin_score(filtered_embeddings, cluster_labels)
        metrics['davies_bouldin_score'] = float(db_score)
        print(f"Davies-Bouldin Index: {db_score:.4f}")
        
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return None
    
    # Additional cluster statistics
    unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
    metrics['num_clusters'] = int(len(unique_clusters))
    metrics['avg_cluster_size'] = float(np.mean(cluster_counts))
    metrics['min_cluster_size'] = int(np.min(cluster_counts))
    metrics['max_cluster_size'] = int(np.max(cluster_counts))
    metrics['cluster_size_std'] = float(np.std(cluster_counts))
    
    print(f"\nCluster Statistics:")
    print(f"Number of clusters: {metrics['num_clusters']}")
    print(f"Average cluster size: {metrics['avg_cluster_size']:.2f}")
    print(f"Min cluster size: {metrics['min_cluster_size']}")
    print(f"Max cluster size: {metrics['max_cluster_size']}")
    print(f"Cluster size std: {metrics['cluster_size_std']:.2f}")
    
    return metrics

# Example usage
if __name__ == "__main__":
    # Replace with your actual run folder path
    run_folder = "/workspace/2404170001/clustering_output1"
    
    metrics = compute_clustering_metrics(run_folder)
    
    if metrics:
        # Save metrics to file
        import json
        with open(os.path.join(run_folder, "clustering_metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nMetrics saved to: {os.path.join(run_folder, 'clustering_metrics.json')}")