# Replace the FaissIVFIndexer clustering section with DBSCAN
# Add these imports at the top of your query_clusterer.py file

from sklearn.cluster import DBSCAN
import numpy as np

# Replace the indexing/clustering section in query_clusterer.py
# Original code uses FaissIVFIndexer, replace with this DBSCAN implementation:

def create_dbscan_clusters(embeddings, eps=0.5, min_samples=5):
    """
    Create clusters using DBSCAN instead of k-means
    
    Args:
        embeddings: numpy array of query embeddings
        eps: The maximum distance between two samples for one to be considered 
             as in the neighborhood of the other
        min_samples: The number of samples in a neighborhood for a point to be 
                    considered as a core point
    
    Returns:
        cluster_labels: array of cluster assignments (-1 for noise)
        n_clusters: number of clusters found
    """
    # Normalize embeddings for better clustering
    embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    cluster_labels = dbscan.fit_predict(embeddings_normalized)
    
    # Number of clusters (excluding noise points labeled as -1)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points")
    
    return cluster_labels, n_clusters

# Modified clustering section for query_clusterer.py
# Replace the section that creates the indexer and gets cluster assignments:

# Instead of:
# indexer = FaissIVFIndexer(config)
# indexer.prepare([output_storage])
# indexer.index([seq_ids], [output_storage])
# clusters = indexer.get_all_cluster_assignments()

# Use this:
def cluster_queries_with_dbscan(output_storage, seq_ids, eps=0.5, min_samples=5):
    """
    Cluster queries using DBSCAN instead of FAISS k-means
    """
    # Convert storage to numpy array if needed
    if isinstance(output_storage, list):
        embeddings = np.concatenate(output_storage)
    else:
        embeddings = output_storage
    
    # Get cluster labels from DBSCAN
    cluster_labels, n_clusters = create_dbscan_clusters(embeddings, eps, min_samples)
    
    # Organize query IDs by cluster
    clusters = [[] for _ in range(n_clusters)]
    noise_cluster = []  # For queries labeled as noise (-1)
    
    for i, (seq_id, cluster_id) in enumerate(zip(seq_ids, cluster_labels)):
        if cluster_id == -1:
            noise_cluster.append(seq_id)
        else:
            clusters[cluster_id].append(seq_id)
    
    # Add noise points as separate small clusters if desired
    # or distribute them among existing clusters
    if noise_cluster:
        print(f"Adding {len(noise_cluster)} noise points as separate clusters")
        for noise_id in noise_cluster:
            clusters.append([noise_id])
    
    # Filter out empty clusters
    clusters = [cluster for cluster in clusters if len(cluster) > 0]
    
    return clusters

# In your main clustering section, replace the FAISS indexing with:
# clusters = cluster_queries_with_dbscan(output_storage, seq_ids, eps=0.3, min_samples=3)

# Note: You may need to tune eps and min_samples parameters:
# - eps: smaller values = more clusters, larger values = fewer clusters  
# - min_samples: minimum points needed to form a cluster
# Recommended starting values: eps=0.3-0.7, min_samples=3-10