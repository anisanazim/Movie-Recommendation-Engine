import torch
import numpy as np
from tqdm import tqdm

def calculate_hit_rate(item_embeddings, query_indices, ground_truth_indices, k=500):
    """
    Calculate hit rate@k for recommendation evaluation.
    
    Args:
        item_embeddings (torch.Tensor): Embeddings of all items
        query_indices (list): Indices of query items
        ground_truth_indices (list): Indices of ground truth items corresponding to queries
        k (int): Number of items to consider in the recommendations
        
    Returns:
        hit_rate (float): Hit rate@k
    """
    hits = 0
    total = len(query_indices)
    
    for i, query_idx in enumerate(tqdm(query_indices, desc="Calculating Hit Rate", disable=len(query_indices) < 1000)):
        # Get query embedding
        query_embedding = item_embeddings[query_idx]
        
        # Compute similarity with all items
        similarities = torch.matmul(query_embedding, item_embeddings.t())
        
        # Get top-k items
        _, indices = torch.topk(similarities, k=k)
        
        # Check if ground truth is in top-k
        if ground_truth_indices[i] in indices.numpy():
            hits += 1
    
    hit_rate = hits / total
    return hit_rate

def calculate_mrr(item_embeddings, query_indices, ground_truth_indices, scale=100):
    """
    Calculate Mean Reciprocal Rank (MRR) for recommendation evaluation.
    
    Args:
        item_embeddings (torch.Tensor): Embeddings of all items
        query_indices (list): Indices of query items
        ground_truth_indices (list): Indices of ground truth items corresponding to queries
        scale (int): Scaling factor for MRR calculation
        
    Returns:
        mrr (float): Mean Reciprocal Rank
    """
    reciprocal_ranks = []
    
    for i, query_idx in enumerate(tqdm(query_indices, desc="Calculating MRR", disable=len(query_indices) < 1000)):
        # Get query embedding
        query_embedding = item_embeddings[query_idx]
        
        # Compute similarity with all items
        similarities = torch.matmul(query_embedding, item_embeddings.t())
        
        # Get rankings
        _, indices = torch.sort(similarities, descending=True)
        indices = indices.numpy()
        
        # Find rank of ground truth
        gt_idx = ground_truth_indices[i]
        rank = np.where(indices == gt_idx)[0][0] + 1  # +1 because ranks start from 1
        
        # Calculate reciprocal rank (scaled)
        reciprocal_rank = 1.0 / (rank / scale)
        reciprocal_ranks.append(reciprocal_rank)
    
    mrr = np.mean(reciprocal_ranks)
    return mrr

def evaluate_embeddings(item_embeddings, test_data, k_values=[10, 50, 100, 500]):
    """
    Comprehensive evaluation of embeddings using multiple metrics.
    
    Args:
        item_embeddings (torch.Tensor): Embeddings of all items
        test_data (dict): Test data containing positive pairs
        k_values (list): List of k values for hit rate calculation
        
    Returns:
        results (dict): Dictionary containing evaluation results
    """
    # Get positive pairs
    positive_pairs = test_data['positive_pairs']
    query_indices = positive_pairs[:, 0].numpy()
    ground_truth_indices = positive_pairs[:, 1].numpy()
    
    # Results dictionary
    results = {}
    
    # Calculate hit rate for different k values
    for k in k_values:
        hit_rate = calculate_hit_rate(item_embeddings, query_indices, ground_truth_indices, k=k)
        results[f'hit_rate@{k}'] = hit_rate
    
    # Calculate MRR
    mrr = calculate_mrr(item_embeddings, query_indices, ground_truth_indices)
    results['mrr'] = mrr
    
    return results

def generate_recommendations(item_embeddings, query_idx, k=10, exclude_query=True):
    """
    Generate recommendations for a query item.
    
    Args:
        item_embeddings (torch.Tensor): Embeddings of all items
        query_idx (int): Index of the query item
        k (int): Number of recommendations to generate
        exclude_query (bool): Whether to exclude the query item from recommendations
        
    Returns:
        recommendations (list): Indices of recommended items
    """
    # Get query embedding
    query_embedding = item_embeddings[query_idx]
    
    # Compute similarity with all items
    similarities = torch.matmul(query_embedding, item_embeddings.t())
    
    if exclude_query:
        # Set similarity with query item to a very low value
        similarities[query_idx] = -float('inf')
    
    # Get top-k items
    _, indices = torch.topk(similarities, k=k)
    
    return indices.numpy()