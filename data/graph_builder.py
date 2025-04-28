import torch
import numpy as np
from tqdm import tqdm

class GraphBuilder:
    """
    Builds and manipulates graph structures for recommendation systems.
    """
    def __init__(self, dataset):
        """
        Initialize the graph builder.
        
        Args:
            dataset: MovieLensDataset instance containing the processed data
        """
        self.dataset = dataset
        
        # Graph components
        self.edge_index = None
        self.edge_weight = None
        
    def build_bipartite_graph(self):
        """
        Build a bipartite graph from user-item interactions.
        
        Returns:
            edge_index (torch.LongTensor): Edge index with shape [2, num_edges]
            edge_weight (torch.FloatTensor): Edge weights corresponding to the ratings
        """
        print("Building bipartite interaction graph...")
        
        ratings_df = self.dataset.ratings_df
        
        # Convert user and movie IDs to indices
        user_indices = [self.dataset.user_id_to_idx[user_id] for user_id in ratings_df['userId']]
        movie_indices = [self.dataset.movie_id_to_idx[movie_id] for movie_id in ratings_df['movieId']]
        
        # Add an offset to user indices to avoid index overlap with movie indices
        num_movies = len(self.dataset.movie_id_to_idx)
        user_indices_offset = [idx + num_movies for idx in user_indices]
        
        # Create a bidirectional edge index (user->movie and movie->user)
        edge_index = torch.LongTensor([
            user_indices_offset + movie_indices,
            movie_indices + user_indices_offset
        ])
        
        # Get edge weights from ratings
        ratings = ratings_df['rating'].values
        edge_weight = torch.FloatTensor(np.concatenate([ratings, ratings]))
        
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        
        print(f"Created bipartite graph with {len(ratings_df)} interactions (bidirectional)")
        
        return edge_index, edge_weight
    
    def build_item_similarity_graph(self, threshold=5):
        """
        Build an item similarity graph based on co-occurrence in user histories.
        
        Args:
            threshold (int): Minimum co-occurrence count to create an edge
            
        Returns:
            edge_index (torch.LongTensor): Edge index with shape [2, num_edges]
            edge_weight (torch.FloatTensor): Edge weights corresponding to co-occurrence counts
        """
        print("Building item similarity graph...")
        
        ratings_df = self.dataset.ratings_df
        
        # Group by user to get each user's rated items
        user_groups = ratings_df.groupby('userId')
        
        # Count co-occurrence of items
        item_co_occurrence = {}
        
        for _, group in tqdm(user_groups, desc="Computing co-occurrences"):
            items = [self.dataset.movie_id_to_idx[movie_id] for movie_id in group['movieId']]
            
            # Count co-occurrences for each pair of items
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    item_i, item_j = items[i], items[j]
                    
                    # Ensure item_i < item_j for consistent ordering
                    if item_i > item_j:
                        item_i, item_j = item_j, item_i
                    
                    # Update co-occurrence count
                    if (item_i, item_j) in item_co_occurrence:
                        item_co_occurrence[(item_i, item_j)] += 1
                    else:
                        item_co_occurrence[(item_i, item_j)] = 1
        
        # Filter by threshold and create edges
        src_nodes = []
        dst_nodes = []
        weights = []
        
        for (item_i, item_j), count in item_co_occurrence.items():
            if count >= threshold:
                # Add bidirectional edges
                src_nodes.extend([item_i, item_j])
                dst_nodes.extend([item_j, item_i])
                weights.extend([count, count])
        
        # Create edge index and weights
        edge_index = torch.LongTensor([src_nodes, dst_nodes])
        edge_weight = torch.FloatTensor(weights)
        
        print(f"Created item similarity graph with {len(src_nodes)//2} unique edges")
        
        return edge_index, edge_weight
    
    def get_adjacency_list(self, edge_index, edge_weight=None):
        """
        Convert edge_index to adjacency list format for efficient operations.
        
        Args:
            edge_index (torch.LongTensor): Edge index with shape [2, num_edges]
            edge_weight (torch.FloatTensor, optional): Edge weights
            
        Returns:
            adj_list (list): Adjacency list where adj_list[i] contains (neighbor, weight) tuples
        """
        # Get max node index
        max_node_idx = edge_index.max().item() + 1
        
        # Initialize adjacency list
        adj_list = [[] for _ in range(max_node_idx)]
        
        # Fill adjacency list with neighbors and weights
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            
            if edge_weight is not None:
                weight = edge_weight[i].item()
            else:
                weight = 1.0
                
            adj_list[src].append((dst, weight))
        
        return adj_list