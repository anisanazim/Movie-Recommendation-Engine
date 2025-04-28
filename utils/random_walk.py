import torch
import numpy as np
from collections import Counter
from tqdm import tqdm

class RandomWalkSampler:
    """
    Implements random walk-based neighborhood sampling as described in the PinSage paper.
    Uses Personalized PageRank (PPR) to identify important neighbors for each node.
    """
    def __init__(self, edge_index, edge_weights=None, walk_length=2, num_walks=100, p=1.0, q=1.0):
        """
        Initialize the random walk sampler.
        
        Args:
            edge_index (torch.LongTensor): Edge index with shape [2, num_edges]
            edge_weights (torch.FloatTensor, optional): Edge weights
            walk_length (int): Length of each random walk
            num_walks (int): Number of random walks to perform per node
            p (float): Return parameter (1 = unbiased)
            q (float): In-out parameter (1 = unbiased)
        """
        self.edge_index = edge_index
        self.edge_weights = edge_weights
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        
        # Prepare adjacency list for efficient random walk
        self._prepare_adjacency_list()
        
    def _prepare_adjacency_list(self):
        """Prepare adjacency list for efficient random walk."""
        # Get max node index
        max_node_idx = self.edge_index.max().item() + 1
        
        # Initialize adjacency list
        self.adj_list = [[] for _ in range(max_node_idx)]
        
        # Fill adjacency list with neighbors and weights
        for i in range(self.edge_index.size(1)):
            src, dst = self.edge_index[0, i].item(), self.edge_index[1, i].item()
            
            if self.edge_weights is not None:
                weight = self.edge_weights[i].item()
            else:
                weight = 1.0
                
            self.adj_list[src].append((dst, weight))
    
    def _single_walk(self, start_node):
        """
        Perform a single random walk starting from start_node.
        
        Args:
            start_node (int): Starting node for the walk
            
        Returns:
            walk (list): List of nodes visited during the walk
        """
        walk = [start_node]
        current_node = start_node
        
        for _ in range(self.walk_length):
            neighbors = self.adj_list[current_node]
            
            if not neighbors:
                break
                
            # Extract destinations and weights
            destinations = [neighbor[0] for neighbor in neighbors]
            weights = np.array([neighbor[1] for neighbor in neighbors])
            
            # Normalize weights to probabilities
            probabilities = weights / weights.sum()
            
            # Select next node based on probabilities
            next_node = np.random.choice(destinations, p=probabilities)
            walk.append(next_node)
            current_node = next_node
            
        return walk
    
    def sample_neighbors(self, node_idx, num_neighbors=10):
        """
        Sample important neighbors for a node using random walks.
        
        Args:
            node_idx (int): Index of the target node
            num_neighbors (int): Number of neighbors to sample
            
        Returns:
            neighbors (list): List of sampled neighbor indices
            weights (list): Importance weights for each neighbor
        """
        # Perform multiple random walks
        walks = [self._single_walk(node_idx) for _ in range(self.num_walks)]
        
        # Count the visits (excluding the starting node)
        visit_counts = Counter()
        for walk in walks:
            for node in walk[1:]:  # Skip the starting node
                visit_counts[node] += 1
        
        # Get the top num_neighbors nodes with highest visit counts
        top_neighbors = sorted(visit_counts.items(), key=lambda x: x[1], reverse=True)[:num_neighbors]
        
        if not top_neighbors:
            return [], []
            
        # Extract neighbors and their normalized visit counts (weights)
        neighbors, counts = zip(*top_neighbors)
        total_counts = sum(counts)
        weights = [count / total_counts for count in counts]
        
        return list(neighbors), weights
    
    def batch_sample_neighbors(self, nodes, num_neighbors=10):
        """
        Sample important neighbors for a batch of nodes.
        
        Args:
            nodes (list or torch.Tensor): List of node indices
            num_neighbors (int): Number of neighbors to sample per node
            
        Returns:
            all_neighbors (list): List of lists containing sampled neighbors for each node
            all_weights (list): List of lists containing weights for sampled neighbors
        """
        if isinstance(nodes, torch.Tensor):
            nodes = nodes.tolist()
            
        all_neighbors = []
        all_weights = []
        
        for node in tqdm(nodes, desc="Sampling neighbors", disable=len(nodes) < 1000):
            neighbors, weights = self.sample_neighbors(node, num_neighbors)
            all_neighbors.append(neighbors)
            all_weights.append(weights)
            
        return all_neighbors, all_weights

    def compute_ppr_matrix(self, nodes, alpha=0.15, num_iterations=10):
        """
        Compute approximate Personalized PageRank scores for efficient neighborhood sampling.
        
        Args:
            nodes (list or torch.Tensor): List of node indices
            alpha (float): Teleport probability
            num_iterations (int): Number of power iterations
            
        Returns:
            ppr_matrix (dict): Dictionary mapping (source, target) pairs to PPR scores
        """
        if isinstance(nodes, torch.Tensor):
            nodes = nodes.tolist()
        
        max_node_idx = max(self.edge_index.max().item() + 1, max(nodes) + 1)
        ppr_matrix = {}
        
        for source in tqdm(nodes, desc="Computing PPR", disable=len(nodes) < 1000):
            # Initialize PPR vector
            ppr = np.zeros(max_node_idx)
            ppr[source] = 1.0
            
            # Initialize residual vector
            residual = np.zeros(max_node_idx)
            residual[source] = 1.0
            
            # Power iteration
            for _ in range(num_iterations):
                # Push operation
                for node, res in enumerate(residual):
                    if res > 0:
                        # Teleport
                        ppr[node] += alpha * res
                        
                        # Distribute residual
                        neighbors = self.adj_list[node]
                        if neighbors:
                            push_val = (1 - alpha) * res
                            for neighbor, weight in neighbors:
                                norm_weight = weight / sum(w for _, w in neighbors)
                                residual[neighbor] += push_val * norm_weight
                        
                        # Reset residual
                        residual[node] = 0
            
            # Store PPR scores
            for target, score in enumerate(ppr):
                if score > 0:
                    ppr_matrix[(source, target)] = score
        
        return ppr_matrix

    def precompute_top_neighbors(self, nodes, num_neighbors=10):
        """
        Precompute the top neighbors for each node using PPR.
        
        Args:
            nodes (list or torch.Tensor): List of node indices
            num_neighbors (int): Number of neighbors to precompute per node
            
        Returns:
            top_neighbors (dict): Dictionary mapping each node to its top neighbors and weights
        """
        # Compute PPR matrix
        ppr_matrix = self.compute_ppr_matrix(nodes)
        
        # Extract top neighbors for each node
        top_neighbors = {}
        for source in nodes:
            # Get all PPR scores for this source
            scores = [(target, score) for (src, target), score in ppr_matrix.items() if src == source]
            
            # Sort by score and take top num_neighbors
            scores.sort(key=lambda x: x[1], reverse=True)
            neighbors = [target for target, _ in scores[:num_neighbors]]
            weights = [score for _, score in scores[:num_neighbors]]
            
            # Normalize weights
            if weights:
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]
            
            top_neighbors[source] = (neighbors, weights)
        
        return top_neighbors