import torch
import numpy as np
from utils.random_walk import RandomWalkSampler

class NegativeSampler:
    """
    Implements negative sampling strategies for training the PinSage model.
    """
    def __init__(self, dataset, random_walk_sampler=None, num_negative_samples=500):
        """
        Initialize the negative sampler.
        
        Args:
            dataset: MovieLensDataset instance containing the processed data
            random_walk_sampler: RandomWalkSampler instance for generating hard negative samples
            num_negative_samples (int): Number of negative samples to generate per batch
        """
        self.dataset = dataset
        self.random_walk_sampler = random_walk_sampler
        self.num_negative_samples = num_negative_samples
        
        # Get movie indices
        self.all_movie_indices = list(range(len(dataset.movie_id_to_idx)))
        
    def sample_random_negatives(self, batch_size, device):
        """
        Sample random negative items for a batch.
        
        Args:
            batch_size (int): Size of the batch
            device: Device to put tensors on
            
        Returns:
            negative_indices (torch.LongTensor): Indices of sampled negative items
        """
        # Sample random movie indices
        negative_indices = torch.tensor(
            np.random.choice(self.all_movie_indices, size=self.num_negative_samples, replace=False),
            device=device
        )
        
        return negative_indices
    
    def sample_hard_negatives(self, query_indices, num_hard_samples=5, max_rank=5000, min_rank=2000):
        """
        Sample hard negative examples using random walks.
        
        Args:
            query_indices (torch.LongTensor): Indices of query items
            num_hard_samples (int): Number of hard negative samples per query
            max_rank (int): Maximum rank for considering hard negatives
            min_rank (int): Minimum rank for considering hard negatives
            
        Returns:
            hard_negative_indices (torch.LongTensor): Indices of hard negative items
        """
        if self.random_walk_sampler is None:
            raise ValueError("RandomWalkSampler is required for hard negative sampling")
        
        hard_negatives = []
        
        for idx in query_indices.cpu().numpy():
            # Perform random walks to rank items by similarity
            visited_counts = {}
            
            # Perform multiple random walks
            for _ in range(100):  # Number of walks
                walk = self.random_walk_sampler._single_walk(idx)
                for node in walk[1:]:  # Exclude starting node
                    visited_counts[node] = visited_counts.get(node, 0) + 1
            
            # Sort by visit count (higher count = more similar)
            ranked_items = sorted(visited_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Select items in the desired rank range (not too similar, not too dissimilar)
            candidates = [item for item, _ in ranked_items[min_rank:max_rank] 
                        if item in self.all_movie_indices]
            
            if not candidates:
                # If no suitable candidates, sample randomly
                sampled = np.random.choice(self.all_movie_indices, size=num_hard_samples, replace=False)
            else:
                # Sample from candidates
                sampled = np.random.choice(candidates, 
                                        size=min(num_hard_samples, len(candidates)), 
                                        replace=False)
                
                # If we didn't get enough, fill with random samples
                if len(sampled) < num_hard_samples:
                    additional = np.random.choice(
                        [i for i in self.all_movie_indices if i not in sampled],
                        size=num_hard_samples - len(sampled),
                        replace=False
                    )
                    sampled = np.concatenate([sampled, additional])
            
            hard_negatives.append(sampled)
        
        return torch.tensor(hard_negatives, device=query_indices.device)
    
    def sample_batch_negatives(self, query_indices, device, epoch=0):
        """
        Sample negative items for a batch, including both random and hard negatives.
        
        Args:
            query_indices (torch.LongTensor): Indices of query items
            device: Device to put tensors on
            epoch (int): Current training epoch
            
        Returns:
            random_negatives (torch.LongTensor): Indices of random negative items
            hard_negatives (torch.LongTensor, optional): Indices of hard negative items
        """
        # Sample random negatives (shared across the batch)
        random_negatives = self.sample_random_negatives(len(query_indices), device)
        
        # If we're past the first epoch, sample hard negatives
        if epoch >= 1 and self.random_walk_sampler is not None:
            # Number of hard negatives increases with epoch
            num_hard = min(epoch, 6)  # Maximum of 6 hard negatives per query
            hard_negatives = self.sample_hard_negatives(query_indices, num_hard_samples=num_hard)
            return random_negatives, hard_negatives
        
        return random_negatives, None