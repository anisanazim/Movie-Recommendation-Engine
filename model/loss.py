import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MaxMarginRankingLoss(nn.Module):
    """
    Max-margin ranking loss for PinSage model training.
    Implements the loss function described in the PinSage paper:
    https://arxiv.org/abs/1806.01973
    """
    def __init__(self, margin=0.1):
        """
        Initialize the max-margin ranking loss.
        
        Args:
            margin (float): Margin for the loss function
        """
        super(MaxMarginRankingLoss, self).__init__()
        self.margin = margin
        
    def forward(self, query_embeddings, positive_embeddings, negative_embeddings):
        """
        Forward pass for the max-margin ranking loss.
        
        Args:
            query_embeddings (torch.Tensor): Embeddings of query items
            positive_embeddings (torch.Tensor): Embeddings of positive (related) items
            negative_embeddings (torch.Tensor): Embeddings of negative (unrelated) items
            
        Returns:
            loss (torch.Tensor): Computed loss value
        """
        # Compute similarities with positive examples
        pos_sim = torch.sum(query_embeddings * positive_embeddings, dim=1)
        
        # Compute similarities with negative examples
        if negative_embeddings.dim() == 3:
            # Multiple negative examples per query
            # [batch_size, num_negatives, embed_dim]
            batch_size, num_negatives, _ = negative_embeddings.size()
            
            # Expand query embeddings to match negative embeddings
            # [batch_size, 1, embed_dim] -> [batch_size, num_negatives, embed_dim]
            expanded_query = query_embeddings.unsqueeze(1).expand_as(negative_embeddings)
            
            # Compute similarities for all negative examples
            # [batch_size, num_negatives]
            neg_sim = torch.sum(expanded_query * negative_embeddings, dim=2)
            
            # Find the max similarity among negative examples
            # [batch_size]
            max_neg_sim, _ = torch.max(neg_sim, dim=1)
            
            # Compute hinge loss: max(0, margin + max_neg_sim - pos_sim)
            loss = F.relu(self.margin + max_neg_sim - pos_sim)
        else:
            # Single negative example per query
            neg_sim = torch.sum(query_embeddings * negative_embeddings, dim=1)
            
            # Compute hinge loss: max(0, margin + neg_sim - pos_sim)
            loss = F.relu(self.margin + neg_sim - pos_sim)
        
        return loss.mean()

class BatchHardTripletLoss(nn.Module):
    """
    Batch hard triplet loss for PinSage model training.
    This loss uses in-batch negatives and finds the hardest triplets.
    """
    def __init__(self, margin=0.1):
        """
        Initialize the batch hard triplet loss.
        
        Args:
            margin (float): Margin for the triplet loss
        """
        super(BatchHardTripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, query_embeddings, positive_embeddings):
        """
        Forward pass for the batch hard triplet loss.
        
        Args:
            query_embeddings (torch.Tensor): Embeddings of query items [batch_size, embed_dim]
            positive_embeddings (torch.Tensor): Embeddings of positive items [batch_size, embed_dim]
            
        Returns:
            loss (torch.Tensor): Computed loss value
        """
        batch_size = query_embeddings.size(0)
        
        # Compute pairwise similarity matrix
        # [batch_size, batch_size]
        sim_matrix = torch.matmul(query_embeddings, positive_embeddings.t())
        
        # Mask out diagonal elements (positive pairs)
        mask = torch.eye(batch_size, device=query_embeddings.device)
        masked_sim = sim_matrix * (1 - mask) - mask * 1e9  # Large negative value for diagonal
        
        # Find hardest negative for each query
        # [batch_size]
        hardest_neg_sim, _ = torch.max(masked_sim, dim=1)
        
        # Compute similarities for positive pairs
        # [batch_size]
        pos_sim = torch.sum(query_embeddings * positive_embeddings, dim=1)
        
        # Compute triplet loss: max(0, margin + hardest_neg_sim - pos_sim)
        loss = F.relu(self.margin + hardest_neg_sim - pos_sim)
        
        return loss.mean()

class CurriculumLoss(nn.Module):
    """
    Curriculum learning loss for PinSage as described in the paper.
    Adds progressively harder negative examples during training.
    """
    def __init__(self, margin=0.1, epoch=0, max_epochs=10, hard_negative_factor=2.0):
        """
        Initialize the curriculum learning loss.
        
        Args:
            margin (float): Margin for the loss function
            epoch (int): Current training epoch
            max_epochs (int): Maximum number of training epochs
            hard_negative_factor (float): Factor to scale the hard negative loss
        """
        super(CurriculumLoss, self).__init__()
        self.margin = margin
        self.epoch = epoch
        self.max_epochs = max_epochs
        self.hard_negative_factor = hard_negative_factor
        
        # Base loss function
        self.base_loss = MaxMarginRankingLoss(margin)
        
    def update_epoch(self, epoch):
        """Update the current epoch."""
        self.epoch = epoch
        
    def forward(self, query_embeddings, positive_embeddings, 
                random_negative_embeddings, hard_negative_embeddings=None):
        """
        Forward pass for the curriculum learning loss.
        
        Args:
            query_embeddings (torch.Tensor): Embeddings of query items
            positive_embeddings (torch.Tensor): Embeddings of positive items
            random_negative_embeddings (torch.Tensor): Embeddings of random negative items
            hard_negative_embeddings (torch.Tensor, optional): Embeddings of hard negative items
            
        Returns:
            loss (torch.Tensor): Computed loss value
        """
        # Compute base loss with random negatives
        base_loss = self.base_loss(query_embeddings, positive_embeddings, random_negative_embeddings)
        
        # If we're in early epochs or no hard negatives provided, return base loss
        if self.epoch < 1 or hard_negative_embeddings is None:
            return base_loss
        
        # Compute loss with hard negatives
        hard_loss = self.base_loss(query_embeddings, positive_embeddings, hard_negative_embeddings)
        
        # Compute the number of hard negatives to use based on curriculum
        # At epoch 1, we use 1 hard negative, and gradually increase
        num_hard_negatives = min(self.epoch, self.max_epochs)
        
        # Scale the hard negative loss based on the number of hard negatives
        hard_weight = num_hard_negatives / self.max_epochs * self.hard_negative_factor
        
        # Combine losses
        total_loss = base_loss + hard_weight * hard_loss
        
        return total_loss