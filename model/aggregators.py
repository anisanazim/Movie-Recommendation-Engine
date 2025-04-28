import torch
import torch.nn as nn
import torch.nn.functional as F

class MeanAggregator(nn.Module):
    """
    Mean aggregator for graph neural networks.
    """
    def __init__(self):
        """Initialize the mean aggregator."""
        super(MeanAggregator, self).__init__()
        
    def forward(self, features, neighbors):
        """
        Aggregate features from neighbors using mean pooling.
        
        Args:
            features (torch.Tensor): Node feature matrix
            neighbors (list): List of lists containing neighbor indices for each node
            
        Returns:
            aggregated (torch.Tensor): Aggregated neighborhood features
        """
        aggregated_list = []
        
        for i, node_neighbors in enumerate(neighbors):
            if not node_neighbors:
                # If no neighbors, use zeros
                aggregated_list.append(torch.zeros_like(features[0]))
                continue
                
            # Get all neighbor features
            neighbor_features = features[node_neighbors]
            
            # Compute mean
            aggregated = torch.mean(neighbor_features, dim=0)
            aggregated_list.append(aggregated)
            
        return torch.stack(aggregated_list)

class WeightedAggregator(nn.Module):
    """
    Weighted aggregator for graph neural networks.
    """
    def __init__(self):
        """Initialize the weighted aggregator."""
        super(WeightedAggregator, self).__init__()
        
    def forward(self, features, neighbors, weights):
        """
        Aggregate features from neighbors using weighted pooling.
        
        Args:
            features (torch.Tensor): Node feature matrix
            neighbors (list): List of lists containing neighbor indices for each node
            weights (list): List of lists containing weights for each neighbor
            
        Returns:
            aggregated (torch.Tensor): Aggregated neighborhood features
        """
        device = features.device
        aggregated_list = []
        
        for i, (node_neighbors, node_weights) in enumerate(zip(neighbors, weights)):
            if not node_neighbors:
                # If no neighbors, use zeros
                aggregated_list.append(torch.zeros_like(features[0]))
                continue
                
            # Get all neighbor features
            neighbor_features = features[node_neighbors]
            
            # Ensure weights match number of neighbors
            node_weights = node_weights[:len(node_neighbors)]
            
            # Convert weights to tensor and normalize
            weight_sum = sum(node_weights)
            if weight_sum == 0:
                # If all weights are zero, use mean
                aggregated = torch.mean(neighbor_features, dim=0)
            else:
                # Normalize weights
                normalized_weights = [w / weight_sum for w in node_weights]
                weight_tensor = torch.tensor(normalized_weights, dtype=torch.float, device=device).view(-1, 1)
                
                # Compute weighted sum
                aggregated = torch.sum(neighbor_features * weight_tensor, dim=0)
            
            aggregated_list.append(aggregated)
            
        return torch.stack(aggregated_list)

class AttentionAggregator(nn.Module):
    """
    Attention-based aggregator for graph neural networks.
    """
    def __init__(self, in_channels):
        """
        Initialize the attention aggregator.
        
        Args:
            in_channels (int): Number of input channels
        """
        super(AttentionAggregator, self).__init__()
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, 1)
        )
        
    def forward(self, features, neighbors, self_features=None):
        """
        Aggregate features from neighbors using attention mechanism.
        
        Args:
            features (torch.Tensor): Node feature matrix
            neighbors (list): List of lists containing neighbor indices for each node
            self_features (torch.Tensor, optional): Features of target nodes
            
        Returns:
            aggregated (torch.Tensor): Aggregated neighborhood features
        """
        if self_features is None:
            self_features = features
            
        device = features.device
        aggregated_list = []
        
        for i, node_neighbors in enumerate(neighbors):
            if not node_neighbors:
                # If no neighbors, use zeros
                aggregated_list.append(torch.zeros_like(features[0]))
                continue
                
            # Get all neighbor features
            neighbor_features = features[node_neighbors]
            
            # Get self feature
            self_feature = self_features[i]
            
            # Repeat self feature for each neighbor
            self_expanded = self_feature.expand(len(node_neighbors), -1)
            
            # Concatenate self and neighbor features
            concat_features = torch.cat([self_expanded, neighbor_features], dim=1)
            
            # Compute attention scores
            attention_scores = self.attention(concat_features)
            attention_weights = F.softmax(attention_scores, dim=0)
            
            # Apply attention weights
            weighted_features = neighbor_features * attention_weights
            
            # Sum weighted features
            aggregated = torch.sum(weighted_features, dim=0)
            aggregated_list.append(aggregated)
            
        return torch.stack(aggregated_list)

class MaxPoolingAggregator(nn.Module):
    """
    Max pooling aggregator for graph neural networks.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initialize the max pooling aggregator.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super(MaxPoolingAggregator, self).__init__()
        
        # MLP for each node before pooling
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU()
        )
        
    def forward(self, features, neighbors):
        """
        Aggregate features from neighbors using max pooling.
        
        Args:
            features (torch.Tensor): Node feature matrix
            neighbors (list): List of lists containing neighbor indices for each node
            
        Returns:
            aggregated (torch.Tensor): Aggregated neighborhood features
        """
        aggregated_list = []
        
        for i, node_neighbors in enumerate(neighbors):
            if not node_neighbors:
                # If no neighbors, use zeros
                aggregated_list.append(torch.zeros(self.mlp[0].out_features, device=features.device))
                continue
                
            # Get all neighbor features
            neighbor_features = features[node_neighbors]
            
            # Apply MLP
            transformed = self.mlp(neighbor_features)
            
            # Apply max pooling
            aggregated, _ = torch.max(transformed, dim=0)
            aggregated_list.append(aggregated)
            
        return torch.stack(aggregated_list)

class ImportanceAggregator(nn.Module):
    """
    Importance-based aggregator for PinSage architecture.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initialize the importance aggregator.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super(ImportanceAggregator, self).__init__()
        
        # Transformation before aggregation
        self.transform = nn.Linear(in_channels, out_channels)
        
        # Normalization
        self.norm = nn.LayerNorm(out_channels)
        
    def forward(self, features, neighbors, importance_weights):
        """
        Aggregate features from neighbors using importance weighting.
        
        Args:
            features (torch.Tensor): Node feature matrix
            neighbors (list): List of lists containing neighbor indices for each node
            importance_weights (list): List of lists containing importance weights
            
        Returns:
            aggregated (torch.Tensor): Aggregated neighborhood features
        """
        device = features.device
        aggregated_list = []
        
        for i, (node_neighbors, node_weights) in enumerate(zip(neighbors, importance_weights)):
            if not node_neighbors:
                # If no neighbors, use zeros
                aggregated_list.append(torch.zeros(self.transform.out_features, device=device))
                continue
                
            # Get all neighbor features
            neighbor_features = features[node_neighbors]
            
            # Apply transformation
            transformed = self.transform(neighbor_features)
            
            # Ensure weights match number of neighbors
            node_weights = node_weights[:len(node_neighbors)]
            
            # Convert weights to tensor and normalize
            weight_sum = sum(node_weights)
            if weight_sum == 0:
                # If all weights are zero, use mean
                aggregated = torch.mean(transformed, dim=0)
            else:
                # Normalize weights
                normalized_weights = [w / weight_sum for w in node_weights]
                weight_tensor = torch.tensor(normalized_weights, dtype=torch.float, device=device).view(-1, 1)
                
                # Compute weighted sum
                aggregated = torch.sum(transformed * weight_tensor, dim=0)
            
            # Apply normalization
            if aggregated.dim() == 1:
                # Expand for LayerNorm
                aggregated = aggregated.unsqueeze(0)
                aggregated = self.norm(aggregated)
                aggregated = aggregated.squeeze(0)
            else:
                aggregated = self.norm(aggregated)
            
            aggregated_list.append(aggregated)
            
        return torch.stack(aggregated_list)