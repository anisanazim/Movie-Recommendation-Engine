import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvLayer(nn.Module):
    """
    Graph convolutional layer implementation for the PinSage architecture.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initialize the graph convolutional layer.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super(GraphConvLayer, self).__init__()
        
        # Linear transformation for node's own features
        self.linear_self = nn.Linear(in_channels, out_channels)
        
        # Linear transformation for aggregated neighborhood features
        self.linear_neigh = nn.Linear(in_channels, out_channels)
        
        # Final transformation after concatenating self and neighborhood features
        self.linear_out = nn.Linear(2 * out_channels, out_channels)
        
        # Batch normalization
        self.bn = nn.BatchNorm1d(out_channels)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize the weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.linear_self.weight)
        nn.init.xavier_uniform_(self.linear_neigh.weight)
        nn.init.xavier_uniform_(self.linear_out.weight)
        
        nn.init.zeros_(self.linear_self.bias)
        nn.init.zeros_(self.linear_neigh.bias)
        nn.init.zeros_(self.linear_out.bias)
        
    def forward(self, x, neigh_x):
        """
        Forward pass for the graph convolutional layer.
        
        Args:
            x (torch.Tensor): Node feature matrix with shape [num_nodes, in_channels]
            neigh_x (torch.Tensor): Aggregated neighborhood features with shape [num_nodes, in_channels]
            
        Returns:
            out (torch.Tensor): Updated node representations with shape [num_nodes, out_channels]
        """
        # Transform node's own features
        self_feat = self.linear_self(x)
        
        # Transform aggregated neighborhood features
        neigh_feat = self.linear_neigh(neigh_x)
        
        # Concatenate transformed features
        combined = torch.cat([self_feat, neigh_feat], dim=1)
        
        # Apply final transformation
        out = self.linear_out(combined)
        
        # Apply batch normalization
        if out.size(0) > 1:  # Only apply if batch size > 1
            out = self.bn(out)
        
        # Apply non-linearity
        out = F.relu(out)
        
        # L2 normalize
        out = F.normalize(out, p=2, dim=1)
        
        return out

class ImportancePoolingLayer(nn.Module):
    """
    Implements importance pooling for aggregating neighborhood features.
    """
    def __init__(self):
        """Initialize the importance pooling layer."""
        super(ImportancePoolingLayer, self).__init__()
        
    def forward(self, x, neighbors, weights):
        """
        Forward pass for importance pooling.
        
        Args:
            x (torch.Tensor): Node feature matrix
            neighbors (list): List of lists containing neighbor indices for each node
            weights (list): List of lists containing importance weights for each neighbor
            
        Returns:
            pooled (torch.Tensor): Pooled neighborhood features
        """
        device = x.device
        pooled_list = []
        
        for i, (node_neighbors, node_weights) in enumerate(zip(neighbors, weights)):
            if not node_neighbors:
                # If no neighbors, use zeros
                pooled_list.append(torch.zeros_like(x[0]))
                continue
            
            # Ensure lists are not empty and contain valid indices
            node_neighbors = [n for n in node_neighbors if n < x.size(0)]
            if not node_neighbors:
                pooled_list.append(torch.zeros_like(x[0]))
                continue
                
            # Adjust weights if necessary
            node_weights = node_weights[:len(node_neighbors)]
            if sum(node_weights) == 0:
                node_weights = [1.0 / len(node_neighbors)] * len(node_neighbors)
            else:
                # Normalize weights
                norm = sum(node_weights)
                node_weights = [w / norm for w in node_weights]
            
            # Get neighbor features
            neighbor_feats = x[node_neighbors]
            
            # Convert weights to tensor
            weight_tensor = torch.tensor(node_weights, dtype=torch.float, device=device).view(-1, 1)
            
            # Compute weighted sum
            pooled = torch.sum(neighbor_feats * weight_tensor, dim=0)
            pooled_list.append(pooled)
        
        return torch.stack(pooled_list)

class WeightedMeanPoolingLayer(nn.Module):
    """
    Implements weighted mean pooling for aggregating neighborhood features.
    """
    def __init__(self):
        """Initialize the weighted mean pooling layer."""
        super(WeightedMeanPoolingLayer, self).__init__()
        
    def forward(self, x, neighbors, weights=None):
        """
        Forward pass for weighted mean pooling.
        
        Args:
            x (torch.Tensor): Node feature matrix
            neighbors (list): List of lists containing neighbor indices for each node
            weights (list, optional): List of lists containing weights for each neighbor
            
        Returns:
            pooled (torch.Tensor): Pooled neighborhood features
        """
        device = x.device
        pooled_list = []
        
        for i, node_neighbors in enumerate(neighbors):
            if not node_neighbors:
                # If no neighbors, use zeros
                pooled_list.append(torch.zeros_like(x[0]))
                continue
            
            # Ensure indices are valid
            node_neighbors = [n for n in node_neighbors if n < x.size(0)]
            if not node_neighbors:
                pooled_list.append(torch.zeros_like(x[0]))
                continue
                
            # Get neighbor features
            neighbor_feats = x[node_neighbors]
            
            # Apply weights if provided
            if weights is not None and len(weights) > i:
                node_weights = weights[i][:len(node_neighbors)]
                if sum(node_weights) == 0:
                    # If all weights are zero, use uniform weights
                    pooled = torch.mean(neighbor_feats, dim=0)
                else:
                    # Normalize weights
                    norm = sum(node_weights)
                    node_weights = [w / norm for w in node_weights]
                    
                    # Convert weights to tensor
                    weight_tensor = torch.tensor(node_weights, dtype=torch.float, device=device).view(-1, 1)
                    
                    # Compute weighted sum
                    pooled = torch.sum(neighbor_feats * weight_tensor, dim=0)
            else:
                # Simple mean pooling if no weights provided
                pooled = torch.mean(neighbor_feats, dim=0)
                
            pooled_list.append(pooled)
        
        return torch.stack(pooled_list)

class MaxPoolingLayer(nn.Module):
    """
    Implements max pooling for aggregating neighborhood features.
    """
    def __init__(self):
        """Initialize the max pooling layer."""
        super(MaxPoolingLayer, self).__init__()
        
    def forward(self, x, neighbors):
        """
        Forward pass for max pooling.
        
        Args:
            x (torch.Tensor): Node feature matrix
            neighbors (list): List of lists containing neighbor indices for each node
            
        Returns:
            pooled (torch.Tensor): Pooled neighborhood features
        """
        pooled_list = []
        
        for node_neighbors in neighbors:
            if not node_neighbors:
                # If no neighbors, use zeros
                pooled_list.append(torch.zeros_like(x[0]))
                continue
            
            # Ensure indices are valid
            node_neighbors = [n for n in node_neighbors if n < x.size(0)]
            if not node_neighbors:
                pooled_list.append(torch.zeros_like(x[0]))
                continue
                
            # Get neighbor features
            neighbor_feats = x[node_neighbors]
            
            # Compute max
            pooled, _ = torch.max(neighbor_feats, dim=0)
            pooled_list.append(pooled)
        
        return torch.stack(pooled_list)