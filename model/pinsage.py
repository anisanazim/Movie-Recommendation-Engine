import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_batch

class GraphConv(MessagePassing):
    """
    Graph convolutional layer for PinSage.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initialize the graph convolutional layer.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super(GraphConv, self).__init__(aggr='add')  # We'll use a weighted sum for aggregation
        
        # Transformation for node's own feature
        self.lin_self = nn.Linear(in_channels, out_channels)
        
        # Transformation for neighborhood features before aggregation
        self.lin_neigh = nn.Linear(in_channels, out_channels)
        
        # Final transformation after concatenation
        self.lin_update = nn.Linear(2 * out_channels, out_channels)
               
    def forward(self, x, edge_index=None, edge_weight=None, importance_weights=None):
        """
        Forward pass for the graph convolutional layer.
        
        Args:
            x (torch.Tensor): Node feature matrix with shape [num_nodes, in_channels]
            edge_index (torch.LongTensor, optional): Edge index with shape [2, num_edges]
            edge_weight (torch.FloatTensor, optional): Edge weights
            importance_weights (torch.FloatTensor, optional): Importance weights for neighborhood aggregation
            
        Returns:
            x_new (torch.Tensor): Updated node feature matrix with shape [num_nodes, out_channels]
        """
        # Transform node's own feature
        x_self = self.lin_self(x)
        
        # If edge_index is None, skip neighborhood aggregation
        if edge_index is None:
            # Just use transformed self features and zero for neighborhood
            x_neigh = torch.zeros_like(x_self)
        else:
            # Compute neighborhood aggregation
            x_neigh = self.propagate(edge_index, x=self.lin_neigh(x), edge_weight=edge_weight,
                                    importance_weights=importance_weights)
        
        # Concatenate own feature with neighborhood feature
        x_concat = torch.cat([x_self, x_neigh], dim=1)
        
        # Apply final transformation
        x_new = self.lin_update(x_concat)
        
        # Apply non-linearity
        x_new = F.relu(x_new)
        
        # Normalize
        x_new = F.normalize(x_new, p=2, dim=1)
        
        return x_new
    
    def message(self, x_j, edge_weight=None, importance_weights=None):
        """
        Message function for the graph convolutional layer.
        
        Args:
            x_j (torch.Tensor): Features of neighboring nodes
            edge_weight (torch.FloatTensor, optional): Edge weights
            importance_weights (torch.FloatTensor, optional): Importance weights
            
        Returns:
            msg (torch.Tensor): Message from neighboring nodes
        """
        msg = x_j
        
        # Apply edge weights if available
        if edge_weight is not None:
            msg = msg * edge_weight.view(-1, 1)
            
        # Apply importance weights if available
        if importance_weights is not None:
            msg = msg * importance_weights.view(-1, 1)
            
        return msg

class ImportancePooling(nn.Module):
    """
    Implements importance pooling for neighborhood aggregation.
    """
    def __init__(self):
        super(ImportancePooling, self).__init__()
        
    def forward(self, x, neighbors, weights):
        """
        Forward pass for importance pooling.
        """
        pooled = []
        max_idx = x.size(0) - 1
        
        for i, (node_neighbors, node_weights) in enumerate(zip(neighbors, weights)):
            # Handle single integer case
            if isinstance(node_neighbors, (int, np.integer)):
                node_neighbors = [node_neighbors]
                node_weights = [1.0]
                
            # Handle empty neighbors case
            if not node_neighbors:
                pooled.append(torch.zeros_like(x[0]))
                continue
                
            # Filter valid indices
            valid_indices = []
            valid_weights = []
            
            for j, idx in enumerate(node_neighbors):
                if isinstance(idx, (int, np.integer)) and idx <= max_idx:
                    valid_indices.append(idx)
                    if j < len(node_weights):
                        valid_weights.append(node_weights[j])
                    else:
                        valid_weights.append(1.0)
            
            # Handle no valid indices
            if not valid_indices:
                pooled.append(torch.zeros_like(x[0]))
                continue
                
            # Get neighbor features
            neighbor_features = x[valid_indices]
            
            # Convert weights to tensor and normalize
            weight_tensor = torch.tensor(valid_weights, device=x.device).view(-1, 1)
            weight_sum = weight_tensor.sum()
            if weight_sum > 0:
                weight_tensor = weight_tensor / weight_sum
            
            # Weighted sum
            weighted_sum = (neighbor_features * weight_tensor).sum(dim=0)
            
            pooled.append(weighted_sum)
            
        return torch.stack(pooled)
class PinSage(nn.Module):
    """
    PinSage model for learning embeddings of items in a graph.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        """
        Initialize the PinSage model.
        
        Args:
            in_channels (int): Number of input channels
            hidden_channels (int): Number of hidden channels
            out_channels (int): Number of output channels (embedding dimension)
            num_layers (int): Number of graph convolutional layers
        """
        super(PinSage, self).__init__()
        
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # Graph convolutional layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GraphConv(hidden_channels, hidden_channels))
            else:
                self.convs.append(GraphConv(hidden_channels, hidden_channels))
        
        # Importance pooling layer
        self.importance_pooling = ImportancePooling()
        
        # Output projection
        self.output_proj = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index=None, sampled_neighbors=None, importance_weights=None):
        """
        Forward pass for the PinSage model.

        Args:
            x (torch.Tensor): Node feature matrix of shape [num_nodes, input_dim].
            edge_index (torch.Tensor, optional): Edge indices for graph convolution.
            sampled_neighbors (list[list[torch.Tensor]], optional): 
                Sampled neighbors for each node at each layer.
            importance_weights (list[list[torch.Tensor]], optional): 
                Importance weights corresponding to sampled neighbors.

        Returns:
            torch.Tensor: Final normalized node embeddings of shape [num_nodes, embedding_dim].
        """
        # Project input features
        h = F.relu(self.input_proj(x))

        # Handle the case when graph structure is not provided
        if edge_index is None and (sampled_neighbors is None or importance_weights is None):
            # Simple MLP path without graph convolution
            for i in range(self.num_layers):
                # Apply a simple transformation
                h = F.relu(self.convs[i].lin_self(h))
            
            # Project to output dimension
            embeddings = self.output_proj(h)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings

        # If we have neighborhood information but no edge_index, use importance pooling
        if edge_index is None and sampled_neighbors is not None and importance_weights is not None:
            # Check if inputs are lists
            per_layer_neighbors = isinstance(sampled_neighbors, list) and isinstance(importance_weights, list)

            # Apply graph convolutional layers with importance pooling
            for i in range(self.num_layers):
                if per_layer_neighbors and len(sampled_neighbors) > i:
                    layer_neighbors = sampled_neighbors[i]
                    layer_weights = importance_weights[i]
                else:
                    # Use same neighbors and weights if not per-layer
                    layer_neighbors = sampled_neighbors
                    layer_weights = importance_weights

                # Pool neighborhood features
                h_neigh = self.importance_pooling(h, layer_neighbors, layer_weights)
                
                # Transform own features
                h_self = self.convs[i].lin_self(h)
                
                # Combine and update
                h_concat = torch.cat([h_self, h_neigh], dim=1)
                h = F.relu(self.convs[i].lin_update(h_concat))
                h = F.normalize(h, p=2, dim=1)
        
        # If we have edge_index, use standard graph convolution
        elif edge_index is not None:
            for i in range(self.num_layers):
                h = self.convs[i](h, edge_index)

        # Project to final embedding dimension
        embeddings = self.output_proj(h)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def get_embeddings(self, x, random_walk_sampler, num_neighbors=10):
        """
        Generate embeddings for a set of nodes.
        
        Args:
            x (torch.Tensor): Node feature matrix
            random_walk_sampler: RandomWalkSampler instance for neighborhood sampling
            num_neighbors (int): Number of neighbors to sample per node
            
        Returns:
            embeddings (torch.Tensor): Generated node embeddings
        """
        # Sample neighbors for each layer
        all_neighbors = []
        all_weights = []
        
        nodes = list(range(x.size(0)))
        
        for _ in range(self.num_layers):
            # Sample neighbors for this layer
            neighbors, weights = random_walk_sampler.batch_sample_neighbors(nodes, num_neighbors)
            all_neighbors.append(neighbors)
            all_weights.append(weights)
        
        # Generate embeddings using neighborhoods
        embeddings = self.forward(x, edge_index=None, sampled_neighbors=all_neighbors, importance_weights=all_weights)
        
        return embeddings