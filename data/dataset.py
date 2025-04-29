import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

class MovieLensDataset:
    """
    Processing the MovieLens 25M dataset for graph-based recommendation.
    """
    def __init__(self, data_dir, min_interactions=5):
        """
        Initialize the dataset.
        
        Args:
            data_dir (str): Directory containing the MovieLens 25M dataset files
            min_interactions (int): Minimum number of interactions a user must have
        """
        self.data_dir = data_dir
        self.min_interactions = min_interactions
        
        # Load raw data
        self.movies_df = None
        self.ratings_df = None
        self.tags_df = None
        self.links_df = None
        
        # Processed data
        self.movie_features = None
        self.edge_index = None
        self.edge_weights = None
        
        # Maps
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}
        self.user_id_to_idx = {}
        self.idx_to_user_id = {}
        
    def load_data(self):
        """Load the MovieLens 25M dataset files."""
        print("Loading MovieLens 25M dataset...")
        
        # Load movies
        movies_path = os.path.join(self.data_dir, 'movies.csv')
        self.movies_df = pd.read_csv(movies_path)
        print(f"Loaded {len(self.movies_df)} movies")
        
        # Load ratings
        ratings_path = os.path.join(self.data_dir, 'ratings.csv')
        self.ratings_df = pd.read_csv(ratings_path)
        print(f"Loaded {len(self.ratings_df)} ratings")
        
        # Filter users with too few interactions
        user_counts = self.ratings_df['userId'].value_counts()
        valid_users = user_counts[user_counts >= self.min_interactions].index
        self.ratings_df = self.ratings_df[self.ratings_df['userId'].isin(valid_users)]
        print(f"After filtering, {len(self.ratings_df)} ratings remain")
        
        # Load tags (optional)
        tags_path = os.path.join(self.data_dir, 'tags.csv')
        if os.path.exists(tags_path):
            self.tags_df = pd.read_csv(tags_path)
            print(f"Loaded {len(self.tags_df)} tags")
        
        # Load links (optional)
        links_path = os.path.join(self.data_dir, 'links.csv')
        if os.path.exists(links_path):
            self.links_df = pd.read_csv(links_path)
        
        # Create ID to index mappings
        self._create_mappings()
        
        return self
    
    def _create_mappings(self):
        """Create mappings between original IDs and continuous indices."""
        # Create movie ID mapping
        unique_movie_ids = self.ratings_df['movieId'].unique()
        self.movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(unique_movie_ids)}
        self.idx_to_movie_id = {idx: movie_id for movie_id, idx in self.movie_id_to_idx.items()}
        
        # Create user ID mapping
        unique_user_ids = self.ratings_df['userId'].unique()
        self.user_id_to_idx = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
        self.idx_to_user_id = {idx: user_id for user_id, idx in self.user_id_to_idx.items()}
        
        print(f"Created mappings for {len(self.movie_id_to_idx)} movies and {len(self.user_id_to_idx)} users")
        
    def build_graph(self):
        """
        Build a bipartite graph from user-item interactions.
        
        Returns:
            edge_index (torch.LongTensor): Edge index with shape [2, num_edges]
            edge_weights (torch.FloatTensor): Edge weights corresponding to the ratings
        """
        print("Building interaction graph...")
        
        # Convert user and movie IDs to indices
        user_indices = torch.tensor([self.user_id_to_idx[user_id] for user_id in self.ratings_df['userId']])
        movie_indices = torch.tensor([self.movie_id_to_idx[movie_id] for movie_id in self.ratings_df['movieId']])
        
        # Add an offset to user indices to avoid index overlap with movie indices
        user_indices = user_indices + len(self.movie_id_to_idx)
        
        # Create a bidirectional edge index (user->movie and movie->user)
        edge_index = torch.stack([
            torch.cat([user_indices, movie_indices]),
            torch.cat([movie_indices, user_indices])
        ], dim=0)
        
        # Get edge weights from ratings
        ratings = torch.tensor(self.ratings_df['rating'].values, dtype=torch.float)
        edge_weights = torch.cat([ratings, ratings])
        
        self.edge_index = edge_index
        self.edge_weights = edge_weights
        
        print(f"Created graph with {len(self.ratings_df)} interactions (bidirectional)")
        
        return edge_index, edge_weights
    
    def extract_movie_features(self, feature_dim=128):
        """
        Extract and process movie features.
        
        Args:
            feature_dim (int): Dimension to reduce the feature vectors to
            
        Returns:
            movie_features (torch.FloatTensor): Feature matrix for movies with shape [num_movies, feature_dim]
        """
        print("Extracting movie features...")
        
        # Merge movie information with mappings
        merged_df = self.movies_df[self.movies_df['movieId'].isin(self.movie_id_to_idx.keys())]
        
        # Extract genre features (one-hot encoding)
        genres = merged_df['genres'].str.get_dummies('|')
        
        # Extract year from title (if available)
        merged_df['year'] = merged_df['title'].str.extract(r'\((\d{4})\)$')
        years = pd.get_dummies(merged_df['year'], prefix='year')
        
        # Combine features
        features_df = pd.concat([genres, years], axis=1).fillna(0)
        
        # Reorder by movie index
        features = np.zeros((len(self.movie_id_to_idx), features_df.shape[1]))
        for movie_id, idx in self.movie_id_to_idx.items():
            if movie_id in merged_df['movieId'].values:
                movie_row = merged_df[merged_df['movieId'] == movie_id].index[0]
                features[idx] = features_df.iloc[movie_row].values
        
        # Convert to tensor
        movie_features = torch.FloatTensor(features)
        
        # Dimensionality reduction if needed (simple approach here)
        if feature_dim < movie_features.shape[1]:
            # You could use PCA here, but for simplicity, we'll use a linear projection
            projection = torch.nn.Linear(movie_features.shape[1], feature_dim)
            movie_features = projection(movie_features)
        
        self.movie_features = movie_features
        
        print(f"Created movie feature matrix with shape {movie_features.shape}")
        
        return movie_features
    
    def get_train_val_test_split(self, val_ratio=0.1, test_ratio=0.2):
        """
        Split the dataset into train/validation/test sets.
        
        Args:
            val_ratio (float): Ratio of validation data
            test_ratio (float): Ratio of test data
            
        Returns:
            train_data (dict): Training data
            val_data (dict): Validation data
            test_data (dict): Test data
        """
        # Get all user-item interactions
        interactions = self.ratings_df[['userId', 'movieId']].copy()
        
        # Convert IDs to indices
        user_indices = []
        movie_indices = []
        
        for _, row in interactions.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            
            if user_id in self.user_id_to_idx and movie_id in self.movie_id_to_idx:
                user_indices.append(self.user_id_to_idx[user_id])
                movie_indices.append(self.movie_id_to_idx[movie_id])
        
        # Get the total number of samples
        n_samples = len(user_indices)
        
        # Shuffle indices
        indices = np.random.permutation(n_samples)
        
        # Calculate split sizes
        n_test = int(test_ratio * n_samples)
        n_val = int(val_ratio * n_samples)
        n_train = n_samples - n_test - n_val
        
        # Split indices
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train+n_val]
        test_indices = indices[n_train+n_val:]
        
        # Create positive pairs for each split
        train_pos_pairs = torch.tensor([[user_indices[i], movie_indices[i]] for i in train_indices])
        val_pos_pairs = torch.tensor([[user_indices[i], movie_indices[i]] for i in val_indices])
        test_pos_pairs = torch.tensor([[user_indices[i], movie_indices[i]] for i in test_indices])
        
        # Get maximum valid index (ensuring all indices are within bounds)
        max_movie_idx = len(self.movie_id_to_idx) - 1
        max_user_idx = len(self.user_id_to_idx) - 1
        
        # Filter out pairs with out-of-bounds indices
        train_valid_mask = (train_pos_pairs[:, 0] <= max_user_idx) & (train_pos_pairs[:, 1] <= max_movie_idx)
        val_valid_mask = (val_pos_pairs[:, 0] <= max_user_idx) & (val_pos_pairs[:, 1] <= max_movie_idx)
        test_valid_mask = (test_pos_pairs[:, 0] <= max_user_idx) & (test_pos_pairs[:, 1] <= max_movie_idx)
        
        train_pos_pairs = train_pos_pairs[train_valid_mask]
        val_pos_pairs = val_pos_pairs[val_valid_mask]
        test_pos_pairs = test_pos_pairs[test_valid_mask]
        
        # Create data dictionaries
        train_data = {'positive_pairs': train_pos_pairs}
        val_data = {'positive_pairs': val_pos_pairs}
        test_data = {'positive_pairs': test_pos_pairs}
        
        print(f"Data split: {len(train_pos_pairs)} train, {len(val_pos_pairs)} validation, {len(test_pos_pairs)} test pairs")
        
        return train_data, val_data, test_data
          
    def _create_split_data(self, ratings_df):
        """Create a PyTorch Geometric Data object for a split."""
        # Convert user and movie IDs to indices
        user_indices = torch.tensor([self.user_id_to_idx[user_id] for user_id in ratings_df['userId']])
        movie_indices = torch.tensor([self.movie_id_to_idx[movie_id] for movie_id in ratings_df['movieId']])
        
        # Add an offset to user indices
        user_indices = user_indices + len(self.movie_id_to_idx)
        
        # Create a bidirectional edge index
        edge_index = torch.stack([
            torch.cat([user_indices, movie_indices]),
            torch.cat([movie_indices, user_indices])
        ], dim=0)
        
        # Get edge weights from ratings
        ratings = torch.tensor(ratings_df['rating'].values, dtype=torch.float)
        edge_weights = torch.cat([ratings, ratings])
        
        # Create positive pairs for training
        positive_pairs = torch.stack([user_indices, movie_indices], dim=1)
        
        # Create Data object
        data = {
            'edge_index': edge_index,
            'edge_weights': edge_weights,
            'positive_pairs': positive_pairs
        }
        
        return data
    
    def create_pytorch_geometric_data(self):
        """
        Create a PyTorch Geometric Data object from the processed data.
        
        Returns:
            data (torch_geometric.data.Data): Data object for PyTorch Geometric
        """
        # Create node feature matrix
        # For users, we'll just use a simple embedding
        num_users = len(self.user_id_to_idx)
        user_features = torch.zeros(num_users, self.movie_features.shape[1])
        
        # Combine movie and user features
        x = torch.cat([self.movie_features, user_features], dim=0)
        
        # Create Data object
        data = Data(
            x=x,
            edge_index=self.edge_index,
            edge_attr=self.edge_weights
        )
        
        return data