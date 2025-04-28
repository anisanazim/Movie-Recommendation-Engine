import torch
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA

class FeatureExtractor:
    """
    Extracts and processes features for movies in the MovieLens dataset.
    """
    def __init__(self, dataset):
        """
        Initialize the feature extractor.
        
        Args:
            dataset: MovieLensDataset instance containing the processed data
        """
        self.dataset = dataset
        
    def extract_movie_features(self, feature_dim=128):
        """
        Extract and process movie features.
        """
        print("Extracting movie features...")
        
        # Merge movie information with mappings
        merged_df = self.dataset.movies_df[self.dataset.movies_df['movieId'].isin(self.dataset.movie_id_to_idx.keys())]
        
        # Extract genre features (one-hot encoding)
        genre_features = self._extract_genre_features(merged_df)
        
        # Extract year features
        years_array = self._extract_year_features(merged_df)
        
        # Make sure years_array is 2D for proper stacking
        if len(years_array.shape) == 1:
            years_array = years_array.reshape(-1, 1)
        
        # Extract title features
        title_features = self._extract_title_features(merged_df)
        
        # Extract tag features if available
        tag_features = None
        if hasattr(self.dataset, 'tags_df') and self.dataset.tags_df is not None:
            tag_features = self._extract_tag_features(merged_df)
        
        # Combine features - ensure all are proper 2D arrays first
        features_list = []
        
        # Add genre features if valid
        if genre_features is not None and len(genre_features.shape) == 2:
            features_list.append(genre_features)
            
        # Add year features if valid
        if years_array is not None and len(years_array.shape) == 2:
            features_list.append(years_array)
            
        # Add title features if valid
        if title_features is not None and len(title_features.shape) == 2:
            features_list.append(title_features)
            
        # Add tag features if valid
        if tag_features is not None and len(tag_features.shape) == 2:
            features_list.append(tag_features)
        
        # Check if we have any features to combine
        if not features_list:
            raise ValueError("No valid features extracted to combine")
        
        # Combine all features
        combined_features = np.hstack(features_list)
        
        # Get the number of features and movies
        num_features = combined_features.shape[1]
        num_movies = len(self.dataset.movie_id_to_idx)
        
        # Create a mapping from movieId to feature row index
        movie_id_to_row = {movie_id: i for i, movie_id in enumerate(merged_df['movieId'])}
        
        # Reorder features by movie index
        features = np.zeros((num_movies, num_features))
        for movie_id, idx in self.dataset.movie_id_to_idx.items():
            if movie_id in movie_id_to_row:  # Check if movie_id exists in merged_df
                row_idx = movie_id_to_row[movie_id]
                if idx < num_movies and row_idx < len(combined_features):
                    features[idx] = combined_features[row_idx]
        
        # Apply dimensionality reduction if needed
        if feature_dim < features.shape[1]:
            print(f"Reducing feature dimension from {features.shape[1]} to {feature_dim} using PCA...")
            # Standardize features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
            
            # Apply PCA
            from sklearn.decomposition import PCA
            pca = PCA(n_components=feature_dim)
            features = pca.fit_transform(features)
            
            print(f"PCA explained variance ratio: {sum(pca.explained_variance_ratio_):.4f}")
        
        # Convert to tensor
        movie_features = torch.FloatTensor(features)
        
        print(f"Created movie feature matrix with shape {movie_features.shape}")
        
        return movie_features
    
    def _extract_genre_features(self, movies_df):
        """Extract one-hot encoded genre features with increased weight."""
        print("Extracting genre features...")
        
        # Extract genres using one-hot encoding
        genres = movies_df['genres'].str.get_dummies('|')
        
        genre_weight = 2.0
        weighted_genres = genres.values * genre_weight
        
        return weighted_genres
    
    def _extract_year_features(self, movies_df):
        """Extract year features from movie titles."""
        print("Extracting year features...")
        
        # Extract year from title
        movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)$')
        
        # Convert to numeric and handle missing values
        years = pd.to_numeric(movies_df['year'], errors='coerce').fillna(0)
        
        # Normalize years
        years_array = years.values.reshape(-1, 1)
        if years_array.max() > 0:
            years_array = years_array / 2020  # Normalize by a recent year
        
        return years_array
    
    def _extract_title_features(self, movies_df):
        """Extract features from movie titles using TF-IDF."""
        print("Extracting title features...")
        
        # Extract title without year
        titles = movies_df['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True)
        
        # Skip if all titles are empty
        if titles.str.strip().str.len().sum() == 0:
            return None
        
        # Apply TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=100,  # Limit number of features to avoid too high dimensionality
            min_df=5,          # Ignore terms that appear in less than 5 documents
            stop_words='english'
        )
        
        try:
            title_features = vectorizer.fit_transform(titles).toarray()
            return title_features
        except:
            print("Failed to extract title features. Skipping...")
            return None
    
    def _extract_tag_features(self, movies_df):
        """Extract features from user tags."""
        print("Extracting tag features...")
        
        tags_df = self.dataset.tags_df
        
        # Filter tags for movies in our dataset
        valid_tags = tags_df[tags_df['movieId'].isin(self.dataset.movie_id_to_idx.keys())]
        
        if len(valid_tags) == 0:
            return None
        
        # Convert any non-string tags to strings and filter out NaN values
        valid_tags['tag'] = valid_tags['tag'].astype(str)
        valid_tags = valid_tags[~valid_tags['tag'].str.contains('nan')]
        movie_tags = valid_tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
        
        # Merge with movies dataframe
        merged = movies_df.merge(movie_tags, on='movieId', how='left')
        merged['tag'] = merged['tag'].fillna('')
        
        # Apply TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=200,  # Limit number of features
            min_df=3,          # Ignore terms that appear in less than 3 documents
            stop_words='english'
        )
        
        try:
            tag_features = vectorizer.fit_transform(merged['tag']).toarray()
            return tag_features
        except:
            print("Failed to extract tag features. Skipping...")
            return None
    
    def create_visual_features(self, feature_dim=128):
        """
        Create dummy visual features for movies (in a real system, these would come from image models).
        
        Args:
            feature_dim (int): Dimension of visual features
            
        Returns:
            visual_features (torch.FloatTensor): Visual feature matrix
        """
        print("Creating dummy visual features...")
        
        # In a real system, these would be real visual features extracted from movie posters
        # Here we just create random features for demonstration
        num_movies = len(self.dataset.movie_id_to_idx)
        visual_features = np.random.randn(num_movies, feature_dim)
        
        # Normalize features
        norms = np.linalg.norm(visual_features, axis=1, keepdims=True)
        visual_features = visual_features / norms
        
        return torch.FloatTensor(visual_features)