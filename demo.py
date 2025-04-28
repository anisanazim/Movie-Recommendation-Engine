#!/usr/bin/env python
"""
Demo script for the PinSage recommendation system.
This provides a quick way to test the system with minimal setup.
"""

import os
import torch
import pandas as pd
from tqdm import tqdm
import argparse

from data.dataset import MovieLensDataset
from utils.random_walk import RandomWalkSampler
from model.pinsage import PinSage

def load_model_and_dataset(args):
    """Load the trained model and dataset."""
    print("Loading dataset and model...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = MovieLensDataset(args.data_dir, min_interactions=args.min_interactions)
    dataset.load_data()
    
    # Load movie embeddings if they exist
    embeddings_path = os.path.join(args.output_dir, 'movie_embeddings.pt')
    if os.path.exists(embeddings_path):
        print(f"Loading pre-computed embeddings from {embeddings_path}")
        all_movie_embeddings = torch.load(embeddings_path, map_location=device)
        model = None
    else:
        print("No pre-computed embeddings found. Loading model...")
        # Define model architecture
        model = PinSage(
            in_channels=args.feature_dim,
            hidden_channels=args.hidden_dim,
            out_channels=args.embed_dim,
            num_layers=args.num_layers,
            # dropout=args.dropout,
            # aggregator_type=args.aggregator_type,
            # use_batch_norm=args.use_batch_norm
        ).to(device)
        
        # Load model weights
        checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            all_movie_embeddings = None
        else:
            print(f"Checkpoint not found at {checkpoint_path}. Please train the model first.")
            return None, None, None
    
    return dataset, model, all_movie_embeddings

def find_movie_by_title(dataset, query):
    """Find a movie by title substring search."""
    # Case-insensitive search
    query = query.lower()
    
    # Search for movies containing the query
    matches = dataset.movies_df[dataset.movies_df['title'].str.lower().str.contains(query)]
    
    if len(matches) == 0:
        print(f"No movies found matching '{query}'")
        return None
    
    # If multiple matches, list them
    if len(matches) > 1:
        print(f"Found {len(matches)} matches for '{query}':")
        for i, (_, movie) in enumerate(matches.iterrows()):
            print(f"{i+1}. {movie['title']} ({movie['genres']})")
        
        # Ask user to select
        selection = input("Enter the number of the movie you want to select (or 'q' to quit): ")
        if selection.lower() == 'q':
            return None
        
        try:
            idx = int(selection) - 1
            if 0 <= idx < len(matches):
                return matches.iloc[idx]['movieId']
            else:
                print("Invalid selection.")
                return None
        except ValueError:
            print("Invalid input.")
            return None
    
    # If only one match, return it directly
    return matches.iloc[0]['movieId']

def generate_movie_recommendations(dataset, movie_id, all_movie_embeddings=None, model=None, k=10):
    """Generate recommendations for a specific movie."""
    if movie_id not in dataset.movie_id_to_idx:
        print(f"Movie ID {movie_id} not found in the dataset.")
        return None
        
    # Get movie index
    movie_idx = dataset.movie_id_to_idx[movie_id]
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # If embeddings are not provided, generate them using the model
    if all_movie_embeddings is None:
        if model is None:
            print("Either embeddings or model must be provided.")
            return None
            
        print("Generating embeddings for all movies...")
        # Extract movie features
        from data.feature_extractor import FeatureExtractor
        feature_extractor = FeatureExtractor(dataset)
        movie_features = feature_extractor.extract_movie_features()
        
        # Build graph
        from data.graph_builder import GraphBuilder
        graph_builder = GraphBuilder(dataset)
        edge_index, edge_weights = graph_builder.build_bipartite_graph()
        
        # Initialize random walk sampler
        random_walk_sampler = RandomWalkSampler(edge_index, edge_weights)
        
        # Generate embeddings
        with torch.no_grad():
            all_movie_embeddings = []
            batch_size = 1024
            
            for i in range(0, len(dataset.movie_id_to_idx), batch_size):
                batch_indices = list(range(i, min(i + batch_size, len(dataset.movie_id_to_idx))))
                batch_features = movie_features[batch_indices].to(device)
                
                # Sample neighbors
                batch_neighbors, batch_weights = random_walk_sampler.batch_sample_neighbors(
                    batch_indices, num_neighbors=50
                )
                
                # Get embeddings
                batch_embeddings = model(batch_features, [batch_neighbors], [batch_weights])
                all_movie_embeddings.append(batch_embeddings.cpu())
            
            all_movie_embeddings = torch.cat(all_movie_embeddings, dim=0)
    
    # Get query embedding
    query_embedding = all_movie_embeddings[movie_idx].unsqueeze(0)
    
    # Compute similarity with all movies
    similarities = torch.matmul(query_embedding, all_movie_embeddings.t()).squeeze()
    
    # Exclude the query movie
    similarities[movie_idx] = -float('inf')
    
    # Get top-k similar movies
    _, indices = torch.topk(similarities, k=k)
    recommendations = indices.numpy()
    
    return recommendations

def display_movie_details(dataset, movie_id):
    """Display details for a specific movie."""
    movie = dataset.movies_df[dataset.movies_df['movieId'] == movie_id]
    
    if movie.empty:
        print(f"Movie ID {movie_id} not found.")
        return
    
    print("\n" + "="*60)
    print(f"Movie: {movie['title'].values[0]}")
    print(f"Genres: {movie['genres'].values[0]}")
    
    # Check if the movie has tags
    if hasattr(dataset, 'tags_df') and dataset.tags_df is not None:
        movie_tags = dataset.tags_df[dataset.tags_df['movieId'] == movie_id]
        if not movie_tags.empty:
            print("\nTags:")
            top_tags = movie_tags['tag'].value_counts().head(10)
            for tag, count in top_tags.items():
                print(f"  • {tag} ({count})")
    
    # Check if the movie has ratings
    movie_ratings = dataset.ratings_df[dataset.ratings_df['movieId'] == movie_id]
    if not movie_ratings.empty:
        avg_rating = movie_ratings['rating'].mean()
        num_ratings = len(movie_ratings)
        print(f"\nAverage Rating: {avg_rating:.2f} (from {num_ratings} ratings)")
    
    print("="*60)

def interactive_demo(args):
    """Run an interactive demo of the recommendation system."""
    # Load model and dataset
    dataset, model, all_movie_embeddings = load_model_and_dataset(args)
    
    if dataset is None:
        return
    
    print("\n"+("="*60))
    print("🎬 PinSage Movie Recommendation Demo 🎬")
    print("="*60)
    
    while True:
        print("\nOptions:")
        print("1. Search for a movie by title")
        print("2. Get recommendations for a specific movie ID")
        print("3. Get popular movies")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == '1':
            # Search for a movie
            query = input("Enter part of a movie title to search: ")
            movie_id = find_movie_by_title(dataset, query)
            
            if movie_id is not None:
                # Display movie details
                display_movie_details(dataset, movie_id)
                
                # Ask if user wants recommendations
                rec_choice = input("Would you like to see recommendations for this movie? (y/n): ")
                if rec_choice.lower() == 'y':
                    num_recs = int(input("How many recommendations? (1-20): "))
                    num_recs = max(1, min(20, num_recs))
                    
                    recommendations = generate_movie_recommendations(
                        dataset, movie_id, all_movie_embeddings, model, k=num_recs
                    )
                    
                    if recommendations is not None:
                        print("\nRecommended Movies:")
                        for i, idx in enumerate(recommendations):
                            rec_movie_id = dataset.idx_to_movie_id[idx]
                            movie = dataset.movies_df[dataset.movies_df['movieId'] == rec_movie_id]
                            if not movie.empty:
                                print(f"{i+1}. {movie['title'].values[0]} - {movie['genres'].values[0]}")
        
        elif choice == '2':
            # Get recommendations for a specific movie ID
            try:
                movie_id = int(input("Enter a movie ID: "))
                
                # Display movie details
                display_movie_details(dataset, movie_id)
                
                num_recs = int(input("How many recommendations? (1-20): "))
                num_recs = max(1, min(20, num_recs))
                
                recommendations = generate_movie_recommendations(
                    dataset, movie_id, all_movie_embeddings, model, k=num_recs
                )
                
                if recommendations is not None:
                    print("\nRecommended Movies:")
                    for i, idx in enumerate(recommendations):
                        rec_movie_id = dataset.idx_to_movie_id[idx]
                        movie = dataset.movies_df[dataset.movies_df['movieId'] == rec_movie_id]
                        if not movie.empty:
                            print(f"{i+1}. {movie['title'].values[0]} - {movie['genres'].values[0]}")
            
            except ValueError:
                print("Invalid movie ID. Please enter a numeric ID.")
        
        elif choice == '3':
            # Get popular movies
            print("\nMost Popular Movies (by number of ratings):")
            movie_counts = dataset.ratings_df['movieId'].value_counts().head(20)
            
            for i, (movie_id, count) in enumerate(movie_counts.items()):
                movie = dataset.movies_df[dataset.movies_df['movieId'] == movie_id]
                if not movie.empty:
                    print(f"{i+1}. {movie['title'].values[0]} - {count} ratings")
        
        elif choice == '4':
            # Exit
            print("Thank you for using the demo!")
            break
        
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

def main():
    parser = argparse.ArgumentParser(description='PinSage Demo')
    parser.add_argument('--data_dir', type=str, default='./data/ml-25m',
                      help='Path to MovieLens dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                      help='Directory for model checkpoints')
    parser.add_argument('--output_dir', type=str, default='./output',
                      help='Directory for outputs')
    parser.add_argument('--min_interactions', type=int, default=5,
                      help='Minimum number of interactions per user')
    
    # Model parameters (only needed if generating embeddings on-the-fly)
    parser.add_argument('--feature_dim', type=int, default=128,
                      help='Dimension of input features')
    parser.add_argument('--hidden_dim', type=int, default=256,
                      help='Dimension of hidden layers')
    parser.add_argument('--embed_dim', type=int, default=128,
                      help='Dimension of final embeddings')
    parser.add_argument('--num_layers', type=int, default=2,
                      help='Number of graph convolutional layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                      help='Dropout rate')
    parser.add_argument('--aggregator_type', type=str, default='importance',
                      help='Type of aggregator to use')
    parser.add_argument('--use_batch_norm', action='store_true',
                      help='Use batch normalization')
    parser.add_argument('--cpu', action='store_true',
                      help='Force CPU usage (even if GPU is available)')
    
    args = parser.parse_args()
    
    interactive_demo(args)

if __name__ == '__main__':
    main()