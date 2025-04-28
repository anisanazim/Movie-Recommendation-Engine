#!/usr/bin/env python
"""
Run script for the PinSage recommendation system.
This script allows running the full pipeline from data processing to evaluation.
"""

import os
import argparse
import torch
import random
import numpy as np

from data.dataset import MovieLensDataset
from data.graph_builder import GraphBuilder
from data.feature_extractor import FeatureExtractor
from data.negative_sampler import NegativeSampler
from model.pinsage import PinSage
from utils.random_walk import RandomWalkSampler
from utils.evaluation import evaluate_embeddings, generate_recommendations
from utils.nearest_neighbors import LSHIndex, WeakANDIndex, benchmark_search_methods

import config

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def process_data():
    """Process the MovieLens dataset."""
    print("Processing MovieLens dataset...")
    
    # Load and process the dataset
    dataset = MovieLensDataset(config.DATA_DIR, min_interactions=config.MIN_INTERACTIONS)
    dataset.load_data()

    # Use a subset of the data to make training more manageable
    print("Using a subset of data for testing...")
    dataset.ratings_df = dataset.ratings_df.sample(frac=0.30, random_state=42)  # Use 30% of ratings

    # # In process_data function in run.py, after loading the dataset
    # if hasattr(config, 'USE_DATA_SUBSET') and config.USE_DATA_SUBSET:
    #     print(f"Using a subset of data ({config.DATA_SUBSET_FRACTION:.20%})...")
    #     dataset.ratings_df = dataset.ratings_df.sample(frac=config.DATA_SUBSET_FRACTION, random_state=42)
    #     print(f"Reduced dataset to {len(dataset.ratings_df)} ratings")
    
    # Build the graph
    graph_builder = GraphBuilder(dataset)
    if config.USE_BIPARTITE_GRAPH:
        edge_index, edge_weights = graph_builder.build_bipartite_graph()
    else:
        edge_index, edge_weights = graph_builder.build_item_similarity_graph(
            threshold=config.SIMILARITY_THRESHOLD
        )
    
    # Extract features
    feature_extractor = FeatureExtractor(dataset)
    movie_features = feature_extractor.extract_movie_features(feature_dim=config.FEATURE_DIM)
    
    # Add visual features if enabled
    if config.USE_VISUAL_FEATURES:
        visual_features = feature_extractor.create_visual_features(feature_dim=config.FEATURE_DIM)
        # Combine content and visual features (simple concatenation for now)
        combined_features = torch.cat([movie_features, visual_features], dim=1)
        # Project back to the desired dimension
        projection = torch.nn.Linear(combined_features.shape[1], config.FEATURE_DIM)
        movie_features = projection(combined_features)
    
    # Create train/val/test splits
    train_data, val_data, test_data = dataset.get_train_val_test_split(
        val_ratio=config.VAL_RATIO, test_ratio=config.TEST_RATIO
    )
    
    return dataset, edge_index, edge_weights, movie_features, train_data, val_data, test_data

def train_model(dataset, edge_index, edge_weights, movie_features, train_data, val_data, test_data):
    """Train the PinSage model."""
    print("Training PinSage model...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize RandomWalkSampler
    random_walk_sampler = RandomWalkSampler(
        edge_index, edge_weights,
        walk_length=config.WALK_LENGTH,
        num_walks=config.NUM_WALKS
    )
    
    # Initialize NegativeSampler
    negative_sampler = NegativeSampler(
        dataset, random_walk_sampler,
        num_negative_samples=config.NUM_NEGATIVE_SAMPLES
    )
    
    # Initialize PinSage model
    model = PinSage(
        in_channels=movie_features.shape[1],
        hidden_channels=config.HIDDEN_DIM,
        out_channels=config.EMBED_DIM,
        num_layers=config.NUM_LAYERS,
        # aggregator_type=config.AGGREGATOR_TYPE,
        # use_batch_norm=config.USE_BATCH_NORM
    ).to(device)
    
    # Print model summary
    print(f"PinSage model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Set up learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    # Train the model using the train script
    from train import train
    checkpoint = train(
        model=model,
        movie_features=movie_features,
        train_data=train_data,
        val_data=val_data,
        random_walk_sampler=random_walk_sampler,
        negative_sampler=negative_sampler,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        dataset=dataset,
        config=config
    )
    
    return model, checkpoint

def evaluate_model(model, movie_features, test_data, random_walk_sampler, dataset):
    """Evaluate the trained model."""
    print("Evaluating model...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate embeddings for all movies
    with torch.no_grad():
        all_movie_embeddings = []
        batch_size = 1024
        
        for i in range(0, len(dataset.movie_id_to_idx), batch_size):
            batch_indices = list(range(i, min(i + batch_size, len(dataset.movie_id_to_idx))))
            batch_features = movie_features[batch_indices].to(device)
            
            # Sample neighbors
            batch_neighbors, batch_weights = random_walk_sampler.batch_sample_neighbors(
                batch_indices, num_neighbors=config.NUM_NEIGHBORS
            )
            
            # Get embeddings
            batch_embeddings = model(
                batch_features, 
                edge_index=None,
                sampled_neighbors=batch_neighbors, 
                importance_weights=batch_weights
            )
            all_movie_embeddings.append(batch_embeddings.cpu())
        
        all_movie_embeddings = torch.cat(all_movie_embeddings, dim=0)
    
    # Evaluate embeddings
    print(f"Evaluating embeddings with k values: {config.K_VALUES}")
    results = evaluate_embeddings(all_movie_embeddings, test_data, k_values=config.K_VALUES)
    
    # Print results
    print("\nEvaluation Results:")
    print("-------------------")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    # Save embeddings
    torch.save(all_movie_embeddings, os.path.join(config.OUTPUT_DIR, 'movie_embeddings.pt'))
    
    return all_movie_embeddings, results

def generate_movie_recommendations(movie_id, num_recommendations, model, movie_features, dataset, random_walk_sampler):
    """Generate recommendations for a specific movie."""
    print(f"Generating {num_recommendations} recommendations for movie ID {movie_id}...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if movie ID exists
    if movie_id not in dataset.movie_id_to_idx:
        print(f"Movie ID {movie_id} not found in the dataset.")
        return None
    
    # Get movie index
    movie_idx = dataset.movie_id_to_idx[movie_id]
    
    # Generate embeddings for all movies
    with torch.no_grad():
        all_movie_embeddings = []
        batch_size = 1024
        
        for i in range(0, len(dataset.movie_id_to_idx), batch_size):
            batch_indices = list(range(i, min(i + batch_size, len(dataset.movie_id_to_idx))))
            batch_features = movie_features[batch_indices].to(device)
            
            # Sample neighbors
            batch_neighbors, batch_weights = random_walk_sampler.batch_sample_neighbors(
                batch_indices, num_neighbors=config.NUM_NEIGHBORS
            )
            
            # Get embeddings
            batch_embeddings = model(batch_features, edge_index=None)
            all_movie_embeddings.append(batch_embeddings.cpu())
        
        all_movie_embeddings = torch.cat(all_movie_embeddings, dim=0)
    
    # Generate recommendations
    recommendations = generate_recommendations(
        all_movie_embeddings, movie_idx, 
        k=num_recommendations, 
        exclude_query=True
    )
    
    # Get movie details
    query_movie = dataset.movies_df[dataset.movies_df['movieId'] == movie_id]
    
    # Print recommendations
    print(f"\nRecommendations for: {query_movie['title'].values[0]}")
    print("---------------------------------------------")
    
    recommended_movies = []
    for i, idx in enumerate(recommendations):
        rec_movie_id = dataset.idx_to_movie_id[idx]
        movie = dataset.movies_df[dataset.movies_df['movieId'] == rec_movie_id]
        
        if not movie.empty:
            title = movie['title'].values[0]
            genres = movie['genres'].values[0]
            print(f"{i+1}. {title} - {genres}")
            
            recommended_movies.append({
                'rank': i+1,
                'movieId': int(rec_movie_id),
                'title': title,
                'genres': genres
            })
    
    return recommended_movies

def main():
    parser = argparse.ArgumentParser(description='PinSage Recommendation System')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate', 'recommend', 'all'],
                       help='Mode to run (train, evaluate, recommend, or all)')
    parser.add_argument('--movie_id', type=int, default=None,
                       help='Movie ID to generate recommendations for')
    parser.add_argument('--num_recommendations', type=int, default=10,
                       help='Number of recommendations to generate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Process data
    dataset, edge_index, edge_weights, movie_features, train_data, val_data, test_data = process_data()
    
    # Initialize RandomWalkSampler
    random_walk_sampler = RandomWalkSampler(
        edge_index, edge_weights,
        walk_length=config.WALK_LENGTH,
        num_walks=config.NUM_WALKS
    )

    # Ensure all indices in training data are valid movie indices
    train_positive_pairs = train_data['positive_pairs']
    # Get only movie indices (second column in the positive pairs)
    train_movie_indices = train_positive_pairs[:, 1]
    # Filter out any movie indices that are out of bounds
    valid_mask = train_movie_indices < len(movie_features)
    valid_pairs = train_positive_pairs[valid_mask]
    # Update the training data
    train_data['positive_pairs'] = valid_pairs

    val_positive_pairs = val_data['positive_pairs']
    val_movie_indices = val_positive_pairs[:, 1]
    valid_val_mask = val_movie_indices < len(movie_features)
    valid_val_pairs = val_positive_pairs[valid_val_mask]
    val_data['positive_pairs'] = valid_val_pairs

    # Train model
    if args.mode in ['train', 'all']:
        model, checkpoint = train_model(
            dataset, edge_index, edge_weights, 
            movie_features, train_data, val_data, test_data
        )
    else:
        # Load trained model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = PinSage(
            in_channels=movie_features.shape[1],
            hidden_channels=config.HIDDEN_DIM,
            out_channels=config.EMBED_DIM,
            num_layers=config.NUM_LAYERS,
            # dropout=config.DROPOUT,
            # aggregator_type=config.AGGREGATOR_TYPE,
            # use_batch_norm=config.USE_BATCH_NORM
        ).to(device)
        
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pt')
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print(f"Checkpoint not found at {checkpoint_path}. Please train the model first.")
            return
    
    # Evaluate model
    if args.mode in ['evaluate', 'all']:
        all_movie_embeddings, results = evaluate_model(
            model, movie_features, test_data, 
            random_walk_sampler, dataset
        )
    
    # Generate recommendations
    if args.mode in ['recommend', 'all']:
        if args.movie_id is None:
            print("Please provide a movie ID with --movie_id")
            return
        
        recommendations = generate_movie_recommendations(
            args.movie_id, args.num_recommendations,
            model, movie_features, dataset, random_walk_sampler
        )

if __name__ == '__main__':
    main()