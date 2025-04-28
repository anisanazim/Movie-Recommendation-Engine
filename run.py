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
import pandas as pd
import copy
from tqdm import tqdm

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

def train_model(dataset, edge_index, edge_weights, movie_features, train_data, val_data, test_data, custom_config=None):
    """Train the PinSage model."""
    print("Training PinSage model...")
    
    # Use custom config if provided (for hyperparameter tuning)
    cfg = custom_config if custom_config is not None else config
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize RandomWalkSampler
    random_walk_sampler = RandomWalkSampler(
        edge_index, edge_weights,
        walk_length=cfg.WALK_LENGTH,
        num_walks=cfg.NUM_WALKS
    )
    
    # Initialize NegativeSampler
    negative_sampler = NegativeSampler(
        dataset, random_walk_sampler,
        num_negative_samples=cfg.NUM_NEGATIVE_SAMPLES
    )
    
    # Initialize PinSage model
    model = PinSage(
        in_channels=movie_features.shape[1],
        hidden_channels=cfg.HIDDEN_DIM,
        out_channels=cfg.EMBED_DIM,
        num_layers=cfg.NUM_LAYERS,
        # aggregator_type=cfg.AGGREGATOR_TYPE,
        # use_batch_norm=cfg.USE_BATCH_NORM
    ).to(device)
    
    # Print model summary
    print(f"PinSage model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    
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
        config=cfg
    )
    
    return model, checkpoint, random_walk_sampler

def evaluate_model_actual_interactions(model, movie_features, dataset, test_data, custom_config=None):
    """Evaluation using actual user-item interactions from test set."""
    print("Evaluating model with actual user-item interactions...")
    
    # Use custom config if provided (for hyperparameter tuning)
    cfg = custom_config if custom_config is not None else config
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate embeddings for all movies
    with torch.no_grad():
        all_movie_embeddings = []
        batch_size = 1024
        
        for i in range(0, len(dataset.movie_id_to_idx), batch_size):
            batch_indices = list(range(i, min(i + batch_size, len(dataset.movie_id_to_idx))))
            batch_features = movie_features[batch_indices].to(device)
            
            # Use simple MLP path
            batch_embeddings = model(batch_features, edge_index=None)
            all_movie_embeddings.append(batch_embeddings.cpu())
        
        all_movie_embeddings = torch.cat(all_movie_embeddings, dim=0)
    
    # Get positive pairs from test data
    if 'positive_pairs' in test_data and len(test_data['positive_pairs']) > 0:
        print("Checking existing test pairs...")
        test_pairs = test_data['positive_pairs']
        
        # Ensure all indices are valid
        max_idx = len(all_movie_embeddings) - 1
        valid_mask = (test_pairs[:, 0] <= max_idx) & (test_pairs[:, 1] <= max_idx)
        test_pairs = test_pairs[valid_mask]
        
        if len(test_pairs) == 0:
            print("Warning: No valid test pairs found in the provided test data.")
            print("Falling back to creating pairs from user-item interactions...")
        else:
            # Limit to a manageable subset for evaluation
            if len(test_pairs) > 5000:
                test_pairs = test_pairs[:5000]
    else:
        print("No existing test pairs found. Creating pairs from user-item interactions...")
        test_pairs = None
    
    # If no valid test pairs found or none existed, create from user interactions
    if test_pairs is None or len(test_pairs) == 0:
        # Get user-item interactions from test data
        test_interactions = dataset.ratings_df[dataset.ratings_df['rating'] >= 4.0]  # High-rated items
        test_interactions = test_interactions.sample(n=min(5000, len(test_interactions)), random_state=42)
        
        # Create test pairs
        new_test_pairs = []
        for _, row in test_interactions.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            
            # Skip if movie_id is not in our dataset
            if movie_id not in dataset.movie_id_to_idx:
                continue
                
            # Get all movies rated by this user
            user_movies = dataset.ratings_df[
                (dataset.ratings_df['userId'] == user_id) & 
                (dataset.ratings_df['rating'] >= 4.0)
            ]['movieId'].values
            
            # Get movie indices
            user_movie_indices = [dataset.movie_id_to_idx[mid] for mid in user_movies if mid in dataset.movie_id_to_idx]
            
            if len(user_movie_indices) >= 2:
                # Use one movie as query and another as positive example
                for i in range(len(user_movie_indices)):
                    for j in range(i+1, len(user_movie_indices)):
                        # Add both directions as valid pairs
                        new_test_pairs.append([user_movie_indices[i], user_movie_indices[j]])
                        new_test_pairs.append([user_movie_indices[j], user_movie_indices[i]])
        
        # Convert to tensor and limit size
        if new_test_pairs:
            test_pairs = torch.tensor(new_test_pairs)
            if len(test_pairs) > 5000:
                test_pairs = test_pairs[:5000]
        else:
            print("Warning: Still no valid test pairs found from user-item interactions!")
            # Fallback to genre similarity for evaluation
            return evaluate_model_genre_similarity(model, movie_features, dataset, custom_config)
    
    # Final check before evaluation
    if test_pairs is None or len(test_pairs) == 0:
        print("Error: No valid test pairs found for evaluation. Falling back to genre similarity.")
        return evaluate_model_genre_similarity(model, movie_features, dataset, custom_config)
    
    # Create test data dictionary
    test_data = {'positive_pairs': test_pairs}
    
    # Evaluate embeddings
    print(f"Evaluating embeddings with {len(test_pairs)} test pairs and k values: {cfg.K_VALUES}")
    results = evaluate_embeddings(all_movie_embeddings, test_data, k_values=cfg.K_VALUES)
    
    # Print results
    print("\nEvaluation Results:")
    print("-------------------")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    # Save embeddings
    torch.save(all_movie_embeddings, os.path.join(cfg.OUTPUT_DIR, 'movie_embeddings.pt'))
    
    return all_movie_embeddings, results

def evaluate_model_genre_similarity(model, movie_features, dataset, custom_config=None):
    """Fallback evaluation using genre similarity (original method)."""
    print("Evaluating model with genre similarity (fallback method)...")
    
    # Use custom config if provided (for hyperparameter tuning)
    cfg = custom_config if custom_config is not None else config
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate embeddings for all movies
    with torch.no_grad():
        all_movie_embeddings = []
        batch_size = 1024
        
        for i in range(0, len(dataset.movie_id_to_idx), batch_size):
            batch_indices = list(range(i, min(i + batch_size, len(dataset.movie_id_to_idx))))
            batch_features = movie_features[batch_indices].to(device)
            
            # Use simple MLP path
            batch_embeddings = model(batch_features, edge_index=None)
            all_movie_embeddings.append(batch_embeddings.cpu())
        
        all_movie_embeddings = torch.cat(all_movie_embeddings, dim=0)
    
    # Create test pairs based on genre similarity
    print("Creating test pairs based on genre similarity...")
    test_pairs = []
    
    # Get movies with genre info
    movies_with_genres = dataset.movies_df[dataset.movies_df['movieId'].isin(dataset.movie_id_to_idx.keys())]
    
    # Sample some movies for evaluation
    sample_size = min(1000, len(movies_with_genres))
    sampled_movies = movies_with_genres.sample(sample_size, random_state=42)
    
    for _, movie in sampled_movies.iterrows():
        movie_id = movie['movieId']
        movie_idx = dataset.movie_id_to_idx[movie_id]
        
        # Get genre for this movie
        genres = movie['genres'].split('|')
        
        # Find another movie with similar genres
        similar_movies = movies_with_genres[
            (movies_with_genres['movieId'] != movie_id) & 
            (movies_with_genres['genres'].apply(lambda x: any(g in x.split('|') for g in genres)))
        ]
        
        if not similar_movies.empty:
            similar_movie = similar_movies.sample(1).iloc[0]
            similar_idx = dataset.movie_id_to_idx[similar_movie['movieId']]
            
            # Add as a test pair
            test_pairs.append([movie_idx, similar_idx])
    
    test_pairs = torch.tensor(test_pairs)
    
    # Create test data dictionary
    test_data = {'positive_pairs': test_pairs}
    
    # Evaluate embeddings
    print(f"Evaluating embeddings with k values: {cfg.K_VALUES}")
    results = evaluate_embeddings(all_movie_embeddings, test_data, k_values=cfg.K_VALUES)
    
    # Print results
    print("\nEvaluation Results (Genre Similarity):")
    print("-------------------")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    # Save embeddings
    torch.save(all_movie_embeddings, os.path.join(cfg.OUTPUT_DIR, 'movie_embeddings.pt'))
    
    return all_movie_embeddings, results

def hyperparameter_tuning():
    """Run hyperparameter tuning for the PinSage model."""
    print("=" * 50)
    print("Starting Hyperparameter Tuning")
    print("=" * 50)
    
    # Process data first
    dataset, edge_index, edge_weights, movie_features, train_data, val_data, test_data = process_data()
    
    # Define parameter grid
    param_grid = {
        'learning_rate': [0.0001, 0.001, 0.01],
        'hidden_dim': [128, 256, 512],
        'embed_dim': [64, 128, 256],
        'num_layers': [1, 2, 3],
        'margin': [0.1, 0.3, 0.5],
        'num_walks': [50, 100, 200]
    }
    
    # Track best performance
    best_hit_rate = 0
    best_params = {}
    results_log = []
    
    # Simplified grid search - we'll start with just learning rate and hidden_dim
    # to make the tuning process faster
    for lr in param_grid['learning_rate']:
        for hidden in param_grid['hidden_dim']:
            # Create a copy of the config with new parameters
            tuned_config = copy.deepcopy(config)
            tuned_config.LEARNING_RATE = lr
            tuned_config.HIDDEN_DIM = hidden
            
            print(f"\nTesting parameters: lr={lr}, hidden_dim={hidden}")
            
            try:
                # Train model with these parameters
                model, checkpoint, random_walk_sampler = train_model(
                    dataset, edge_index, edge_weights, 
                    movie_features, train_data, val_data, test_data,
                    custom_config=tuned_config
                )
                
                # Evaluate model
                _, eval_results = evaluate_model_actual_interactions(
                    model, movie_features, dataset, test_data, custom_config=tuned_config
                )
                
                # Log results
                param_results = {
                    'learning_rate': lr,
                    'hidden_dim': hidden,
                    'embed_dim': tuned_config.EMBED_DIM,
                    'num_layers': tuned_config.NUM_LAYERS,
                    'margin': tuned_config.MARGIN,
                    'num_walks': tuned_config.NUM_WALKS,
                    'hit_rate@10': eval_results.get('hit_rate@10', 0),
                    'ndcg@10': eval_results.get('ndcg@10', 0),
                    'mrr': eval_results.get('mrr', 0)
                }
                results_log.append(param_results)
                
                # Track best performance
                hit_rate = eval_results.get('hit_rate@10', 0)
                if hit_rate > best_hit_rate:
                    best_hit_rate = hit_rate
                    best_params = {
                        'learning_rate': lr,
                        'hidden_dim': hidden,
                        'embed_dim': tuned_config.EMBED_DIM,
                        'num_layers': tuned_config.NUM_LAYERS,
                        'margin': tuned_config.MARGIN,
                        'num_walks': tuned_config.NUM_WALKS
                    }
                    
                    # Save best model
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'config': best_params,
                        'results': eval_results
                    }, os.path.join(tuned_config.CHECKPOINT_DIR, 'best_tuned_model.pt'))
                    
                    print(f"New best model saved! Hit rate@10: {hit_rate:.4f}")
                
            except Exception as e:
                print(f"Error during training with params lr={lr}, hidden={hidden}: {str(e)}")
                continue
    
    # Save all results
    if results_log:
        results_df = pd.DataFrame(results_log)
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        results_df.to_csv(os.path.join(config.OUTPUT_DIR, 'hyperparameter_tuning_results.csv'), index=False)
    
    print("=" * 50)
    print(f"Best parameters: {best_params}")
    print(f"Best hit rate@10: {best_hit_rate:.4f}")
    print("=" * 50)
    
    return best_params, best_hit_rate

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
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['train', 'evaluate', 'recommend', 'all', 'tune'],
                       help='Mode to run (train, evaluate, recommend, all, or tune)')
    parser.add_argument('--movie_id', type=int, default=None,
                       help='Movie ID to generate recommendations for')
    parser.add_argument('--num_recommendations', type=int, default=10,
                       help='Number of recommendations to generate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Hyperparameter tuning mode
    if args.mode == 'tune':
        best_params, best_hit_rate = hyperparameter_tuning()
        return
    
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
        model, checkpoint, _ = train_model(
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
    
    # Evaluate model with actual interactions
    if args.mode in ['evaluate', 'all']:
        all_movie_embeddings, results = evaluate_model_actual_interactions(
            model, movie_features, dataset, test_data
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