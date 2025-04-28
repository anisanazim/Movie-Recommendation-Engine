import argparse
import os
import torch
import numpy as np
from data.dataset import MovieLensDataset
from model.pinsage import PinSage
from utils.random_walk import RandomWalkSampler
from utils.evaluation import evaluate_embeddings, generate_recommendations
from utils.nearest_neighbors import LSHIndex, WeakANDIndex, benchmark_search_methods

def main():
    parser = argparse.ArgumentParser(description='PinSage for MovieLens')
    
    # Mode selection
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate', 'inference'],
                      help='Mode to run (train, evaluate, or inference)')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./data/ml-25m',
                      help='Path to MovieLens dataset')
    parser.add_argument('--min_interactions', type=int, default=5,
                      help='Minimum number of interactions per user')
    
    # Model parameters
    parser.add_argument('--feature_dim', type=int, default=128,
                      help='Dimension of input features')
    parser.add_argument('--hidden_dim', type=int, default=256,
                      help='Dimension of hidden layers')
    parser.add_argument('--embed_dim', type=int, default=128,
                      help='Dimension of final embeddings')
    parser.add_argument('--num_layers', type=int, default=2,
                      help='Number of graph convolutional layers')
    parser.add_argument('--num_neighbors', type=int, default=50,
                      help='Number of neighbors to sample per node')
    
    # Random walk parameters
    parser.add_argument('--walk_length', type=int, default=2,
                      help='Length of random walks')
    parser.add_argument('--num_walks', type=int, default=100,
                      help='Number of random walks per node')
    
    # Training parameters
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                      help='Directory for model checkpoints')
    parser.add_argument('--checkpoint', type=str, default='best_model.pt',
                      help='Checkpoint file to load')
    
    # Evaluation parameters
    parser.add_argument('--k_values', type=str, default='10,50,100,500',
                      help='Comma-separated list of k values for hit rate calculation')
    
    # Inference parameters
    parser.add_argument('--movie_id', type=int, default=None,
                      help='Movie ID to generate recommendations for')
    parser.add_argument('--num_recommendations', type=int, default=10,
                      help='Number of recommendations to generate')
    parser.add_argument('--search_method', type=str, default='exact', choices=['exact', 'lsh', 'ivf'],
                      help='Search method for nearest neighbor lookup')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and process dataset
    print("Loading and processing dataset...")
    dataset = MovieLensDataset(args.data_dir, min_interactions=args.min_interactions)
    dataset.load_data()
    
    # Build graph and extract features
    edge_index, edge_weights = dataset.build_graph()
    movie_features = dataset.extract_movie_features(feature_dim=args.feature_dim)
    
    # Initialize RandomWalkSampler
    random_walk_sampler = RandomWalkSampler(
        edge_index, edge_weights, 
        walk_length=args.walk_length, 
        num_walks=args.num_walks
    )
    
    # Initialize model
    model = PinSage(
        in_channels=movie_features.shape[1],
        hidden_channels=args.hidden_dim,
        out_channels=args.embed_dim,
        num_layers=args.num_layers
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint)
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        if args.mode != 'train':
            print(f"Checkpoint not found at {checkpoint_path}. Please train the model first.")
            return
    
    if args.mode == 'train':
        # Run training
        from train import train
        train_args = argparse.Namespace(
            data_dir=args.data_dir,
            min_interactions=args.min_interactions,
            val_ratio=0.1,
            test_ratio=0.2,
            feature_dim=args.feature_dim,
            hidden_dim=args.hidden_dim,
            embed_dim=args.embed_dim,
            num_layers=args.num_layers,
            num_neighbors=args.num_neighbors,
            walk_length=args.walk_length,
            num_walks=args.num_walks,
            batch_size=512,
            epochs=10,
            lr=0.001,
            margin=0.1,
            num_negative_samples=500,
            num_workers=4,
            seed=42,
            no_cuda=not torch.cuda.is_available(),
            k=500,
            eval_every=1,
            patience=3,
            checkpoint_dir=args.checkpoint_dir
        )
        train(train_args)
        
    elif args.mode == 'evaluate':
        # Run evaluation
        model.eval()
        
        # Get train/val/test splits
        train_data, val_data, test_data = dataset.get_train_val_test_split(
            val_ratio=0.1, test_ratio=0.2
        )
        
        # Generate embeddings for all movies
        print("Generating embeddings for all movies...")
        with torch.no_grad():
            all_movie_embeddings = []
            batch_size = 1024
            
            for i in range(0, len(dataset.movie_id_to_idx), batch_size):
                batch_indices = list(range(i, min(i + batch_size, len(dataset.movie_id_to_idx))))
                batch_features = movie_features[batch_indices].to(device)
                
                # Sample neighbors
                batch_neighbors, batch_weights = random_walk_sampler.batch_sample_neighbors(
                    batch_indices, num_neighbors=args.num_neighbors
                )
                
                # Get embeddings
                batch_embeddings = model(batch_features, [batch_neighbors], [batch_weights])
                all_movie_embeddings.append(batch_embeddings.cpu())
            
            all_movie_embeddings = torch.cat(all_movie_embeddings, dim=0)
        
        # Evaluate embeddings
        k_values = [int(k) for k in args.k_values.split(',')]
        print(f"Evaluating embeddings with k values: {k_values}")
        
        results = evaluate_embeddings(all_movie_embeddings, test_data, k_values=k_values)
        
        # Print results
        print("\nEvaluation Results:")
        print("-------------------")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
        
        # Benchmark search methods
        print("\nBenchmarking search methods...")
        query_indices = test_data['positive_pairs'][:100, 0].numpy() - len(dataset.movie_id_to_idx)
        query_embeddings = all_movie_embeddings[query_indices]
        
        benchmark_search_methods(
            all_movie_embeddings, 
            query_embeddings, 
            k=10, 
            methods=['exact', 'lsh', 'ivf']
        )
        
    elif args.mode == 'inference':
        # Run inference (generate recommendations)
        if args.movie_id is None:
            print("Please provide a movie ID with --movie_id")
            return
        
        # Check if movie ID exists
        if args.movie_id not in dataset.movie_id_to_idx:
            print(f"Movie ID {args.movie_id} not found in the dataset.")
            return
        
        # Get movie index
        movie_idx = dataset.movie_id_to_idx[args.movie_id]
        
        # Generate embeddings for all movies
        print("Generating embeddings for all movies...")
        with torch.no_grad():
            all_movie_embeddings = []
            batch_size = 1024
            
            for i in range(0, len(dataset.movie_id_to_idx), batch_size):
                batch_indices = list(range(i, min(i + batch_size, len(dataset.movie_id_to_idx))))
                batch_features = movie_features[batch_indices].to(device)
                
                # Sample neighbors
                batch_neighbors, batch_weights = random_walk_sampler.batch_sample_neighbors(
                    batch_indices, num_neighbors=args.num_neighbors
                )
                
                # Get embeddings
                batch_embeddings = model(batch_features, [batch_neighbors], [batch_weights])
                all_movie_embeddings.append(batch_embeddings.cpu())
            
            all_movie_embeddings = torch.cat(all_movie_embeddings, dim=0)
        
        # Generate recommendations
        print(f"Generating {args.num_recommendations} recommendations for movie ID {args.movie_id}...")
        
        # Choose search method
        if args.search_method == 'exact':
            # Exact search (brute force)
            query_embedding = all_movie_embeddings[movie_idx].unsqueeze(0)
            similarities = torch.matmul(query_embedding, all_movie_embeddings.t()).squeeze()
            similarities[movie_idx] = -float('inf')  # Exclude query movie
            _, indices = torch.topk(similarities, k=args.num_recommendations)
            recommended_indices = indices.numpy()
            
        elif args.search_method == 'lsh':
            # LSH-based search
            lsh_index = LSHIndex(args.embed_dim)
            lsh_index.build(all_movie_embeddings)
            
            query_embedding = all_movie_embeddings[movie_idx].unsqueeze(0).numpy()
            _, indices = lsh_index.search(query_embedding, k=args.num_recommendations + 1)
            recommended_indices = indices[0]
            
            # Remove query movie if present
            recommended_indices = [idx for idx in recommended_indices if idx != movie_idx][:args.num_recommendations]
            
        elif args.search_method == 'ivf':
            # Inverted file index search (Weak AND)
            ivf_index = WeakANDIndex(args.embed_dim)
            ivf_index.build(all_movie_embeddings)
            
            query_embedding = all_movie_embeddings[movie_idx].unsqueeze(0).numpy()
            _, indices = ivf_index.search(query_embedding, k=args.num_recommendations + 1)
            recommended_indices = indices[0]
            
            # Remove query movie if present
            recommended_indices = [idx for idx in recommended_indices if idx != movie_idx][:args.num_recommendations]
        
        # Print recommendations
        query_movie = dataset.movies_df[dataset.movies_df['movieId'] == args.movie_id]
        print(f"\nRecommendations for: {query_movie['title'].values[0]}")
        print("---------------------------------------------")
        
        for i, idx in enumerate(recommended_indices):
            movie_id = dataset.idx_to_movie_id[idx]
            movie = dataset.movies_df[dataset.movies_df['movieId'] == movie_id]
            if not movie.empty:
                print(f"{i+1}. {movie['title'].values[0]} - {movie['genres'].values[0]}")

if __name__ == '__main__':
    main()