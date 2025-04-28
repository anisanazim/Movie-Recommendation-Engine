import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
import pandas as pd

from data.dataset import MovieLensDataset
from model.pinsage import PinSage
from utils.random_walk import RandomWalkSampler
from utils.nearest_neighbors import LSHIndex, WeakANDIndex

def generate_all_embeddings(model, movie_features, dataset, random_walk_sampler, args, device):
    """
    Generate embeddings for all movies.
    
    Args:
        model: Trained PinSage model
        movie_features: Movie feature matrix
        dataset: MovieLensDataset instance
        random_walk_sampler: RandomWalkSampler instance
        args: Command line arguments
        device: Device to use
        
    Returns:
        all_movie_embeddings: Tensor of embeddings for all movies
    """
    model.eval()
    
    print("Generating embeddings for all movies...")
    with torch.no_grad():
        all_movie_embeddings = []
        batch_size = args.batch_size
        
        for i in range(0, len(dataset.movie_id_to_idx), batch_size):
            batch_indices = list(range(i, min(i + batch_size, len(dataset.movie_id_to_idx))))
            batch_features = movie_features[batch_indices].to(device)
            
            # Sample neighbors for each layer
            all_neighbors = []
            all_weights = []
            
            for _ in range(args.num_layers):
                # Sample neighbors
                neighbors, weights = random_walk_sampler.batch_sample_neighbors(
                    batch_indices, num_neighbors=args.num_neighbors
                )
                all_neighbors.append(neighbors)
                all_weights.append(weights)
            
            # Generate embeddings
            batch_embeddings = model(batch_features, all_neighbors, all_weights)
            all_movie_embeddings.append(batch_embeddings.cpu())
        
        all_movie_embeddings = torch.cat(all_movie_embeddings, dim=0)
    
    return all_movie_embeddings

def build_search_index(embeddings, args):
    """
    Build a search index for efficient nearest neighbor search.
    
    Args:
        embeddings: Movie embeddings
        args: Command line arguments
        
    Returns:
        index: Search index for nearest neighbor lookup
    """
    if args.search_method == 'lsh':
        # LSH-based index
        print("Building LSH index...")
        index = LSHIndex(
            dim=embeddings.shape[1], 
            num_bits=args.lsh_bits, 
            num_tables=args.lsh_tables
        )
        index.build(embeddings)
        return index
    
    elif args.search_method == 'ivf':
        # Inverted file index (Weak AND)
        print("Building IVF index...")
        index = WeakANDIndex(
            dim=embeddings.shape[1],
            num_partitions=args.ivf_partitions,
            candidates_factor=args.ivf_factor
        )
        index.build(embeddings)
        return index
    
    else:
        # Exact search (no index needed)
        return None

def generate_recommendations(query_idx, embeddings, index, args, dataset):
    """
    Generate recommendations for a query movie.
    
    Args:
        query_idx: Index of the query movie
        embeddings: Movie embeddings
        index: Search index for nearest neighbor lookup
        args: Command line arguments
        dataset: MovieLensDataset instance
        
    Returns:
        recommendations: List of recommended movie IDs and metadata
    """
    k = args.num_recommendations + 1  # Add 1 to account for the query itself
    
    if args.search_method == 'exact':
        # Exact search (brute force)
        query_embedding = embeddings[query_idx].unsqueeze(0)
        similarities = torch.matmul(query_embedding, embeddings.t()).squeeze()
        similarities[query_idx] = -float('inf')  # Exclude query movie
        _, indices = torch.topk(similarities, k=k)
        recommended_indices = indices.numpy()
        
    else:
        # Use the search index
        query_embedding = embeddings[query_idx].unsqueeze(0).numpy()
        _, indices = index.search(query_embedding, k=k)
        recommended_indices = indices[0]
        
        # Remove query movie if present
        recommended_indices = [idx for idx in recommended_indices if idx != query_idx][:args.num_recommendations]
    
    # Get movie details
    recommendations = []
    
    for idx in recommended_indices:
        movie_id = dataset.idx_to_movie_id[idx]
        movie = dataset.movies_df[dataset.movies_df['movieId'] == movie_id]
        
        if not movie.empty:
            recommendations.append({
                'movieId': int(movie_id),
                'title': movie['title'].values[0],
                'genres': movie['genres'].values[0],
                'index': int(idx)
            })
    
    return recommendations

def save_embeddings(embeddings, output_dir, dataset):
    """
    Save generated embeddings to disk.
    
    Args:
        embeddings: Movie embeddings
        output_dir: Directory to save embeddings
        dataset: MovieLensDataset instance
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save embeddings
    torch.save(embeddings, os.path.join(output_dir, 'movie_embeddings.pt'))
    
    # Save mapping from movie ID to index
    movie_mapping = {
        'movieId': list(dataset.movie_id_to_idx.keys()),
        'index': list(dataset.movie_id_to_idx.values())
    }
    
    pd.DataFrame(movie_mapping).to_csv(
        os.path.join(output_dir, 'movie_mapping.csv'), index=False
    )
    
    print(f"Saved embeddings and mapping to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='PinSage Inference')
    
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
    
    # Inference parameters
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                      help='Directory for model checkpoints')
    parser.add_argument('--checkpoint', type=str, default='best_model.pt',
                      help='Checkpoint file to load')
    parser.add_argument('--batch_size', type=int, default=1024,
                      help='Batch size for inference')
    parser.add_argument('--output_dir', type=str, default='./output',
                      help='Directory to save output')
    
    # Recommendation parameters
    parser.add_argument('--movie_id', type=int, default=None,
                      help='Movie ID to generate recommendations for')
    parser.add_argument('--num_recommendations', type=int, default=10,
                      help='Number of recommendations to generate')
    parser.add_argument('--search_method', type=str, default='exact', 
                      choices=['exact', 'lsh', 'ivf'],
                      help='Search method for nearest neighbor lookup')
    
    # LSH parameters
    parser.add_argument('--lsh_bits', type=int, default=256,
                      help='Number of bits for LSH')
    parser.add_argument('--lsh_tables', type=int, default=16,
                      help='Number of tables for LSH')
    
    # IVF parameters
    parser.add_argument('--ivf_partitions', type=int, default=100,
                      help='Number of partitions for IVF')
    parser.add_argument('--ivf_factor', type=int, default=10,
                      help='Candidates factor for IVF')
    
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
        print(f"Checkpoint not found at {checkpoint_path}.")
        return
    
    # Generate embeddings for all movies
    embeddings = generate_all_embeddings(
        model, movie_features, dataset, random_walk_sampler, args, device
    )
    
    # Save embeddings
    save_embeddings(embeddings, args.output_dir, dataset)
    
    # Build search index
    index = build_search_index(embeddings, args)
    
    # Generate recommendations if a movie ID is provided
    if args.movie_id is not None:
        # Check if movie ID exists
        if args.movie_id not in dataset.movie_id_to_idx:
            print(f"Movie ID {args.movie_id} not found in the dataset.")
            return
        
        # Get movie index
        movie_idx = dataset.movie_id_to_idx[args.movie_id]
        
        # Generate recommendations
        print(f"Generating {args.num_recommendations} recommendations for movie ID {args.movie_id}...")
        recommendations = generate_recommendations(
            movie_idx, embeddings, index, args, dataset
        )
        
        # Print recommendations
        query_movie = dataset.movies_df[dataset.movies_df['movieId'] == args.movie_id]
        print(f"\nRecommendations for: {query_movie['title'].values[0]}")
        print("---------------------------------------------")
        
        for i, rec in enumerate(recommendations):
            print(f"{i+1}. {rec['title']} - {rec['genres']}")
        
        # Save recommendations
        os.makedirs(args.output_dir, exist_ok=True)
        pd.DataFrame(recommendations).to_csv(
            os.path.join(args.output_dir, f'recommendations_{args.movie_id}.csv'), 
            index=False
        )
    
if __name__ == '__main__':
    main()