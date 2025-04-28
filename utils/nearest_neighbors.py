import numpy as np
import torch
import faiss
import time
from tqdm import tqdm

class LSHIndex:
    """
    LSH-based index for efficient nearest neighbor search in high-dimensional spaces.
    Uses the FAISS library for implementation.
    """
    def __init__(self, dim, num_bits=256, num_tables=16):
        """
        Initialize the LSH index.
        
        Args:
            dim (int): Dimensionality of the embeddings
            num_bits (int): Number of bits for the hash codes
            num_tables (int): Number of hash tables
        """
        self.dim = dim
        self.num_bits = num_bits
        self.num_tables = num_tables
        
        # Create index
        self.index = faiss.IndexLSH(dim, num_bits, num_tables)
        
    def build(self, embeddings):
        """
        Build the index from a set of embeddings.
        
        Args:
            embeddings (numpy.ndarray or torch.Tensor): Embeddings to index
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        # Ensure embeddings are float32
        embeddings = embeddings.astype(np.float32)
        
        # Build index
        self.index.train(embeddings)
        self.index.add(embeddings)
        
        print(f"Built LSH index with {len(embeddings)} embeddings")
        
    def search(self, queries, k=10):
        """
        Search for nearest neighbors.
        
        Args:
            queries (numpy.ndarray or torch.Tensor): Query embeddings
            k (int): Number of neighbors to retrieve
            
        Returns:
            distances (numpy.ndarray): Distances to nearest neighbors
            indices (numpy.ndarray): Indices of nearest neighbors
        """
        if isinstance(queries, torch.Tensor):
            queries = queries.cpu().numpy()
        
        # Ensure queries are float32
        queries = queries.astype(np.float32)
        
        # Search
        distances, indices = self.index.search(queries, k)
        
        return distances, indices

class WeakANDIndex:
    """
    Efficient two-level retrieval process based on the Weak AND operator.
    Implementation based on the description in the PinSage paper.
    """
    def __init__(self, dim, num_partitions=100, candidates_factor=10):
        """
        Initialize the Weak AND index.
        
        Args:
            dim (int): Dimensionality of the embeddings
            num_partitions (int): Number of partitions for first-level search
            candidates_factor (int): Factor to determine number of candidates in first level
        """
        self.dim = dim
        self.num_partitions = num_partitions
        self.candidates_factor = candidates_factor
        
        # First-level index (quantizer)
        self.quantizer = faiss.IndexFlatL2(dim)
        
        # Second-level index (inverted file index)
        self.index = faiss.IndexIVFFlat(self.quantizer, dim, num_partitions)
        
    def build(self, embeddings):
        """
        Build the index from a set of embeddings.
        
        Args:
            embeddings (numpy.ndarray or torch.Tensor): Embeddings to index
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        # Ensure embeddings are float32
        embeddings = embeddings.astype(np.float32)
        
        # Train index
        self.index.train(embeddings)
        
        # Add embeddings
        self.index.add(embeddings)
        
        print(f"Built Weak AND index with {len(embeddings)} embeddings")
        
    def search(self, queries, k=10):
        """
        Search for nearest neighbors using the Weak AND approach.
        
        Args:
            queries (numpy.ndarray or torch.Tensor): Query embeddings
            k (int): Number of neighbors to retrieve
            
        Returns:
            distances (numpy.ndarray): Distances to nearest neighbors
            indices (numpy.ndarray): Indices of nearest neighbors
        """
        if isinstance(queries, torch.Tensor):
            queries = queries.cpu().numpy()
        
        # Ensure queries are float32
        queries = queries.astype(np.float32)
        
        # Set number of probes (partitions to search)
        self.index.nprobe = min(self.num_partitions, 20)  # Typically 10-20% of partitions
        
        # Search
        distances, indices = self.index.search(queries, k)
        
        return distances, indices

def benchmark_search_methods(embeddings, queries, k=10, methods=None):
    """
    Benchmark different nearest neighbor search methods.
    
    Args:
        embeddings (torch.Tensor or numpy.ndarray): Item embeddings
        queries (torch.Tensor or numpy.ndarray): Query embeddings
        k (int): Number of neighbors to retrieve
        methods (list): List of search methods to benchmark
        
    Returns:
        results (dict): Dictionary with benchmark results
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    
    if isinstance(queries, torch.Tensor):
        queries = queries.cpu().numpy()
    
    # Ensure data is float32
    embeddings = embeddings.astype(np.float32)
    queries = queries.astype(np.float32)
    
    dim = embeddings.shape[1]
    
    if methods is None:
        methods = ['exact', 'lsh', 'ivf']
    
    results = {}
    
    for method in methods:
        print(f"Benchmarking {method} search...")
        
        if method == 'exact':
            # Exact search (brute force)
            index = faiss.IndexFlatL2(dim)
            index.add(embeddings)
            
            # Measure search time
            start_time = time.time()
            distances, indices = index.search(queries, k)
            search_time = time.time() - start_time
            
            results['exact'] = {
                'distances': distances,
                'indices': indices,
                'search_time': search_time,
                'index_size': index.ntotal,
                'method': 'Exact (Brute Force)'
            }
            
        elif method == 'lsh':
            # LSH-based search
            lsh_index = LSHIndex(dim)
            lsh_index.build(embeddings)
            
            # Measure search time
            start_time = time.time()
            distances, indices = lsh_index.search(queries, k)
            search_time = time.time() - start_time
            
            results['lsh'] = {
                'distances': distances,
                'indices': indices,
                'search_time': search_time,
                'index_size': lsh_index.index.ntotal,
                'method': 'Locality-Sensitive Hashing'
            }
            
        elif method == 'ivf':
            # Inverted file index search (Weak AND)
            ivf_index = WeakANDIndex(dim)
            ivf_index.build(embeddings)
            
            # Measure search time
            start_time = time.time()
            distances, indices = ivf_index.search(queries, k)
            search_time = time.time() - start_time
            
            results['ivf'] = {
                'distances': distances,
                'indices': indices,
                'search_time': search_time,
                'index_size': ivf_index.index.ntotal,
                'method': 'Weak AND (IVF)'
            }
    
    # Print results
    print("\nBenchmark Results:")
    print("-----------------")
    for method, data in results.items():
        print(f"{data['method']}:")
        print(f"  Search time: {data['search_time']:.6f} seconds")
        print(f"  Index size: {data['index_size']} vectors")
    
    # Compare accuracy against exact search (if available)
    if 'exact' in results:
        exact_indices = results['exact']['indices']
        
        for method, data in results.items():
            if method != 'exact':
                # Calculate recall (% of exact nearest neighbors found)
                recall = 0
                for i in range(len(queries)):
                    exact_set = set(exact_indices[i])
                    method_set = set(data['indices'][i])
                    intersection = exact_set.intersection(method_set)
                    recall += len(intersection) / k
                
                recall /= len(queries)
                results[method]['recall'] = recall
                print(f"  {data['method']} recall@{k}: {recall:.4f}")
    
    return results