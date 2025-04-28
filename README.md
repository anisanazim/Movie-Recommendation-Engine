# PinSage-Inspired Movie Recommendation System

This project implements a movie recommendation system using a Graph Convolutional Network (GCN) inspired by the PinSage architecture described in the research paper ["Graph Convolutional Neural Networks for Web-Scale Recommender Systems"](https://arxiv.org/abs/1806.01973). The implementation focuses on learning high-quality item embeddings by leveraging both item content features and the graph structure of user-item interactions.

## Features

- **Graph Convolutional Networks (GCNs)** for learning item embeddings
- **Random Walk-Based Neighborhood Sampling** for efficient training on large graphs
- **Importance Pooling** to incorporate node importance in neighborhood aggregation
- **Curriculum Learning** with progressively harder negative examples
- **Locality-Sensitive Hashing (LSH)** for efficient nearest-neighbor search
- **Weak AND** operator for two-level retrieval
- Evaluation metrics including Hit Rate and Mean Reciprocal Rank (MRR)

## Project Structure

```
pinsage-recommender/
├── data/
│   ├── __init__.py
│   ├── dataset.py           # Dataset loading and processing
│   ├── graph_builder.py     # Build graph from interaction
│   ├── feature_extractor.py # Extract and process movie
│   ├── negative_sampler.py  # Negative sampling strategies
│   └── ml-25m/              # MovieLens dataset files
│
├── model/
│   ├── __init__.py
│   ├── pinsage.py           # PinSage model implementation
│   ├── layers.py            # Graph convolutional layers
│   ├── aggregators.py       # Neighborhood aggregation functions
│   └── loss.py              # Loss functions
│
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── random_walk.py       # Random walk sampling
│   ├── evaluation.py        # Evaluation metrics
│   └── nearest_neighbors.py # Efficient nearest neighbor search
│
├── checkpoints/             # Directory for saved models
├── output/                  # Directory for outputs
├── config.py                # Configuration settings
├── download_dataset.py      # To download the datasets from site
├── train.py                 # Training script
├── inference.py             # Inference script
├── main.py                  # Main entry point
├── run.py                   # Pipeline running script
├── README.md                # readme file
├── QUICKSTART.md            # Quickstart guide
├── requirement.txt          # requirements
└── SETUP_INSTRUCTIONS.md    # Instructions for setup


## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/pinsage-recommendation.git
cd pinsage-recommendation
```

2. Create a virtual environment:

```bash
python -m venv pinsage_env
source pinsage_env/bin/activate  # On Windows: pinsage_env\Scripts\activate
```

3. Install requirements:

```bash
pip install torch torch_geometric pandas numpy scikit-learn matplotlib tqdm
pip install torch_sparse torch_scatter  # For graph operations
pip install faiss-gpu  # For efficient nearest neighbor search (or faiss-cpu if no GPU)
```

## Dataset Preparation

1. Download the MovieLens 25M dataset from [https://grouplens.org/datasets/movielens/25m/](https://grouplens.org/datasets/movielens/25m/)
2. Extract the dataset to the `data/ml-25m` directory

## Usage

### Training

Train the PinSage model on the MovieLens dataset:

```bash
python train.py --data_dir data/ml-25m --batch_size 512 --epochs 10 --lr 0.001
```

Or use the main script:

```bash
python main.py --mode train --data_dir data/ml-25m
```

### Evaluation

Evaluate the trained model:

```bash
python main.py --mode evaluate --data_dir data/ml-25m --checkpoint_dir checkpoints
```

### Inference

Generate recommendations for a specific movie:

```bash
python inference.py --data_dir data/ml-25m --movie_id 1 --num_recommendations 10
```

Or using the main script:

```bash
python main.py --mode inference --data_dir data/ml-25m --movie_id 1 --num_recommendations 10
```

### Search Methods

The system supports three different search methods for nearest neighbor lookup:

1. **Exact Search** (brute force): Most accurate but slowest
   ```bash
   python main.py --mode inference --search_method exact
   ```

2. **Locality-Sensitive Hashing (LSH)**: Fast approximate search
   ```bash
   python main.py --mode inference --search_method lsh --lsh_bits 256 --lsh_tables 16
   ```

3. **Inverted File Index (IVF) with Weak AND**: Balance between speed and accuracy
   ```bash
   python main.py --mode inference --search_method ivf --ivf_partitions 100 --ivf_factor 10
   ```

## Key Implementation Details

### Random Walk-Based Neighborhood Sampling

Instead of using full k-hop neighborhoods, PinSage samples important nodes using random walks:

```python
# Sample neighbors using random walks
neighbors, weights = random_walk_sampler.batch_sample_neighbors(
    nodes, num_neighbors=args.num_neighbors
)
```

### Importance Pooling

Weights node features in neighborhood aggregation based on random walk visit counts:

```python
# Importance pooling for neighborhood aggregation
h_neigh = importance_pooling(h, neighbors, weights)
```

### Hard Negative Sampling

Generates challenging negative examples for better training:

```python
# Sample both random and hard negative examples
random_negatives, hard_negatives = negative_sampler.sample_batch_negatives(
    query_indices, device, epoch=epoch
)
```

### Curriculum Learning

Gradually introduces harder examples during training:

```python
# Number of hard negatives increases with epoch
num_hard = min(epoch, 6)  # Maximum of 6 hard negatives per query
```

## Performance Optimization

This implementation includes several optimizations for large-scale graphs:

1. **Efficient Neighborhood Sampling**: Uses random walks instead of full neighborhood expansion
2. **Batch Processing**: Processes nodes in batches for memory efficiency
3. **Fast Nearest Neighbor Search**: LSH and Weak AND for efficient recommendation retrieval
4. **Early Stopping**: Prevents overfitting with patience-based early stopping

## Citation

If you use this code in your research, please cite the original PinSage paper:

```
@inproceedings{ying2018graph,
  title={Graph Convolutional Neural Networks for Web-Scale Recommender Systems},
  author={Ying, Rex and He, Ruining and Chen, Kaifeng and Eksombatchai, Pong and Hamilton, William L. and Leskovec, Jure},
  booktitle={Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={974--983},
  year={2018}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.