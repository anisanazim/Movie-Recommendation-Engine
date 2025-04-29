# Setting Up and Running the PinSage Movie Recommendation System

This document provides detailed instructions for setting up and running the PinSage-inspired Movie Recommendation System.

## Environment Setup

### Prerequisites
- Python 3.8+ installed
- Pip package manager
- CUDA-compatible GPU (optional but recommended)

### Installation Steps

1. **Create a virtual environment**:
   ```bash
   # Create a new virtual environment
   python -m venv pinsage_env
   
   # Activate the environment
   # On Windows:
   pinsage_env\Scripts\activate
   # On macOS/Linux:
   source pinsage_env/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   # Install PyTorch (with CUDA if available)
   # For CUDA 11.7:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
   # For CPU only:
   pip install torch torchvision torchaudio
   
   # Install other dependencies
   pip install numpy pandas scikit-learn matplotlib tqdm
   
   # Install graph neural network libraries
   pip install torch-geometric
   pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-{TORCH_VERSION}+{CUDA_VERSION}.html
   
   # Install FAISS for efficient nearest neighbor search
   # For GPU:
   pip install faiss-gpu
   # For CPU:
   pip install faiss-cpu
   ```

3. **Download the MovieLens dataset**:
   ```bash
   # Create data directory
   mkdir -p data
   
   # Download the dataset
   wget https://files.grouplens.org/datasets/movielens/ml-25m.zip -O data/ml-25m.zip
   
   # Extract the dataset
   unzip data/ml-25m.zip -d data/
   ```

## Project Structure

Ensure your project has the following directory structure:
```
pinsage-recommender/
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   ├── graph_builder.py
│   ├── feature_extractor.py
│   ├── negative_sampler.py
│   └── ml-25m/          # MovieLens dataset files
│
├── model/
│   ├── __init__.py
│   ├── pinsage.py
│   ├── layers.py
│   ├── aggregators.py
│   └── loss.py
│
├── utils/
│   ├── __init__.py
│   ├── random_walk.py
│   ├── evaluation.py
│   └── nearest_neighbors.py
│
├── checkpoints/         # Directory for saved models
├── output/              # Directory for outputs
├── config.py            # Configuration settings
├── train.py             # Training script
├── inference.py         # Inference script
├── main.py              # Main entry point
└── run.py               # Pipeline running script
```

## Configuration

Before running the system, check the configuration parameters in `config.py`. Key parameters include:

- `DATA_DIR`: Path to the MovieLens dataset
- `FEATURE_DIM`: Dimension of input features
- `HIDDEN_DIM`: Dimension of hidden layers
- `EMBED_DIM`: Dimension of final embeddings
- `NUM_LAYERS`: Number of graph convolutional layers
- `WALK_LENGTH`: Length of random walks
- `NUM_WALKS`: Number of random walks per node
- `NUM_NEIGHBORS`: Number of neighbors to sample per node
- `BATCH_SIZE`: Batch size for training
- `EPOCHS`: Number of training epochs
- `LEARNING_RATE`: Learning rate for optimization

## Running the System

The system can be run in different modes using the `run.py` script:

### 1. Training Mode

Train the PinSage model from scratch:

```bash
python run.py --mode train --seed 42
```

This will:
- Process the MovieLens dataset
- Build the graph structure
- Extract features for movies
- Train the PinSage model
- Save the best model checkpoint

### 2. Evaluation Mode

Evaluate a trained model on the test set:

```bash
python run.py --mode evaluate --seed 42
```

This will:
- Load the trained model
- Generate embeddings for all movies
- Evaluate using metrics like Hit Rate@k and MRR
- Save the embeddings

### 3. Recommendation Mode

Generate recommendations for a specific movie:

```bash
python run.py --mode recommend --movie_id 1 --num_recommendations 10
```

This will:
- Load the trained model
- Generate embeddings for all movies
- Find the most similar movies to the specified movie
- Print the recommendations

### 4. Full Pipeline

Run the entire pipeline from training to recommendations:

```bash
python run.py --mode all --movie_id 1 --num_recommendations 10
```

## Example Usage

Here's an example workflow:

1. **Process data and train the model**:
   ```bash
   python run.py --mode train
   ```

2. **Generate recommendations for "Toy Story"** (MovieID: 1):
   ```bash
   python run.py --mode recommend --movie_id 1 --num_recommendations 10
   ```

3. **Generate recommendations for "The Dark Knight"** (MovieID: 58559):
   ```bash
   python run.py --mode recommend --movie_id 58559 --num_recommendations 10
   ```

## Troubleshooting

1. **Memory issues during training**:
   - Reduce `BATCH_SIZE` in config.py
   - Reduce `NUM_NEIGHBORS` to sample fewer neighbors
   - Reduce `NUM_WALKS` to perform fewer random walks

2. **Slow training**:
   - Increase `BATCH_SIZE` if memory allows
   - Reduce `NUM_NEGATIVE_SAMPLES`
   - Use GPU acceleration if available

3. **Poor recommendations**:
   - Increase `EPOCHS` for more training
   - Adjust `MARGIN` in the loss function
   - Try different aggregator types
   - Use more informative movie features

## Advanced Usage

### Customizing Graph Structure

The system supports two types of graphs:
- Bipartite user-item graph (default)
- Item similarity graph based on co-occurrence

To use the item similarity graph, modify `config.py`:
```python
USE_BIPARTITE_GRAPH = False
SIMILARITY_THRESHOLD = 5  # Minimum co-occurrence count
```

### Efficient Nearest Neighbor Search

For large-scale deployment, the system supports different search methods:
- Exact search (brute force)
- Locality-Sensitive Hashing (LSH)
- Inverted File Index with Weak AND

Modify `config.py` to change the search method:
```python
SEARCH_METHOD = "lsh"  # Options: "exact", "lsh", "ivf"
LSH_BITS = 256
LSH_TABLES = 16
```

## Performance Optimization

For better performance:

1. **Feature Engineering**:
   - Enable visual features: `USE_VISUAL_FEATURES = True`
   - Add more movie metadata if available

2. **Model Architecture**:
   - Try different aggregator types: `AGGREGATOR_TYPE = "attention"`
   - Adjust number of layers: `NUM_LAYERS = 3`

3. **Training Strategy**:
   - Adjust curriculum learning parameters
   - Use learning rate scheduling
