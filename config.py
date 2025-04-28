"""
Configuration settings for the PinSage recommendation system.
"""

import os

# Data settings
DATA_DIR = "./data/ml-25m"  # Path to MovieLens dataset
MIN_INTERACTIONS = 5        # Minimum number of interactions per user to include

# Graph settings
USE_BIPARTITE_GRAPH = True  # If True, use bipartite user-item graph; otherwise, use item similarity graph
SIMILARITY_THRESHOLD = 5    # Minimum co-occurrence count for item similarity graph

# Feature settings
FEATURE_DIM = 128           # Dimension of input features
USE_VISUAL_FEATURES = False # Whether to use visual features

# Model settings
HIDDEN_DIM = 256            # Dimension of hidden layers
EMBED_DIM = 128             # Dimension of final embeddings
NUM_LAYERS = 2              # Number of graph convolutional layers
AGGREGATOR_TYPE = "importance"  # Options: "mean", "weighted", "attention", "max", "importance"
DROPOUT = 0.2               # Dropout rate
USE_BATCH_NORM = True       # Whether to use batch normalization

# Random walk settings
WALK_LENGTH = 2             # Length of random walks
NUM_WALKS = 100             # Number of random walks per node
NUM_NEIGHBORS = 50          # Number of neighbors to sample per node

# Training settings
BATCH_SIZE = 512            # Batch size for training
EPOCHS = 10                 # Number of training epochs
LEARNING_RATE = 0.001       # Learning rate
MARGIN = 0.1                # Margin for max-margin loss
NUM_NEGATIVE_SAMPLES = 500  # Number of negative samples per batch
HARD_NEG_FACTOR = 2.0       # Factor to scale hard negative loss
NUM_WORKERS = 4             # Number of workers for data loading
VAL_RATIO = 0.1             # Ratio of validation data
TEST_RATIO = 0.2            # Ratio of test data

# Evaluation settings
K_VALUES = [10, 50, 100, 500]  # k values for hit rate calculation
EVAL_EVERY = 1              # Evaluate model every n epochs
PATIENCE = 3                # Early stopping patience

# Search settings
SEARCH_METHOD = "exact"     # Options: "exact", "lsh", "ivf"
LSH_BITS = 256              # Number of bits for LSH
LSH_TABLES = 16             # Number of tables for LSH
IVF_PARTITIONS = 100        # Number of partitions for IVF
IVF_FACTOR = 10             # Candidates factor for IVF

# File paths
CHECKPOINT_DIR = "./checkpoints"  # Directory for model checkpoints
OUTPUT_DIR = "./output"           # Directory for output files

# Make sure directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Add or modify these settings:
USE_DATA_SUBSET = True      # Whether to use a subset of data
DATA_SUBSET_FRACTION = 0.01  # Fraction of data to use (1%)