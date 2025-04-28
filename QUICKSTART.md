# PinSage Movie Recommender - Quick Start Guide

This guide will help you quickly set up and run the PinSage Movie Recommendation System.

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/pinsage-recommender.git
cd pinsage-recommender
```

## 2. Set Up Environment

```bash
# Create a virtual environment
python -m venv pinsage_env

# Activate the environment
# On Windows:
pinsage_env\Scripts\activate
# On macOS/Linux:
source pinsage_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 3. Download Dataset

```bash
# Run the download script
python download_dataset.py
```

## 4. Run the Demo

For a quick interactive demo that uses pre-trained embeddings (if available):

```bash
python demo.py
```

## 5. Training the Model

To train the model from scratch:

```bash
python run.py --mode train
```

This will:
- Process the MovieLens dataset
- Build the graph structure
- Train the PinSage model
- Save the model checkpoint and embeddings

## 6. Generate Recommendations

Once trained, you can generate recommendations:

```bash
# For a specific movie (e.g., Toy Story, ID: 1)
python run.py --mode recommend --movie_id 1 --num_recommendations 10
```

## Common Issues

1. **Memory errors during training**:
   ```bash
   # Reduce batch size
   python run.py --mode train --batch_size 256
   ```

2. **Slow training**:
   ```bash
   # Make sure to use GPU if available
   # Check your GPU is recognized:
   python -c "import torch; print('GPU available:', torch.cuda.is_available())"
   ```

3. **"File not found" errors**:
   ```bash
   # Make sure the dataset is downloaded and extracted correctly
   python download_dataset.py
   ```

## Next Steps

1. **Experiment with different model configurations** by modifying `config.py`
2. **Explore the interactive demo** to see recommendations for different movies
3. **Evaluate model performance** with `python run.py --mode evaluate`
4. **Add your own movie features** by extending the `FeatureExtractor` class

For more detailed information, refer to `README.md` and `SETUP_INSTRUCTIONS.md`.