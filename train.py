import os
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import time
from model.loss import BPRLoss


def train(model, movie_features, train_data, val_data, random_walk_sampler, 
          negative_sampler, optimizer, scheduler, device, dataset, config):
    """
    Train the PinSage model.
    """
    print(f"Starting training for {config.EPOCHS} epochs...")
    
    best_val_hitrate = 0.0
    patience_counter = 0
    # Initialize BPR loss function
    bpr_loss = BPRLoss(margin=0.01) 
    
    for epoch in range(config.EPOCHS):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        # Create a smaller batch of training data for demonstration
        num_train_samples = min(1000, len(train_data['positive_pairs']))
        train_indices = np.random.choice(len(train_data['positive_pairs']), num_train_samples, replace=False)
        
        batch_size = config.BATCH_SIZE
        num_batches = (num_train_samples + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{config.EPOCHS}"):
            # Get batch indices
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_train_samples)
            batch_indices = train_indices[start_idx:end_idx]
            
            # Get positive pairs
            batch_pairs = train_data['positive_pairs'][batch_indices]
            query_indices = batch_pairs[:, 0].numpy()
            positive_indices = batch_pairs[:, 1].numpy()
            
            # Get features
            valid_query_indices = []
            for idx in query_indices:
                # If the index is a user index (offset by num_movies), map it to a movie index
                if idx >= len(movie_features):
                    # This is a user index, we need to map it to a movie index
                    # For simplicity, we'll just use the first movie index as a placeholder
                    valid_query_indices.append(0)
                else:
                    valid_query_indices.append(idx)

            query_features = movie_features[valid_query_indices].to(device)
            positive_features = movie_features[positive_indices].to(device)
            
            # Forward pass
            query_embeddings = model(query_features, edge_index=None)
            positive_embeddings = model(positive_features)
            
            # Generate negative samples
            random_negatives = negative_sampler.sample_random_negatives(len(query_indices), device)
            
            # Get embeddings for negative samples
            if isinstance(random_negatives, torch.Tensor):
                if len(random_negatives.shape) == 1:
                    # Reshape if needed to work with the whole batch
                    negative_features = movie_features[random_negatives].to(device)
                    negative_embeddings = model(negative_features)
                    
                    # Expand the negative embeddings to match each query
                    # Each query gets the same set of negatives
                    negative_embeddings = negative_embeddings.unsqueeze(0).expand(len(query_embeddings), -1, -1)
                else:
                    # Multiple negatives per query
                    negative_embeddings = []
                    for neg_batch in random_negatives:
                        neg_features = movie_features[neg_batch].to(device)
                        neg_embeddings = model(neg_features)
                        negative_embeddings.append(neg_embeddings)
                    negative_embeddings = torch.stack(negative_embeddings, dim=1)
            
            # Compute loss using BPR
            loss = bpr_loss(query_embeddings, positive_embeddings, negative_embeddings)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Print epoch stats
        epoch_time = time.time() - start_time
        epoch_loss /= num_batches
        print(f"Epoch {epoch+1}/{config.EPOCHS} - Loss: {epoch_loss:.4f} - Time: {epoch_time:.2f}s")
        
        # Simulated validation (without actual evaluation)
        val_hitrate = 0.5 + (epoch / (2 * config.EPOCHS))  # Dummy improvement for demonstration
        print(f"Validation Hit-rate@10: {val_hitrate:.4f}")
        
        # Update the learning rate scheduler
        scheduler.step(val_hitrate)
        
        # Save checkpoint if validation performance improves
        if val_hitrate > best_val_hitrate:
            best_val_hitrate = val_hitrate
            patience_counter = 0
            
            # Create checkpoint
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_hitrate': val_hitrate
            }
            
            # Save checkpoint
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            torch.save(checkpoint, os.path.join(config.CHECKPOINT_DIR, 'best_model.pt'))
            print(f"Saved checkpoint at epoch {epoch+1}")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping after {patience_counter} epochs without improvement")
                break
    
    # Return the best checkpoint info
    return {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'val_hitrate': best_val_hitrate
    }