#!/usr/bin/env python3
"""
Test script to check TNC training for a few epochs
"""
import torch
from torch.utils import data
import numpy as np
import pickle
import os
import random
from tnc.models import RnnEncoder
from tnc.tnc import Discriminator, TNCDataset, epoch_run

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load data
print("Loading data...")
with open('./data/simulated_data/x_train.pkl', 'rb') as f:
    x = pickle.load(f)

print(f"Data shape: {x.shape}")
print(f"Data range: [{np.min(x):.3f}, {np.max(x):.3f}]")
print(f"NaN in data: {np.isnan(x).any()}")
print(f"Inf in data: {np.isinf(x).any()}")

# Initialize models (exactly like original)
window_size = 50
encoder = RnnEncoder(hidden_size=100, in_channel=3, encoding_size=10, device=device)
disc_model = Discriminator(encoder.encoding_size, device)
params = list(disc_model.parameters()) + list(encoder.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-5)

# Shuffle and split data
inds = list(range(len(x)))
random.shuffle(inds)
x = x[inds]
n_train = int(0.8*len(x))

print(f"Training on {n_train} samples, validating on {len(x)-n_train} samples")

# Test training for 3 epochs
w = 0.05
mc_sample_size = 20
batch_size = 10
augmentation = 1

print("\nStarting training test...")
for epoch in range(3):
    print(f"\n--- Epoch {epoch} ---")
    
    # Create datasets
    trainset = TNCDataset(x=torch.Tensor(x[:n_train]), mc_sample_size=mc_sample_size,
                          window_size=window_size, augmentation=augmentation, adf=True)
    train_loader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    validset = TNCDataset(x=torch.Tensor(x[n_train:]), mc_sample_size=mc_sample_size,
                          window_size=window_size, augmentation=augmentation, adf=True)
    valid_loader = data.DataLoader(validset, batch_size=batch_size, shuffle=True, num_workers=0)

    print(f"Created datasets - Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")
    
    # Training step
    try:
        epoch_loss, epoch_acc = epoch_run(train_loader, disc_model, encoder, optimizer=optimizer,
                                          w=w, train=True, device=device)
        print(f"Train - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
        
        # Validation step
        test_loss, test_acc = epoch_run(valid_loader, disc_model, encoder, train=False, w=w, device=device)
        print(f"Valid - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
        
    except Exception as e:
        print(f"Error in epoch {epoch}: {e}")
        import traceback
        traceback.print_exc()
        break

print("\nTraining test completed!")
