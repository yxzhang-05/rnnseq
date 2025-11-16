#!/usr/bin/env python3
"""
eval_autoencoder_generalization.py

Script to evaluate the generalization capability of RNNAutoencoder models.
This includes training the model, testing on seen/unseen data, and analyzing
representations using overlap and similarity metrics.

Usage:
    python eval_autoencoder_generalization.py

Outputs:
    - Training loss curves
    - Reconstruction errors on train/test sets
    - Similarity matrices and plots for latent representations
    - Quantitative metrics (e.g., mean reconstruction error, overlap scores)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
import os
from os.path import join
import pickle
from itertools import product

# Import from project modules
from model import RNNAutoencoder, LowRankLinear
from train import generate_sequences, train, tokenwise_test
from analysis_utils import compute_overlap_AE, plot_overlap_AE, compute_cosine_similarity
import par_transfer as par  # Import the parameter module

# Set default clustering if not defined
if not hasattr(par, 'clustering'):
    par.clustering = 'class'

# Configuration
L = 6  # sequence length
d_input = 4  # alphabet size
d_hidden = 32  # hidden layer size (reduced for speed)
num_layers = 1
d_latent = 8  # latent dimension (reduced)
n_epochs = 20  # training epochs (reduced)
batch_size = 16  # reduced
n_batches = 50  # number of training batches (reduced)
objective = 'MSE'  # loss function
device = 'cpu'

# Set par parameters to match
par.L = L

# Data split: train on subset, test on unseen
train_pairs = [(0,1), (0,2)]  # train on these token pairs
test_pairs = [(0,3), (1,2), (1,3), (2,3)]  # test on unseen pairs

def generate_train_test_sequences(N, L, train_pairs, test_pairs):
    """Generate sequences for train and test sets based on token pairs."""
    train_sequences = []
    test_sequences = []
    
    for a, b in train_pairs:
        # Generate patterns for train pairs
        seq1 = [a if i % 2 == 0 else b for i in range(L)]
        seq2 = [a if (i // 2) % 2 == 0 else b for i in range(L)]
        seq3 = [a if (i // 3) % 2 == 0 else b for i in range(L)]
        train_sequences.extend([seq1, seq2, seq3])
    
    for a, b in test_pairs:
        # Generate patterns for test pairs
        seq1 = [a if i % 2 == 0 else b for i in range(L)]
        seq2 = [a if (i // 2) % 2 == 0 else b for i in range(L)]
        seq3 = [a if (i // 3) % 2 == 0 else b for i in range(L)]
        test_sequences.extend([seq1, seq2, seq3])
    
    # Convert to tensors
    train_seq = torch.tensor(train_sequences, dtype=torch.int64).T
    test_seq = torch.tensor(test_sequences, dtype=torch.int64).T
    
    # One-hot encode
    X_train = F.one_hot(train_seq, num_classes=N).float()
    X_test = F.one_hot(test_seq, num_classes=N).float()
    
    return X_train, X_test

def evaluate_reconstruction(model, X, task='RNNAuto', objective='MSE'):
    """Compute reconstruction loss on given data."""
    model.eval()
    loss_fn = lambda output, target: F.mse_loss(output, target, reduction="mean")
    
    with torch.no_grad():
        latent, output, _ = model(X)
        loss = loss_fn(output, X)
    return loss.item()

def analyze_latent_representations(model, X, tokens, types, filename):
    """Analyze and plot latent representations using overlap analysis."""
    model.eval()
    with torch.no_grad():
        latent, _, _ = model(X)
        # latent shape: (seq_len, batch, d_latent) -> we take last timestep
        latent_last = latent[-1]  # (batch, d_latent)
    
    # Compute overlap matrix (dot product)
    overlap = np.dot(latent_last.numpy(), latent_last.numpy().T)
    
    # Compute overlap
    M_t, sorted_ticklabels = compute_overlap_AE(overlap, tokens, types)
    
    # Plot overlap
    plot_overlap_AE(M_t, sorted_ticklabels, filename, 'RNNAuto')
    
    # Compute cosine similarity
    sim_matrix = compute_cosine_similarity(latent_last.unsqueeze(0))  # Add time dim
    return sim_matrix.squeeze(0).numpy()

def main():
    print("Starting Autoencoder Generalization Evaluation...")
    
    # Create output directory
    out_dir = "autoencoder_generalization_results"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(join(out_dir, 'figs'), exist_ok=True)
    
    # Set par parameters
    par.L = L
    par.folder = out_dir
    print("Generating train/test sequences...")
    X_train, X_test = generate_train_test_sequences(d_input, L, train_pairs, test_pairs)
    print(f"Train sequences shape: {X_train.shape}")
    print(f"Test sequences shape: {X_test.shape}")
    
    # For RNNAuto, y_train/y_test are not needed, but train function expects them
    # Create dummy labels
    y_train = torch.zeros(X_train.shape[1], d_input)  # dummy labels
    y_test = torch.zeros(X_test.shape[1], d_input)    # dummy labels
    
    # Prepare tokens and types for analysis
    tokens_train = [''.join(map(str, seq)) for seq in X_train.argmax(dim=-1).T.tolist()]
    tokens_test = [''.join(map(str, seq)) for seq in X_test.argmax(dim=-1).T.tolist()]
    types_train = [f"pair_{a}_{b}" for a, b in train_pairs for _ in range(3)]  # 3 patterns per pair
    types_test = [f"pair_{a}_{b}" for a, b in test_pairs for _ in range(3)]
    
    # Initialize model
    model = RNNAutoencoder(
        d_input=d_input,
        d_hidden=d_hidden,
        num_layers=num_layers,
        d_latent=d_latent,
        sequence_length=L,
        nonlinearity='relu',
        device=device,
        init_weights=None,
        layer_type=LowRankLinear
    )
    
    optimizer = Adam(model.parameters(), lr=0.001)
    
    # Training
    print("Training autoencoder...")
    train_losses = []
    for epoch in range(n_epochs):
        # Train for one epoch
        train(X_train, y_train, model, optimizer, objective, n_batches, batch_size, 'RNNAuto')
        
        # Evaluate reconstruction on train set
        train_loss = evaluate_reconstruction(model, X_train)
        train_losses.append(train_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
    
    # Save training curve
    plt.figure()
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss (MSE)')
    plt.title('Autoencoder Training Loss')
    plt.savefig(join(out_dir, 'training_loss.png'))
    plt.close()
    
    # Evaluate on test set
    test_loss = evaluate_reconstruction(model, X_test)
    print(f"Test Reconstruction Loss: {test_loss:.4f}")
    
    # Analyze representations
    print("Analyzing latent representations...")
    
    # Train set analysis
    sim_train = analyze_latent_representations(model, X_train, tokens_train, types_train, 
                                               join(out_dir, 'latent_overlap_train'))
    
    # Test set analysis
    sim_test = analyze_latent_representations(model, X_test, tokens_test, types_test, 
                                              join(out_dir, 'latent_overlap_test'))
    
    # Compute generalization metrics
    print("Computing generalization metrics...")
    
    # Mean similarity within train pairs vs across pairs
    train_within_sim = []
    train_across_sim = []
    for i, type_i in enumerate(types_train):
        for j, type_j in enumerate(types_train):
            if i != j:
                if type_i == type_j:
                    train_within_sim.append(sim_train[i, j])
                else:
                    train_across_sim.append(sim_train[i, j])
    
    train_within_mean = np.mean(train_within_sim)
    train_across_mean = np.mean(train_across_sim)
    
    # Similar for test
    test_within_sim = []
    test_across_sim = []
    for i, type_i in enumerate(types_test):
        for j, type_j in enumerate(types_test):
            if i != j:
                if type_i == type_j:
                    test_within_sim.append(sim_test[i, j])
                else:
                    test_across_sim.append(sim_test[i, j])
    
    test_within_mean = np.mean(test_within_sim)
    test_across_mean = np.mean(test_across_sim)
    
    # Save results
    results = {
        'train_loss_final': train_losses[-1],
        'test_loss': test_loss,
        'train_within_similarity': train_within_mean,
        'train_across_similarity': train_across_mean,
        'test_within_similarity': test_within_mean,
        'test_across_similarity': test_across_mean,
        'train_losses': train_losses
    }
    
    with open(join(out_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print("Results saved to:", out_dir)
    print(f"Train Loss: {train_losses[-1]:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Train Within-Pair Similarity: {train_within_mean:.4f}")
    print(f"Train Across-Pair Similarity: {train_across_mean:.4f}")
    print(f"Test Within-Pair Similarity: {test_within_mean:.4f}")
    print(f"Test Across-Pair Similarity: {test_across_mean:.4f}")
    
    print("Evaluation complete!")

if __name__ == "__main__":
    main()