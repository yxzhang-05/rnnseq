import random
import numpy as np
import torch
import torch.nn.functional as F
from model import RNNAutoencoder
from sequences import findStructures, replace_symbols as seq_replace_symbols
import string
import itertools
import os
import pandas as pd
import matplotlib.pyplot as plt
from analysis_utils import radial_pattern_score


# Fixed parameters
L, m, alpha = 4, 2, 6
epochs = 1000
lr = 1e-3
d_hidden = 8
d_latent_hidden = 4
weight_decay = 1e-3
num_layers = 1
device = torch.device('cpu')

# Scan parameters
d_latent_values = [1, 2, 4, 8, 16, 32, 64]
seeds = list(range(10))  # 10 different seeds

SAVE_DIR = "results/latent_dim_scan"
os.makedirs(SAVE_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def generate_instances(alpha, L, m, frac_train=0.8):
    alphabet = list(string.ascii_lowercase[:alpha])
    types = sum(findStructures(alphabet, L, m), [])
    all_perms = list(itertools.permutations(alphabet, m))

    train_seqs, test_seqs = [], []
    train_labels, test_labels = [], []

    for type_idx, t in enumerate(types):
        seqs = [seq_replace_symbols(t, perm) for perm in all_perms]
        n = len(seqs)
        perm_idx = np.random.permutation(n)
        split = int(frac_train * n)

        train_idx = perm_idx[:split]
        test_idx = perm_idx[split:]

        train_seqs.extend([seqs[i] for i in train_idx])
        train_labels.extend([type_idx] * len(train_idx))

        test_seqs.extend([seqs[i] for i in test_idx])
        test_labels.extend([type_idx] * len(test_idx))

    return (np.array(train_seqs), np.array(test_seqs), 
            np.array(train_labels), np.array(test_labels), types)


def sequences_to_tensor(sequences, alpha):
    letter_to_idx = {l: i for i, l in enumerate(string.ascii_lowercase[:alpha])}
    one_hot = []
    for seq in sequences:
        seq_onehot = []
        for c in seq:
            vec = [0] * alpha
            vec[letter_to_idx[c]] = 1
            seq_onehot.append(vec)
        one_hot.append(seq_onehot)
    one_hot = torch.tensor(one_hot, dtype=torch.float)
    return one_hot.permute(1, 0, 2)


def train_model(model, X_train, X_test, test_labels, train_labels, types, n_epochs, lr, weight_decay):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        hidden, latent, output = model(X_train)
        ce_loss = F.cross_entropy(
            output.reshape(-1, output.shape[-1]),
            torch.argmax(X_train, dim=-1).reshape(-1)
        )
        ce_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # Final evaluation
    model.eval()
    with torch.no_grad():
        _, train_latent, train_output = model(X_train)
        pred_train = torch.argmax(train_output, dim=-1)
        target_train = torch.argmax(X_train, dim=-1)
        train_acc = (pred_train == target_train).all(dim=0).float().mean().item()

        _, test_latent, test_output = model(X_test)
        pred_test = torch.argmax(test_output, dim=-1)
        target_test = torch.argmax(X_test, dim=-1)
        test_acc = (pred_test == target_test).all(dim=0).float().mean().item()

        # Compute radial scores
        if train_latent.ndim == 3:
            train_latent_2d = train_latent.permute(1, 0, 2).reshape(train_latent.shape[1], -1)
            test_latent_2d = test_latent.permute(1, 0, 2).reshape(test_latent.shape[1], -1)
        else:
            train_latent_2d = train_latent
            test_latent_2d = test_latent

        train_latent_np = train_latent_2d.cpu().numpy()
        test_latent_np = test_latent_2d.cpu().numpy()
        train_labels_np = train_labels.cpu().numpy()
        test_labels_np = test_labels.cpu().numpy()

        train_radial, train_P, train_L, train_A = radial_pattern_score(
            train_latent_np, train_labels_np, types, return_components=True)
        test_radial, test_P, test_L, test_A = radial_pattern_score(
            test_latent_np, test_labels_np, types, return_components=True)

    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_radial': train_radial,
        'test_radial': test_radial,
        'train_P': train_P,
        'train_L': train_L,
        'train_A': train_A,
        'test_P': test_P,
        'test_L': test_L,
        'test_A': test_A,
    }


def run_scan():
    results = []

    total_runs = len(d_latent_values) * len(seeds)
    current_run = 0

    for d_latent in d_latent_values:
        for seed in seeds:
            current_run += 1
            print(f"\n{'='*60}")
            print(f"Run {current_run}/{total_runs}: d_latent={d_latent}, seed={seed}")
            print(f"{'='*60}")

            set_seed(seed)

            # Generate data
            seq_train, seq_test, labels_train, labels_test, types = generate_instances(
                alpha, L, m, frac_train=0.8)
            X_train = sequences_to_tensor(seq_train, alpha).to(device)
            X_test = sequences_to_tensor(seq_test, alpha).to(device)
            train_labels = torch.tensor(labels_train, dtype=torch.long)
            test_labels = torch.tensor(labels_test, dtype=torch.long)

            # Create model
            model = RNNAutoencoder(
                alpha, d_hidden, d_latent_hidden, num_layers, d_latent, L
            ).to(device)

            # Train and evaluate
            metrics = train_model(
                model, X_train, X_test, test_labels, train_labels, types,
                n_epochs=epochs, lr=lr, weight_decay=weight_decay
            )

            # Store results
            result = {
                'd_latent': d_latent,
                'seed': seed,
                **metrics
            }
            results.append(result)

            print(f"  Train Acc: {metrics['train_acc']:.4f}, Test Acc: {metrics['test_acc']:.4f}")
            print(f"  Train Radial: {metrics['train_radial']:.4f}, Test Radial: {metrics['test_radial']:.4f}")

    # Save results
    df = pd.DataFrame(results)
    csv_path = os.path.join(SAVE_DIR, "scan_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n\nResults saved to {csv_path}")

    return df


def plot_results(df):
    """Plot results with error bars for both train and test"""
    
    # Metrics to plot: acc, radial and its three components
    metrics_to_plot = [
        ('acc', 'Accuracy'),
        ('radial', 'Radial Score (P+L+A)/3'),
        ('P', 'Planarity (P)'),
        ('L', 'Linearity (L)'),
        ('A', 'Alignment (A)'),
    ]
    
    # Colors matching phase_scan.py
    metric_colors = {
        'acc': '#2c3e50',  # Dark blue-gray for accuracy
        'P': '#1f77b4',  # Blue
        'L': '#ff7f0e',  # Orange
        'A': '#2ca02c',  # Green
        'radial': '#9467bd'  # Purple for average
    }

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for idx, (metric, label) in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        # Plot both train and test with distinct styles
        for phase, linestyle, marker, alpha_val in [
            ('train', '-', 'o', 0.7), 
            ('test', '--', 's', 0.9)
        ]:
            full_metric = f'{phase}_{metric}'
            grouped = df.groupby('d_latent')[full_metric]
            means = grouped.mean()
            stds = grouped.std()
            
            d_latent_vals = means.index.values
            
            # Use metric-specific color
            color = metric_colors.get(metric, '#333333')
            
            # Plot with error bars
            ax.errorbar(d_latent_vals, means.values, yerr=stds.values, 
                       marker=marker, linewidth=2.5, markersize=8, capsize=5, capthick=2,
                       color=color, label=phase.capitalize(), alpha=alpha_val,
                       linestyle=linestyle)
        
        ax.set_xlabel('$d_{latent}$', fontsize=13, fontweight='bold')
        ax.set_ylabel(label.split('(')[0].strip(), fontsize=12)
        ax.set_title(label, fontsize=13, fontweight='bold', 
                    color=metric_colors.get(metric, 'black'))
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        ax.set_xscale('log', base=2)
        ax.set_xticks(d_latent_vals)
        ax.set_xticklabels([str(int(x)) for x in d_latent_vals])
        ax.legend(loc='best', fontsize=10, frameon=True, fancybox=True, shadow=True)
        ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plot_path = os.path.join(SAVE_DIR, "scan_results.svg")
    fig.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    print(f"Plots saved to {plot_path}")

    # Also create a summary table for both train and test
    summary_metrics = ['train_acc', 'test_acc', 'train_radial', 'test_radial', 
                      'train_P', 'test_P', 'train_L', 'test_L', 'train_A', 'test_A']
    summary = df.groupby('d_latent')[summary_metrics].agg(['mean', 'std'])
    summary_path = os.path.join(SAVE_DIR, "summary.csv")
    summary.to_csv(summary_path)
    print(f"Summary saved to {summary_path}")
    print("\nSummary Statistics:")
    print(summary)


if __name__ == '__main__':
    print("Starting latent dimension scan...")
    print(f"d_latent values: {d_latent_values}")
    print(f"Seeds: {seeds}")
    print(f"Total runs: {len(d_latent_values) * len(seeds)}")
    
    df = run_scan()
    plot_results(df)
    
    print("\n\nScan complete!")
