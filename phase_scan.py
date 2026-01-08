import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from multiprocessing import Pool, cpu_count
import warnings
import gc

# Import existing functions from test.py
from test import set_seed, generate_instances, sequences_to_tensor, train 
from model import RNNAutoencoder
from analysis_utils import radial_pattern_score


# ========== EXPERIMENT PARAMETERS ==========
# Fixed parameters
L = 4  # sequence length
m = 2  # number of distinct letters
epochs = 500  # Reduced from 1000 for faster execution
lr = 1e-3
weight_decay = 1e-3
num_layers = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available

# Phase scan parameters
N_SEEDS = 5  
N_LATENT = 8
N_LATENT_HIDDEN_POWERS = [2, 3, 4, 5, 6] 
N_HIDDEN_POWERS = [3, 4, 5, 6]  
ALPHA_VALUES = [6] 

# Output configuration
SAVE_DIR = "results/phase_scan"
os.makedirs(SAVE_DIR, exist_ok=True)


def _run_experiment_wrapper(args):
    """Wrapper function for multiprocessing."""
    alpha, n_hidden, n_latent, n_latent_hidden, n_lat_hid_pow, n_hid_pow, seed = args
    metrics = run_single_experiment(alpha, n_hidden, n_latent, n_latent_hidden, seed)
    return {
        'alpha': alpha,
        'n_latent': n_latent,
        'n_latent_hidden': n_latent_hidden,
        'n_hidden': n_hidden,
        'n_latent_hidden_pow': n_lat_hid_pow,
        'n_hidden_pow': n_hid_pow,
        'seed': seed,
        'train_acc': metrics['train_acc'],
        'test_acc': metrics['test_acc'],
        'train_radial': metrics['train_radial'],
        'test_radial': metrics['test_radial'],
        'train_P': metrics['train_P'],
        'train_L': metrics['train_L'],
        'train_A': metrics['train_A'],
        'test_P': metrics['test_P'],
        'test_L': metrics['test_L'],
        'test_A': metrics['test_A'],
    }


def run_single_experiment(alpha, n_hidden, n_latent, n_latent_hidden, seed):
    """Run a single experiment with specific parameters and seed."""
    set_seed(seed)
    
    # Catch warnings to identify problematic parameter combinations
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Generate data using imported function
        seq_train, seq_test, labels_train, labels_test, types = generate_instances(alpha, L, m, frac_train=0.8)
        X_train = sequences_to_tensor(seq_train, alpha).to(device)
        X_test = sequences_to_tensor(seq_test, alpha).to(device)
        train_labels_tensor = torch.tensor(labels_train, dtype=torch.long)
        test_labels_tensor = torch.tensor(labels_test, dtype=torch.long)
        
        # Create model
        model = RNNAutoencoder(alpha, n_hidden, d_latent_hidden=n_latent_hidden, num_layers=num_layers, 
                            d_latent=n_latent, sequence_length=L).to(device)
        
        # Train using imported function (silent mode for phase scan)
        history = train(model, X_train, X_test, test_labels_tensor, train_labels=train_labels_tensor, 
                        types=types, n_epochs=epochs, lr=lr, weight_decay=weight_decay, verbose=False)
        
        # Extract metrics from history (radial scores already computed during training)
        train_acc = history['train_acc'][-1] if history['train_acc'] else np.nan
        test_acc = history['test_acc'][-1] if history['test_acc'] else np.nan
        train_radial = history['train_radial'][-1] if history['train_radial'] else 0.0
        test_radial = history['test_radial'][-1] if history['test_radial'] else 0.0
        
        # Get P, L, A components from history if available
        train_P = history['train_P'][-1]
        train_L = history['train_L'][-1]
        train_A = history['train_A'][-1]
        test_P = history['test_P'][-1]
        test_L = history['test_L'][-1]
        test_A = history['test_A'][-1]
        
        # Clean up to free memory
        del model
        del X_train, X_test
        del train_labels_tensor, test_labels_tensor
        del history
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Force garbage collection to prevent memory accumulation
        gc.collect()
        
        # Check if any warnings occurred
        if len(w) > 0:
            for warning in w:
                if "invalid value encountered in divide" in str(warning.message):
                    print(f" Warning at n_hidden={n_hidden}, n_latent={n_latent}, seed={seed}: {warning.message}")
                    break
    
    return {
        'train_acc': train_acc, 'test_acc': test_acc, 
        'train_radial': train_radial, 'test_radial': test_radial,
        'train_P': train_P, 'train_L': train_L, 'train_A': train_A,
        'test_P': test_P, 'test_L': test_L, 'test_A': test_A
    }


def run_phase_scan(alpha_list, n_latent_hidden_powers, n_hidden_powers, n_seeds, n_latent=N_LATENT, n_processes=None):
    """Run the full phase scan experiment with multiprocessing.
    
    Args:
        n_latent: Output latent dimension (d_latent), fixed value
        n_latent_hidden_powers: Powers for intermediate latent dimension (d_latent_hidden)
        n_processes: Number of parallel processes. If None, uses cpu_count()-1.
    """
    # Prepare all experiment configurations
    experiment_configs = []
    for alpha in alpha_list:
        for n_lat_hid_pow in n_latent_hidden_powers:
            n_latent_hidden = 2 ** n_lat_hid_pow
            for n_hid_pow in n_hidden_powers:
                n_hidden = 2 ** n_hid_pow
                for seed in range(n_seeds):
                    experiment_configs.append((alpha, n_hidden, n_latent, n_latent_hidden, n_lat_hid_pow, n_hid_pow, seed))
    
    # Determine number of processes
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)  # Leave one core free
    
    print(f"Running {len(experiment_configs)} experiments using {n_processes} parallel processes...")
    
    # Run experiments in parallel
    results = []
    with Pool(processes=n_processes, maxtasksperchild=10) as pool:
        # Use imap_unordered for progress bar
        with tqdm(total=len(experiment_configs), desc="Phase scan") as pbar:
            for result in pool.imap_unordered(_run_experiment_wrapper, experiment_configs):
                results.append(result)
                pbar.update(1)
    
    return pd.DataFrame(results)


def plot_phase_diagram(df, alpha, phase='test', save_dir=SAVE_DIR):
    """Plot phase diagram with P, L, A components and average radial score.
    
    Note: Plots n_hidden vs n_latent_hidden (not n_latent which is fixed).
    """
    # Filter for specific alpha
    df_alpha = df[df['alpha'] == alpha].copy()
    
    # Create figure with 4 subplots: P, L, A, Average
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    metrics = [f'{phase}_P', f'{phase}_L', f'{phase}_A', f'{phase}_radial']
    metric_labels = {
        f'{phase}_P': 'Planarity (P)',
        f'{phase}_L': 'Linearity (L)',
        f'{phase}_A': 'Alignment (A)',
        f'{phase}_radial': 'Average (P+L+A)/3',
    }
    metric_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green for P, L, A
    
    # Find global vmin and vmax for P, L, A, Average to keep colorbar consistent
    vmin_radial = 0.0
    vmax_radial = 1.0
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Compute mean and std across seeds
        grouped = df_alpha.groupby(['n_latent_hidden_pow', 'n_hidden_pow'])[metric].agg(['mean', 'std'])
        grouped = grouped.reset_index()
        
        # Create pivot table for heatmap
        pivot = grouped.pivot(index='n_hidden_pow', columns='n_latent_hidden_pow', values='mean')
        pivot_std = grouped.pivot(index='n_hidden_pow', columns='n_latent_hidden_pow', values='std')
        
        # Set vmin and vmax to keep colorbar consistent
        vmin, vmax = vmin_radial, vmax_radial
        
        # Create heatmap
        im = ax.imshow(pivot.values, cmap='viridis', aspect='auto', origin='lower', 
                       vmin=vmin, vmax=vmax)
        
        # Set ticks and labels
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels([f'{2**int(p)}' for p in pivot.columns])
        ax.set_yticklabels([f'{2**int(p)}' for p in pivot.index])
        
        ax.set_xlabel('$n_{latent_hidden}$', fontsize=14, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('$n_{hidden}$', fontsize=14, fontweight='bold')
        
        # Color the title based on component
        title_color = 'black'
        if idx < 3:
            title_color = metric_colors[idx]
        ax.set_title(f'{metric_labels[metric]}', fontsize=14, fontweight='bold', color=title_color)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(metric_labels[metric], fontsize=14)
        
        # Annotate with std only (mean is shown by color)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                std_val = pivot_std.values[i, j]
                if not np.isnan(std_val):
                    text = f'±{std_val:.3f}'
                    ax.text(j, i, text, ha='center', va='center', 
                           color='white', fontsize=6, alpha=0.8)
    
    plt.tight_layout()
    
    # Save
    filename = f'phase_diagram_{phase}_alpha{alpha}.png'
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"    {phase.capitalize()} phase diagram saved to: {filepath}")
    
    # Also save as SVG
    filepath_svg = os.path.join(save_dir, filename.replace('.png', '.svg'))
    fig.savefig(filepath_svg, bbox_inches='tight')
    
    plt.close(fig)



def main():
    """Main function to run the phase scan experiment."""
    print("=" * 80)
    print("PHASE SCAN EXPERIMENT")
    print("=" * 80)
    print(f"\nParameters:")
    print(f"  L = {L}, m = {m}")
    print(f"  Alpha values: {ALPHA_VALUES}")
    print(f"  n_latent (fixed): {N_LATENT}")
    print(f"  n_latent_hidden powers: {N_LATENT_HIDDEN_POWERS} -> sizes: {[2**p for p in N_LATENT_HIDDEN_POWERS]}")
    print(f"  n_hidden powers: {N_HIDDEN_POWERS} -> sizes: {[2**p for p in N_HIDDEN_POWERS]}")
    print(f"  Seeds per configuration: {N_SEEDS}")
    print(f"  Epochs: {epochs}")
    print(f"  Total runs: {len(ALPHA_VALUES) * len(N_LATENT_HIDDEN_POWERS) * len(N_HIDDEN_POWERS) * N_SEEDS}")
    print(f"\nResults will be saved to: {SAVE_DIR}")
    print("=" * 80 + "\n")
    
    # Run phase scan
    print("Starting phase scan...")
    results_df = run_phase_scan(ALPHA_VALUES, N_LATENT_HIDDEN_POWERS, N_HIDDEN_POWERS, N_SEEDS)
    
    # Save results to CSV
    csv_path = os.path.join(SAVE_DIR, "phase_scan_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to CSV: {csv_path}")
    
    # Generate plots for each alpha
    print("\nGenerating phase diagrams...")
    for alpha in ALPHA_VALUES:
        print(f"\n  Processing alpha = {alpha}...")
        
        # Phase diagram for train set
        plot_phase_diagram(results_df, alpha, phase='train', save_dir=SAVE_DIR)
        
        # Phase diagram for test set
        plot_phase_diagram(results_df, alpha, phase='test', save_dir=SAVE_DIR)
        
    
    print("\n" + "=" * 80)
    print("PHASE SCAN COMPLETE!")
    print("=" * 80)
    print(f"\nAll results saved to: {SAVE_DIR}")


if __name__ == '__main__':
    main()
