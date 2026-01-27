import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Import existing functions from test.py
from test import set_seed, generate_instances, sequences_to_tensor, train 
from model import RNN


# ========== EXPERIMENT PARAMETERS ==========
# Fixed parameters
L = 4  # sequence length
m = 2  # number of distinct letters
epochs = 500  # Reduced from 1000 for faster execution
lr = 1e-3
weight_decay = 1e-3
num_layers = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available


# Phase scan参数：扫描alpha
N_SEEDS = 10
SEED_POOL_MAX = 10000  # 从更大范围采样seed
ALPHA_VALUES = [4, 5, 6, 7, 8, 9, 10]
D_HIDDEN_VALUES = [1, 2, 4, 8, 16, 32, 64] 
num_layers = 1

# Output configuration
SAVE_DIR = "results/phase_scan"
os.makedirs(SAVE_DIR, exist_ok=True)


def _run_experiment_wrapper(args):
    """Wrapper function for multiprocessing."""
    alpha, n_hidden, n_latent, n_latent_hidden, n_lat_hid_pow, n_hid_pow, seed = args
    metrics = run_single_experiment(alpha, n_hidden, n_latent, n_latent_hidden, seed)
    return metrics


def run_single_experiment(alpha, n_hidden, n_latent, n_latent_hidden, seed):
    """Run a single experiment with specific parameters and seed."""
    set_seed(seed)
    seq_train, seq_test, labels_train, labels_test, types = generate_instances(alpha, L, m, frac_train=0.8)
    X_train = sequences_to_tensor(seq_train, alpha).to(device)
    X_test = sequences_to_tensor(seq_test, alpha).to(device)
    train_labels_tensor = torch.tensor(labels_train, dtype=torch.long)
    test_labels_tensor = torch.tensor(labels_test, dtype=torch.long)
    model = RNN(
        d_input=alpha,
        d_hidden=n_hidden,
        num_layers=num_layers,
        d_output=alpha,
        output_activation=None,
        nonlinearity='relu',
        device=device
    ).to(device)
    history = train(model, X_train, X_test, test_labels_tensor, train_labels=train_labels_tensor, 
                   types=types, n_epochs=epochs, lr=lr, weight_decay=weight_decay, verbose=False)
    train_acc = history['train_acc'][-1] if history['train_acc'] else np.nan
    test_acc = history['test_acc'][-1] if history['test_acc'] else np.nan
    train_radial = history['train_radial'][-1] if history['train_radial'] else 0.0
    test_radial = history['test_radial'][-1] if history['test_radial'] else 0.0
    train_P = history['train_P'][-1] if history['train_P'] else 0.0
    train_L = history['train_L'][-1] if history['train_L'] else 0.0
    train_A = history['train_A'][-1] if history['train_A'] else 0.0
    test_P = history['test_P'][-1] if history['test_P'] else 0.0
    test_L = history['test_L'][-1] if history['test_L'] else 0.0
    test_A = history['test_A'][-1] if history['test_A'] else 0.0
    return {
        'alpha': alpha,
        'd_hidden': n_hidden,
        'seed': seed,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_radial': train_radial,
        'test_radial': test_radial,
        'train_P': train_P,
        'train_L': train_L,
        'train_A': train_A,
        'test_P': test_P,
        'test_L': test_L,
        'test_A': test_A
    }


def run_phase_scan(alpha_list, n_latent_hidden_powers, n_hidden_powers, n_seeds, n_processes=None):
    """Run the full phase scan experiment with multiprocessing.
    
    Args:
        n_latent: Output latent dimension (d_latent), fixed value
        n_latent_hidden_powers: Powers for intermediate latent dimension (d_latent_hidden)
        n_processes: Number of parallel processes. If None, uses cpu_count()-1.
    """
    # Prepare all experiment configurations
    rng = np.random.default_rng(42)
    seed_choices = rng.choice(SEED_POOL_MAX, size=n_seeds, replace=False)
    experiment_configs = []
    for alpha in alpha_list:
        for d_hidden in D_HIDDEN_VALUES:
            for seed in seed_choices:
                experiment_configs.append((alpha, d_hidden, 0, 0, 0, 0, int(seed)))
    
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


def plot_phase_heatmap(df, metric='acc', phase='test', save_dir=SAVE_DIR):
    """画heatmap: 横轴d_hidden，纵轴alpha，色彩为metric (train/test)"""
    metric_map = {
        'acc': f'{phase}_acc', 
        'radial': f'{phase}_radial',
        'P': f'{phase}_P',
        'L': f'{phase}_L',
        'A': f'{phase}_A'
    }
    metric_label = {
        'acc': 'Accuracy', 
        'radial': 'Radial Score (P+L+A)/3',
        'P': 'Planarity (P)',
        'L': 'Linearity (L)',
        'A': 'Alignment (A)'
    }
    value_col = metric_map[metric]
    # 计算均值
    pivot = df.groupby(['alpha', 'd_hidden'])[value_col].mean().unstack()
    # 计算std
    pivot_std = df.groupby(['alpha', 'd_hidden'])[value_col].std().unstack()
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(pivot.values, cmap='viridis', aspect='auto', vmin=0, vmax=1, origin='lower')
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([str(x) for x in pivot.columns])
    ax.set_yticklabels([str(x) for x in pivot.index])
    ax.set_xlabel('$d_{hidden}$', fontsize=13, fontweight='bold')
    ax.set_ylabel(r'$\alpha$', fontsize=13, fontweight='bold')
    ax.set_title(f'{metric_label[metric]}', fontsize=15, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric_label[metric], fontsize=13)
    # 标注std
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            std_val = pivot_std.values[i, j]
            if not np.isnan(std_val):
                ax.text(j, i, f'±{std_val:.2f}', ha='center', va='center', color='white', fontsize=8, alpha=0.8)
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"phase_scan_{metric}_{phase}.svg")
    fig.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Heatmap saved to {plot_path}")



def main():
    """Main function to run the phase scan experiment."""
    print("=" * 80)
    print("PHASE SCAN EXPERIMENT")
    print("=" * 80)
    print(f"\nParameters:")
    print(f"  L = {L}, m = {m}")
    print(f"  Alpha values: {ALPHA_VALUES}")
    print(f"  Seeds per configuration: {N_SEEDS}")
    print(f"  Epochs: {epochs}")
    print(f"\nResults will be saved to: {SAVE_DIR}")
    print("=" * 80 + "\n")
    
    # Run phase scan
    print("Starting phase scan...")
    results_df = run_phase_scan(ALPHA_VALUES, [], [], N_SEEDS)
    # Save results to CSV
    csv_path = os.path.join(SAVE_DIR, "phase_scan_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to CSV: {csv_path}")
    # 生成一个综合的summary文件
    print("\nGenerating summary statistics...")
    metric_cols = ['train_acc', 'test_acc', 'train_radial', 'test_radial', 'train_P', 'test_P', 'train_L', 'test_L', 'train_A', 'test_A']
    summary_df = results_df.groupby(['alpha', 'd_hidden'])[metric_cols].agg(['mean', 'std'])
    summary_path = os.path.join(SAVE_DIR, "phase_scan_summary.csv")
    summary_df.to_csv(summary_path)
    print(f"Summary saved to {summary_path}")
    
    # 画train/test heatmap
    print("\nGenerating phase scan heatmaps...")
    for metric in ['acc', 'radial', 'P', 'L', 'A']:
        plot_phase_heatmap(results_df, metric=metric, phase='train', save_dir=SAVE_DIR)
        plot_phase_heatmap(results_df, metric=metric, phase='test', save_dir=SAVE_DIR)
    print("\n" + "=" * 80)
    print("PHASE SCAN COMPLETE!")
    print("=" * 80)
    print(f"\nAll results saved to: {SAVE_DIR}")


if __name__ == '__main__':
    main()
