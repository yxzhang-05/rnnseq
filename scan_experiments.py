import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from model import RNNAutoencoder
from test import (
    SAVE_DIR,
    device,
    d_latent,
    epochs,
    generate_instances,
    lr,
    num_layers,
    sequences_to_tensor,
    set_seed,
    train,
    weight_decay,
)


PLOT_FONT = 13
SCAN_SPLIT_SEED = 2024
LMA_SCAN_SEEDS = list(range(40, 51))
CONTROL_TASK = {'L': 6, 'm': 2, 'alpha': 4}
HIDDEN_VALUES = [2, 4, 8, 16, 32]
HIDDEN_GRID_SEEDS = list(range(10))


def _make_run_dir(subdir):
    base_dir = os.path.join(SAVE_DIR, subdir)
    os.makedirs(base_dir, exist_ok=True)
    run_dir = os.path.join(base_dir, datetime.now().strftime('run_%Y%m%d_%H%M%S'))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _safe_write_csv(df, path):
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        root, ext = os.path.splitext(path)
        fallback_path = f"{root}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        df.to_csv(fallback_path, index=False)
        print(f"Permission denied for {path}; saved to {fallback_path} instead.")
        return fallback_path


def _build_fixed_scan_data(alpha_v, L_v, m_v, split_seed_v, device_v):
    set_seed(split_seed_v)
    seq_train, seq_test, _, labels_test, _ = generate_instances(alpha_v, L_v, m_v, frac_train=0.8)
    X_train = sequences_to_tensor(seq_train, alpha_v).to(device_v)
    X_test = sequences_to_tensor(seq_test, alpha_v).to(device_v)
    test_labels = torch.tensor(labels_test, dtype=torch.long)
    return X_train, X_test, test_labels


def _run_single_lma_scan(alpha_v, L_v, d_hidden_v, d_latent_hidden_v, seed_v, n_epochs_v, device_v, X_train, X_test, test_labels):
    set_seed(seed_v)
    model = RNNAutoencoder(alpha_v, d_hidden_v, d_latent_hidden_v, num_layers, d_latent, L_v).to(device_v)
    history = train(
        model,
        X_train,
        X_test,
        test_labels,
        train_labels=None,
        types=None,
        n_epochs=n_epochs_v,
        lr=lr,
        weight_decay=weight_decay,
        print_final=False,
    )
    return {
        'train_acc': float(history['train_acc'][-1]),
        'test_acc': float(history['test_acc'][-1]),
    }


def _plot_curve_with_std(ax, x_vals, means, stds, color, label):
    x = np.array(x_vals, dtype=float)
    y = np.array(means, dtype=float)
    s = np.array(stds, dtype=float)
    ax.plot(x, y, color=color, linewidth=2.2, marker='o', markersize=6, label=label)
    ax.fill_between(x, y - s, y + s, color=color, alpha=0.22, linewidth=0)


def L_m_alpha_scan(save_dir=None, n_epochs=None):
    run_dir = _make_run_dir('l_m_alpha_scan') if save_dir is None else save_dir
    os.makedirs(run_dir, exist_ok=True)
    n_epochs = epochs if n_epochs is None else n_epochs
    checkpoint_path = os.path.join(run_dir, 'three_experiments_results_checkpoint.csv')

    experiments = [
        {'name': 'Experiment 1', 'scan': 'alpha', 'values': [2, 4, 6, 8, 10], 'fixed': {'L': 4, 'm': 2}, 'seeds': LMA_SCAN_SEEDS},
        {'name': 'Experiment 2', 'scan': 'L', 'values': [4, 5, 6, 7, 8], 'fixed': {'alpha': 6, 'm': 2}, 'seeds': LMA_SCAN_SEEDS},
        {'name': 'Experiment 3', 'scan': 'm', 'values': [1, 2, 3], 'fixed': {'alpha': 6, 'L': 6}, 'seeds': LMA_SCAN_SEEDS},
    ]

    all_results = []
    for exp in experiments:
        scan_key, vals, seeds = exp['scan'], exp['values'], exp['seeds']
        print('\n' + '=' * 70)
        print(f"{exp['name']} | fixed={exp['fixed']} | scan {scan_key}={vals} | seeds={seeds}")
        print('=' * 70)
        for sv in vals:
            params = dict(exp['fixed'])
            params[scan_key] = sv
            alpha_v, L_v, m_v = params['alpha'], params['L'], params['m']
            X_train, X_test, test_labels = _build_fixed_scan_data(alpha_v, L_v, m_v, SCAN_SPLIT_SEED, device)
            for sd in seeds:
                print(f"{exp['name']} -> {scan_key}={sv}, seed={sd} (alpha={alpha_v}, L={L_v}, m={m_v})")
                metrics = _run_single_lma_scan(alpha_v, L_v, d_hidden_v=4, d_latent_hidden_v=2,
                    seed_v=sd, n_epochs_v=n_epochs, device_v=device,
                    X_train=X_train, X_test=X_test, test_labels=test_labels)
                all_results.append({
                    'experiment': exp['name'], 'scan_param': scan_key, 'scan_value': sv, 'seed': sd,
                    'alpha': alpha_v, 'L': L_v, 'm': m_v, **metrics,
                })
                _safe_write_csv(pd.DataFrame(all_results), checkpoint_path)

    df = pd.DataFrame(all_results)
    csv_path = _safe_write_csv(df, os.path.join(run_dir, 'three_experiments_results.csv'))

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3), constrained_layout=True)
    colors = {'train': '#1f77b4', 'test': '#ff7f0e'}
    for ax, exp in zip(axes, experiments):
        sub = df[df['experiment'] == exp['name']]
        x_vals = exp['values']
        train_stats = sub.groupby('scan_value')['train_acc'].agg(['mean', 'std']).reindex(x_vals)
        test_stats = sub.groupby('scan_value')['test_acc'].agg(['mean', 'std']).reindex(x_vals)
        _plot_curve_with_std(ax, x_vals, train_stats['mean'].values, train_stats['std'].fillna(0).values, colors['train'], 'Train Acc')
        _plot_curve_with_std(ax, x_vals, test_stats['mean'].values, test_stats['std'].fillna(0).values, colors['test'], 'Test Acc')
        ax.set_title(exp['name'])
        ax.set_xlabel(exp['scan'])
        ax.set_xticks(x_vals)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.28, linestyle=':')

    axes[0].set_ylabel('Accuracy')
    axes[-1].legend(loc='lower right', fontsize=10, frameon=True)
    fig_path = os.path.join(run_dir, 'three_experiments_acc.svg')
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    print(f"Run directory: {run_dir}")
    print(f"Saved csv: {csv_path}")
    print(f"Saved plot: {fig_path}")
    return df


def _run_hidden_grid_once(alpha_v, L_v, m_v, d_hidden_v, d_latent_hidden_v, seed_v, n_epochs_v, X_train, X_test, test_labels):
    set_seed(seed_v)
    model = RNNAutoencoder(alpha_v, d_hidden_v, d_latent_hidden_v, num_layers, d_latent, L_v).to(device)
    history = train(
        model,
        X_train,
        X_test,
        test_labels,
        train_labels=None,
        types=None,
        n_epochs=n_epochs_v,
        lr=lr,
        weight_decay=weight_decay,
        print_final=False,
    )
    return {
        'train_acc': float(history['train_acc'][-1]),
        'test_acc': float(history['test_acc'][-1]),
    }


def _plot_hidden_heatmap(df, phase, save_dir):
    value_col = f'{phase}_acc'
    mean_table = df.groupby(['d_latent_hidden', 'd_hidden'])[value_col].mean().unstack()
    std_table = df.groupby(['d_latent_hidden', 'd_hidden'])[value_col].std(ddof=0).unstack()

    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    im = ax.imshow(mean_table.values, cmap='viridis', aspect='auto', origin='lower', vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(mean_table.columns)))
    ax.set_xticklabels([str(int(v)) for v in mean_table.columns])
    ax.set_yticks(np.arange(len(mean_table.index)))
    ax.set_yticklabels([str(int(v)) for v in mean_table.index])
    ax.set_xlabel('d_hidden', fontsize=PLOT_FONT)
    ax.set_ylabel('d_latent_hidden', fontsize=PLOT_FONT)
    ax.set_title(f'{phase.capitalize()} Accuracy', fontsize=PLOT_FONT)

    for i in range(len(mean_table.index)):
        for j in range(len(mean_table.columns)):
            mean_val = mean_table.iloc[i, j]
            std_val = std_table.iloc[i, j]
            if np.isnan(mean_val):
                continue
            text_color = 'white' if mean_val < 0.55 else 'black'
            ax.text(j, i, f'{mean_val:.2f}\n±{0.0 if np.isnan(std_val) else std_val:.2f}',
                ha='center', va='center', color=text_color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f'{phase.capitalize()} accuracy', fontsize=PLOT_FONT)
    fig.tight_layout()
    plot_path = os.path.join(save_dir, f'control_624_{phase}_acc_heatmap.svg')
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def hidden_grid_heatmap_scan(save_dir=None, n_epochs=None):
    run_dir = _make_run_dir('control_624_hidden_heatmap') if save_dir is None else save_dir
    os.makedirs(run_dir, exist_ok=True)
    n_epochs = epochs if n_epochs is None else n_epochs
    checkpoint_path = os.path.join(run_dir, 'control_624_hidden_scan_checkpoint.csv')

    alpha_v = CONTROL_TASK['alpha']
    L_v = CONTROL_TASK['L']
    m_v = CONTROL_TASK['m']
    X_train, X_test, test_labels = _build_fixed_scan_data(alpha_v, L_v, m_v, SCAN_SPLIT_SEED, device)

    all_results = []
    total_runs = len(HIDDEN_VALUES) * len(HIDDEN_VALUES) * len(HIDDEN_GRID_SEEDS)
    run_idx = 0
    for d_hidden_v in HIDDEN_VALUES:
        for d_latent_hidden_v in HIDDEN_VALUES:
            for seed_v in HIDDEN_GRID_SEEDS:
                run_idx += 1
                print(
                    f'Run {run_idx}/{total_runs}: '
                    f'L={L_v}, m={m_v}, alpha={alpha_v}, '
                    f'd_hidden={d_hidden_v}, d_latent_hidden={d_latent_hidden_v}, seed={seed_v}'
                )
                metrics = _run_hidden_grid_once(
                    alpha_v, L_v, m_v, d_hidden_v, d_latent_hidden_v, seed_v,
                    n_epochs, X_train, X_test, test_labels,
                )
                all_results.append({
                    'L': L_v,
                    'm': m_v,
                    'alpha': alpha_v,
                    'd_hidden': d_hidden_v,
                    'd_latent_hidden': d_latent_hidden_v,
                    'seed': seed_v,
                    **metrics,
                })
                _safe_write_csv(pd.DataFrame(all_results), checkpoint_path)

    df = pd.DataFrame(all_results)
    csv_path = _safe_write_csv(df, os.path.join(run_dir, 'control_624_hidden_scan_results.csv'))
    summary = df.groupby(['d_latent_hidden', 'd_hidden'])[['train_acc', 'test_acc']].agg(['mean', 'std'])
    summary_path = os.path.join(run_dir, 'control_624_hidden_scan_summary.csv')
    summary.to_csv(summary_path)

    train_plot_path = _plot_hidden_heatmap(df, phase='train', save_dir=run_dir)
    test_plot_path = _plot_hidden_heatmap(df, phase='test', save_dir=run_dir)

    print(f'Run directory: {run_dir}')
    print(f'Saved raw csv: {csv_path}')
    print(f'Saved summary: {summary_path}')
    print(f'Saved train heatmap: {train_plot_path}')
    print(f'Saved test heatmap: {test_plot_path}')
    return df


if __name__ == '__main__':
    hidden_grid_heatmap_scan()