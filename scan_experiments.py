import os
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import numpy as np
import pandas as pd
import torch
import warnings
import random

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
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
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
    if len(w) > 0:
        msgs = [str(x.message) for x in w]
        print(f"Warnings encountered during run (seed={seed_v}): {msgs}")
        return None
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
                # try the provided seed, but if warnings occur, replace with other seeds (up to max attempts)
                tried = set([int(sd)])
                attempts = 0
                max_replacements = 5
                current_seed = int(sd)
                while True:
                    print(f"{exp['name']} -> {scan_key}={sv}, seed={current_seed} (alpha={alpha_v}, L={L_v}, m={m_v})")
                    metrics = _run_single_lma_scan(alpha_v, L_v, d_hidden_v=4, d_latent_hidden_v=2,
                        seed_v=current_seed, n_epochs_v=n_epochs, device_v=device,
                        X_train=X_train, X_test=X_test, test_labels=test_labels)
                    if metrics is None:
                        attempts += 1
                        print(f"Run produced warnings, discarding seed {current_seed} (attempt {attempts}/{max_replacements}).")
                        if attempts >= max_replacements:
                            print("Max replacement attempts reached; skipping this seed.")
                            break
                        # pick a new seed not tried yet
                        new_seed = int(np.random.randint(0, 100000))
                        while new_seed in tried:
                            new_seed = int(np.random.randint(0, 100000))
                        tried.add(new_seed)
                        current_seed = new_seed
                        continue
                    all_results.append({
                        'experiment': exp['name'], 'scan_param': scan_key, 'scan_value': sv, 'seed': current_seed,
                        'alpha': alpha_v, 'L': L_v, 'm': m_v, **metrics,
                    })
                    _safe_write_csv(pd.DataFrame(all_results), checkpoint_path)
                    break

    df_raw = pd.DataFrame(all_results)
    csv_path = _safe_write_csv(df_raw, os.path.join(run_dir, 'three_experiments_results.csv'))
    # remove runs with NaN/inf accuracies (these likely produced warnings during the run)
    invalid_mask = df_raw[['train_acc', 'test_acc']].replace([np.inf, -np.inf], np.nan).isnull().any(axis=1)
    invalid_idxs = df_raw[invalid_mask].index.tolist()
    # try to replace invalid runs by re-running the same configuration with new random seeds
    max_replacements = 5
    for idx in invalid_idxs:
        row = df_raw.loc[idx].to_dict()
        print(f"Invalid run (idx={idx}) encountered: {row}; dropping without replacement.")
    df = df_raw.replace([np.inf, -np.inf], np.nan).dropna(subset=['train_acc', 'test_acc']).reset_index(drop=True)
    dropped = len(df_raw) - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} runs with invalid accuracies (warnings encountered).")

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
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
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
    if len(w) > 0:
        msgs = [str(x.message) for x in w]
        print(f"Warnings encountered during hidden-grid run (seed={seed_v}): {msgs}")
        return None
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
                current_seed = int(seed_v)
                print(
                    f'Run {run_idx}/{total_runs}: '
                    f'L={L_v}, m={m_v}, alpha={alpha_v}, '
                    f'd_hidden={d_hidden_v}, d_latent_hidden={d_latent_hidden_v}, seed={current_seed}'
                )
                metrics = _run_hidden_grid_once(
                    alpha_v, L_v, m_v, d_hidden_v, d_latent_hidden_v, current_seed,
                    n_epochs, X_train, X_test, test_labels,
                )
                if metrics is None:
                    print(f"Run produced warnings, skipping this configuration (seed={current_seed}).")
                    continue
                all_results.append({
                    'L': L_v,
                    'm': m_v,
                    'alpha': alpha_v,
                    'd_hidden': d_hidden_v,
                    'd_latent_hidden': d_latent_hidden_v,
                    'seed': current_seed,
                    **metrics,
                })
                _safe_write_csv(pd.DataFrame(all_results), checkpoint_path)

    df_raw = pd.DataFrame(all_results)
    csv_path = _safe_write_csv(df_raw, os.path.join(run_dir, 'control_624_hidden_scan_results.csv'))
    # try to replace invalid runs by re-running the same configuration with new random seeds
    invalid_mask = df_raw[['train_acc', 'test_acc']].replace([np.inf, -np.inf], np.nan).isnull().any(axis=1)
    invalid_idxs = df_raw[invalid_mask].index.tolist()
    for idx in invalid_idxs:
        row = df_raw.loc[idx].to_dict()
        print(f"Invalid run (idx={idx}) encountered: {row}; dropping without replacement.")
    df = df_raw.replace([np.inf, -np.inf], np.nan).dropna(subset=['train_acc', 'test_acc']).reset_index(drop=True)
    dropped = len(df_raw) - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} runs with invalid accuracies (warnings encountered).")
    summary = df.groupby(['d_latent_hidden', 'd_hidden'])[['train_acc', 'test_acc']].agg(['mean', 'std'])
    summary_path = os.path.join(run_dir, 'control_624_hidden_scan_summary.csv')
    summary.to_csv(summary_path)

    # === counts/proportions by cell for acc==1, acc<=0.1 (zero), and 0.1<acc<1 ===
    def _acc_counts_table(df, phase, zero_thresh=0.1):
        col = f'{phase}_acc'
        records = []
        grp = df.groupby(['d_latent_hidden', 'd_hidden'])
        for (d_latent_hidden, d_hidden), sub in grp:
            total = len(sub)
            if total == 0:
                n_one = n_zero = n_between = 0
            else:
                n_one = int((sub[col] >= 1.0).sum())
                n_zero = int((sub[col] <= zero_thresh).sum())
                n_between = int(((sub[col] > zero_thresh) & (sub[col] < 1.0)).sum())
            records.append({
                'd_latent_hidden': int(d_latent_hidden),
                'd_hidden': int(d_hidden),
                'n_total': int(total),
                'n_one': int(n_one),
                'n_zero': int(n_zero),
                'n_between': int(n_between),
                'p_one': float(n_one / total) if total > 0 else np.nan,
                'p_zero': float(n_zero / total) if total > 0 else np.nan,
                'p_between': float(n_between / total) if total > 0 else np.nan,
            })
        out = pd.DataFrame(records)
        out = out.sort_values(['d_latent_hidden', 'd_hidden']).reset_index(drop=True)
        return out

    counts_train = _acc_counts_table(df, 'train')
    counts_test = _acc_counts_table(df, 'test')
    counts_train_path = os.path.join(run_dir, 'control_624_hidden_scan_counts_train.csv')
    counts_test_path = os.path.join(run_dir, 'control_624_hidden_scan_counts_test.csv')
    _safe_write_csv(counts_train, counts_train_path)
    _safe_write_csv(counts_test, counts_test_path)
    # generate per-seed bar plots for train and test (colored by cividis)
    train_plot_path = _plot_hidden_seedbars(df, phase='train', save_dir=run_dir)
    test_plot_path = _plot_hidden_seedbars(df, phase='test', save_dir=run_dir)
    print(f'Run directory: {run_dir}')
    print(f'Saved raw csv: {csv_path}')
    print(f'Saved summary: {summary_path}')
    print(f'Saved train heatmap: {train_plot_path}')
    print(f'Saved test heatmap: {test_plot_path}')
    return df


def _plot_hidden_seedbars(df_all, phase, save_dir):
    # df_all contains per-seed runs with columns: d_latent_hidden, d_hidden, seed, <phase>_acc
    value_col = f'{phase}_acc'
    dl_vals = sorted(df_all['d_latent_hidden'].unique())
    dh_vals = sorted(df_all['d_hidden'].unique())
    n_rows = len(dl_vals)
    n_cols = len(dh_vals)

    # Determine maximum number of seeds in any cell for consistent bar widths
    cell_counts = df_all.groupby(['d_latent_hidden', 'd_hidden']).size()
    max_per_cell = int(cell_counts.max()) if not cell_counts.empty else 1
    bar_width = 1.0 / max_per_cell * 0.9  # leave small gap

    cmap = plt.get_cmap('cividis')

    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels([str(int(v)) for v in dh_vals])
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels([str(int(v)) for v in dl_vals])
    ax.set_xlabel('d_hidden', fontsize=PLOT_FONT)
    ax.set_ylabel('d_latent_hidden', fontsize=PLOT_FONT)
    ax.set_title(f'{phase.capitalize()}: per-seed accuracies (left→right seeds; heights ∈ [0,1])', fontsize=PLOT_FONT)

    for i, dl in enumerate(dl_vals):
        for j, dh in enumerate(dh_vals):
            sub = df_all[(df_all['d_latent_hidden'] == dl) & (df_all['d_hidden'] == dh)]
            x0 = j
            y0 = i
            # draw cell border
            ax.add_patch(Rectangle((x0, y0), 1, 1, fill=False, edgecolor='gray', linewidth=0.6))
            if sub.empty:
                # no runs
                ax.add_patch(Rectangle((x0 + 0.05, y0 + 0.05), 0.9, 0.9, facecolor='#f0f0f0', edgecolor='none', alpha=0.6))
                continue
            accs = np.array(sub[value_col].dropna().astype(float))
            if accs.size == 0:
                continue
            accs_sorted = np.sort(accs)
            n = accs_sorted.size
            # center the sequence of bars within the cell
            total_width = n * bar_width
            start_x = x0 + 0.5 - total_width / 2
            for k, acc in enumerate(accs_sorted):
                bx = start_x + k * bar_width
                color = cmap(acc)
                # bar height scaled to cell height (0..1)
                height = max(0.001, float(acc)) * 0.95
                ax.add_patch(Rectangle((bx, y0 + 0.02), bar_width, height, facecolor=color, edgecolor='none', alpha=0.95))

    # ensure axes increase left->right and bottom->top
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    fig.tight_layout()
    plot_path = os.path.join(save_dir, f'control_624_{phase}_acc_seedbars.svg')
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path

    


if __name__ == '__main__':
    hidden_grid_heatmap_scan()