import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_phase_heatmap_from_csv(df, metric='acc', phase='test', save_dir='results/phase_scan'):
    """画heatmap: 横轴d_hidden，纵轴alpha，色彩为metric (train/test)"""
    metric_map = {'acc': f'{phase}_acc', 'radial': f'{phase}_radial'}
    metric_label = {'acc': 'Accuracy', 'radial': 'Radial Score (P+L+A)/3'}
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
    plot_path = os.path.join(save_dir, f"heatmap_{metric}_{phase}.svg")
    fig.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Heatmap saved to {plot_path}")


def main():
    """Main function to read CSV and plot heatmaps."""
    # Configuration
    csv_file = 'results/phase_scan/phase_scan_results.csv'
    save_dir = 'results/phase_scan'
    
    # Check if CSV exists
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found at {csv_file}")
        print("\nAvailable CSV files in results/phase_scan/:")
        for root, dirs, files in os.walk('results/phase_scan'):
            for file in files:
                if file.endswith('.csv'):
                    print(f"  {os.path.join(root, file)}")
        return
    
    # Read CSV
    print(f"Reading CSV from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Create save directory if needed
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate heatmaps for acc and radial
    print("\nGenerating heatmaps...")
    for metric in ['acc', 'radial']:
        plot_phase_heatmap_from_csv(df, metric=metric, phase='train', save_dir=save_dir)
        plot_phase_heatmap_from_csv(df, metric=metric, phase='test', save_dir=save_dir)
    
    print("\n" + "=" * 80)
    print("PLOTTING COMPLETE!")
    print("=" * 80)
    print(f"All plots saved to: {save_dir}")


if __name__ == '__main__':
    main()
