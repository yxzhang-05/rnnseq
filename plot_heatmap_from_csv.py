import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_phase_diagram_from_csv(df, alpha, phase='test', save_dir='results/phase_scan'):
    """Plot phase diagram with P, L, A components and average radial score from CSV data."""
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
        grouped = df_alpha.groupby(['n_latent_pow', 'n_hidden_pow'])[metric].agg(['mean', 'std'])
        grouped = grouped.reset_index()
        
        # Create pivot table for heatmap
        pivot = grouped.pivot(index='n_hidden_pow', columns='n_latent_pow', values='mean')
        pivot_std = grouped.pivot(index='n_hidden_pow', columns='n_latent_pow', values='std')
        
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
        
        ax.set_xlabel('$n_{latent}$', fontsize=14, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('$n_{hidden}$', fontsize=14, fontweight='bold')
        
        # Color the title based on component
        title_color = 'black'
        if idx < 3:
            title_color = metric_colors[idx]
        ax.set_title(f'{metric_labels[metric]}', fontsize=14, fontweight='bold', color=title_color)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
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
    filename = f'{phase}_alpha{alpha}.png'
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  {phase.capitalize()} phase diagram saved to: {filepath}")
    
    # Also save as SVG
    filepath_svg = os.path.join(save_dir, filename.replace('.png', '.svg'))
    fig.savefig(filepath_svg, bbox_inches='tight')
    
    plt.close(fig)


def main():
    """Main function to read CSV and plot heatmaps."""
    # Configuration
    csv_file = 'results/phase_scan/alpha_10/phase_scan_results.csv'
    save_dir = 'results/phase_scan/alpha_10'
    
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
    
    # Get unique alpha values
    alpha_values = df['alpha'].unique()
    print(f"Alpha values found: {alpha_values}")
    
    # Create save directory if needed
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate plots for each alpha
    print("\nGenerating phase diagrams...")
    for alpha in alpha_values:
        print(f"\nProcessing alpha = {alpha}...")
        
        # Phase diagram for train set
        plot_phase_diagram_from_csv(df, alpha, phase='train', save_dir=save_dir)
        
        # Phase diagram for test set
        plot_phase_diagram_from_csv(df, alpha, phase='test', save_dir=save_dir)
    
    print("\n" + "=" * 80)
    print("PLOTTING COMPLETE!")
    print("=" * 80)
    print(f"All plots saved to: {save_dir}")


if __name__ == '__main__':
    main()
