import random
import collections
import numpy as np
import torch
import torch.nn.functional as F
from model import RNN
from sequences import findStructures, replace_symbols as seq_replace_symbols
import string
import itertools
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import warnings
from analysis_utils import radial_pattern_score
  

seed = 42
L, m, alpha = 4, 2, 6
epochs = 1000 
lr = 1e-3
d_hidden = 4
weight_decay = 1e-3 
num_layers = 1
device = torch.device('cpu')
SAVE_DIR = "results"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def generate_instances(alpha, L, m, frac_train=0.8):
    alphabet = list(string.ascii_lowercase[:alpha])
    types = sum(findStructures(alphabet, L, m), [])  # flatten
    all_perms = list(itertools.permutations(alphabet, m))

    train_seqs, test_seqs = [], []
    train_labels, test_labels = [], []

    for type_idx, t in enumerate(types):
        # generate sequences for this type
        seqs = [seq_replace_symbols(t, perm) for perm in all_perms]

        # shuffle and split
        n = len(seqs)
        perm_idx = np.random.permutation(n)
        split = int(frac_train * n)

        train_idx = perm_idx[:split]
        test_idx  = perm_idx[split:]

        # append train
        train_seqs.extend([seqs[i] for i in train_idx])
        train_labels.extend([type_idx] * len(train_idx))

        # append test
        test_seqs.extend([seqs[i] for i in test_idx])
        test_labels.extend([type_idx] * len(test_idx))

    return (np.array(train_seqs), np.array(test_seqs), np.array(train_labels), np.array(test_labels), types)


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


def plot_diagnostics(model, X_train, train_labels, X_test, test_labels, types, save_dir=SAVE_DIR, history=None, seq_train=None, seq_test=None):

    os.makedirs(save_dir, exist_ok=True)

    def _safe_numpy(t):
        return t.cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t)

    model.eval()
    with torch.no_grad():
        h_tr, out_tr = model(X_train)
        h_te, out_te = model(X_test)

    datasets = [
        (h_tr, train_labels, 'train', seq_train),
        (h_te, test_labels, 'test', seq_test)
    ]

    for hidden, labels, phase, seqs in datasets:
        labels_np = _safe_numpy(labels)
        hidden_np = hidden.cpu().numpy() if hidden is not None else None
        if hidden_np.ndim == 3:
            z_flat = hidden_np.transpose(1, 0, 2).reshape(hidden_np.shape[1], -1)
        else:
            z_flat = hidden_np

        # Extract letter combinations
        letter_combos = [''.join(sorted(set(seq))) for seq in seqs]

        # =======================================================
        # 1. Latent PCA Spectrum (Global) - Uses z_flat
        # =======================================================
        try:
            # z_flat: (N, T*d)
            if hidden_np.ndim == 3:
                z_flat = hidden_np.transpose(1, 0, 2).reshape(hidden_np.shape[1], -1)
            else:
                z_flat = hidden_np

            # Check variance and warn if low
            data_std = np.std(z_flat)
            if data_std < 1e-6:
                print(f"WARNING: Low variance in latent space ({phase}): std={data_std:.2e}")
                print(f"PCA may produce unreliable results. Consider training longer.")

            n_comp = min(z_flat.shape[1], z_flat.shape[0] - 1, 10)
            n_comp = max(1, n_comp)

            with warnings.catch_warnings():
                warnings.filterwarnings('always', category=RuntimeWarning)
                pca_lat = PCA(n_components=n_comp)
                pca_lat.fit(z_flat)
            evr_lat = pca_lat.explained_variance_ratio_

            fig, ax = plt.subplots(figsize=(3, 3))
            ax.bar(range(1, len(evr_lat)+1), evr_lat, color='C1')
            ax.set_xlabel('PC')
            ax.set_ylabel('Explained variance ratio')
            ax.set_title(f'{phase.capitalize()}')
            fname = os.path.join(save_dir, f"latent_pca_spectrum_{phase}.svg")
            fig.savefig(fname, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"Latent PCA spectrum failed ({phase}): {e}")

        # # =======================================================
        # # 2. Latent Similarity Heatmap - Uses z_flat
        # # =======================================================
        # try:
        #     # Center the latent representations
        #     z_centered = z_flat - z_flat.mean(axis=0, keepdims=True)

        #     # Sort only by type 
        #     order = []
        #     for type_idx in range(len(types)):
        #         type_mask = (labels_np == type_idx)
        #         type_indices = np.where(type_mask)[0]
        #         order.extend(type_indices)
        #     order = np.array(order)

        #     sim = 1 - squareform(pdist(z_centered, metric='cosine'))
        #     sim_ord = sim[np.ix_(order, order)]

        #     fig, ax = plt.subplots(figsize=(10, 10))
        #     im = ax.imshow(sim_ord, cmap='viridis', vmin=-1, vmax=1, aspect='equal')
        #     ax.set_title(f'Global Similarity Heatmap ({phase})', fontsize=12)
        #     fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        #     # Calculate positions for each type
        #     type_sample_counts = []
        #     for type_idx in range(len(types)):
        #         type_mask = (labels_np == type_idx)
        #         n_samples = type_mask.sum()
        #         type_sample_counts.append(n_samples)

        #     # Draw boundaries: only black lines between types
        #     cumulative_pos = 0
        #     for type_idx, n_samples in enumerate(type_sample_counts):
        #         if type_idx < len(types) - 1:
        #             type_boundary = cumulative_pos + n_samples - 0.5
        #             ax.axhline(type_boundary, color='k', linewidth=1.5)
        #             ax.axvline(type_boundary, color='k', linewidth=1.5)
        #         cumulative_pos += n_samples

        #     ax.set_xticks([])
        #     ax.set_yticks([])

        #     fname = os.path.join(save_dir, f"similarity_heatmap_{phase}.svg")
        #     fig.savefig(fname, bbox_inches='tight')
        #     plt.close(fig)
        # except Exception as e:
        #     print(f"Similarity heatmap plot failed ({phase}): {e}")

        # =======================================================
        # 3. Latent PCA Scatter 2D (PC1-PC2) - Uses all hidden states (N*T, d)
        # =======================================================
        try:
            # 用z_flat为PCA基底和投影空间
            if hidden_np.ndim == 3:
                z_flat = hidden_np.transpose(1, 0, 2).reshape(hidden_np.shape[1], -1)
            else:
                z_flat = hidden_np

            data_std = np.std(z_flat)
            if data_std < 1e-6:
                print(f"WARNING: Low variance for PC1-PC2 ({phase}): std={data_std:.2e}")

            n_components = min(6, z_flat.shape[0] - 1, z_flat.shape[1])
            n_components = max(2, n_components)

            with warnings.catch_warnings():
                warnings.filterwarnings('always', category=RuntimeWarning)
                pca_lat = PCA(n_components=n_components)
                proj_lat = pca_lat.fit_transform(z_flat)

            fig, ax = plt.subplots(figsize=(6, 6))
            colors = plt.cm.tab20(np.linspace(0, 1, len(types)))

            for tidx in range(len(types)):
                mask = (labels_np == tidx)
                if mask.sum() > 0:
                    ax.scatter(proj_lat[mask, 0], proj_lat[mask, 1],
                              c=[colors[tidx]], s=15, alpha=0.7,
                              label=types[tidx], edgecolors='none')

            ax.set_xlabel('PC1', fontsize=11)
            ax.set_ylabel('PC2', fontsize=11)
            ax.grid(True, alpha=0.2)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12),
                     ncol=min(5, len(types)), frameon=False, fontsize=9)

            fname = os.path.join(save_dir, f"latent_pca_12_{phase}.svg")
            fig.savefig(fname, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"Latent PCA PC1-PC2 failed ({phase}): {e}")
        
        # =======================================================
        # 4. Latent PCA Scatter 2D (PC2-PC3) - Uses all hidden states (N*T, d)
        # =======================================================
        try:
            # 用z_flat为PCA基底和投影空间
            if hidden_np.ndim == 3:
                z_flat = hidden_np.transpose(1, 0, 2).reshape(hidden_np.shape[1], -1)
            else:
                z_flat = hidden_np

            data_std = np.std(z_flat)
            if data_std < 1e-6:
                print(f"WARNING: Low variance for PC2-PC3 ({phase}): std={data_std:.2e}")

            n_components = min(3, z_flat.shape[0] - 1, z_flat.shape[1])
            n_components = max(3, n_components)

            with warnings.catch_warnings():
                warnings.filterwarnings('always', category=RuntimeWarning)
                pca_lat = PCA(n_components=n_components)
                proj_lat = pca_lat.fit_transform(z_flat)

            fig, ax = plt.subplots(figsize=(6, 6))
            colors = plt.cm.tab20(np.linspace(0, 1, len(types)))

            for tidx in range(len(types)):
                mask = (labels_np == tidx)
                if mask.sum() > 0 and n_components > 2:
                    ax.scatter(proj_lat[mask, 1], proj_lat[mask, 2],
                              c=[colors[tidx]], s=15, alpha=0.7,
                              label=types[tidx], edgecolors='none')

            ax.set_xlabel('PC2', fontsize=11)
            ax.set_ylabel('PC3', fontsize=11)
            ax.grid(True, alpha=0.2)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12),
                     ncol=min(5, len(types)), frameon=False, fontsize=9)

            fname = os.path.join(save_dir, f"latent_pca_23_{phase}.svg")
            fig.savefig(fname, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"Latent PCA PC2-PC3 failed ({phase}): {e}")
        
        
        # =======================================================
        # 5. Latent PCA 3D (PC1-PC2-PC3) by Type - Uses all hidden states (N*T, d)
        # =======================================================
        try:
            # 用z_flat为PCA基底和投影空间
            if hidden_np.ndim == 3:
                z_flat = hidden_np.transpose(1, 0, 2).reshape(hidden_np.shape[1], -1)
            else:
                z_flat = hidden_np

            n_components = min(3, z_flat.shape[0], z_flat.shape[1])
            pca_lat = PCA(n_components=n_components)
            proj_lat = pca_lat.fit_transform(z_flat)

            fig = go.Figure()
            colors = plt.cm.tab20(np.linspace(0, 1, len(types)))
            colors_hex = ['#%02x%02x%02x' % tuple(int(c*255) for c in colors[i][:3]) for i in range(len(types))]

            for tidx in range(len(types)):
                mask = (labels_np == tidx)
                if mask.sum() > 0:
                    x = proj_lat[mask, 0]
                    y = proj_lat[mask, 1] if n_components > 1 else np.zeros(mask.sum())
                    z = proj_lat[mask, 2] if n_components > 2 else np.zeros(mask.sum())
                    seq_labels = np.array(seqs)[mask] if seqs is not None else None
                    fig.add_trace(go.Scatter3d(
                        x=x, y=y, z=z,
                        mode='markers+text',
                        name=types[tidx],
                        marker=dict(size=4, color=colors_hex[tidx], opacity=0.7),
                        text=seq_labels,
                        textposition='top center',
                        textfont=dict(size=8)
                    ))

            fig.update_layout(
                title=f"Latent PCA by Type ({phase}) - {n_components} components",
                scene=dict(
                    xaxis_title='PC1',
                    yaxis_title='PC2' if n_components > 1 else 'PC2 (N/A)',
                    zaxis_title='PC3' if n_components > 2 else 'PC3 (N/A)'
                ),
                margin=dict(l=0, r=0, b=0, t=30),
                legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='center', x=0.5)
            )

            fname = os.path.join(save_dir, f"latent_pca_{phase}.html")
            fig.write_html(fname)
        except Exception as e:
            print(f"Latent PCA 3D by type failed ({phase}): {e}")
    
        # =======================================================
        # 6. Sequence Trajectories in Hidden Space 
        # =======================================================
        if hidden_np is not None:
            L_ts = hidden_np.shape[0]
            try:
                # 用z_flat为PCA基底
                if hidden_np.ndim == 3:
                    z_flat = hidden_np.transpose(1, 0, 2).reshape(hidden_np.shape[1], -1)
                else:
                    z_flat = hidden_np

                T, N, d = hidden_np.shape if hidden_np.ndim == 3 else (1, ) + hidden_np.shape
                feat_dim = z_flat.shape[1]

                def pad_traj_point(t, h):
                    # t: 当前时刻, h: (d,)
                    vec = np.zeros(feat_dim, dtype=h.dtype)
                    start = t * d
                    end = start + d
                    vec[start:end] = h
                    return vec

                # 2D PCA
                pca = PCA(n_components=2)
                pca.fit(z_flat)

                n_types = len(types)
                total_panels = n_types + 1
                ncols = min(3, total_panels)
                nrows = (total_panels + ncols - 1) // ncols

                fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows))
                if total_panels == 1:
                    axes = np.array([axes])
                axes = axes.flatten()
                colors = plt.cm.tab20(np.linspace(0, 1, len(types)))

                for type_idx in range(n_types):
                    ax = axes[type_idx]
                    type_mask = (labels_np == type_idx)
                    type_indices = np.where(type_mask)[0]
                    for i, batch_idx in enumerate(type_indices):
                        traj_points = []
                        for t in range(L_ts):
                            h_t = hidden_np[t, batch_idx, :]
                            traj_points.append(pad_traj_point(t, h_t))
                        traj_points = np.array(traj_points)
                        if np.std(traj_points) < 1e-8:
                            continue
                        proj_traj = pca.transform(traj_points)
                        ax.plot(proj_traj[:, 0], proj_traj[:, 1], color=colors[type_idx], alpha=0.5, linewidth=1.2)
                        ax.scatter(proj_traj[:, 0], proj_traj[:, 1], color=colors[type_idx], s=30, alpha=0.8, zorder=3)
                        # 起点为黑色实心大圆点，终点为黑色空心大圆点
                        ax.scatter(proj_traj[0, 0], proj_traj[0, 1], color='black', s=80, marker='o', zorder=10, label='Start' if i == 0 else None)
                        ax.scatter(proj_traj[-1, 0], proj_traj[-1, 1], facecolors='none', edgecolors='black', s=80, marker='o', zorder=10, label='End' if i == 0 else None)
                    handles, labels_ = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels_, handles))
                    ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=9, frameon=False)
                    ax.set_title(types[type_idx], fontsize=13, fontweight='bold')
                    ax.set_xlabel('PC1', fontsize=12)
                    ax.set_ylabel('PC2', fontsize=12)
                    ax.grid(True, alpha=0.2)

                ax_all = axes[n_types]
                for i, batch_idx in enumerate(range(len(labels_np))):
                    type_idx = labels_np[batch_idx]
                    traj_points = []
                    for t in range(L_ts):
                        h_t = hidden_np[t, batch_idx, :]
                        traj_points.append(pad_traj_point(t, h_t))
                    traj_points = np.array(traj_points)
                    proj_traj = pca.transform(traj_points)
                    ax_all.plot(proj_traj[:, 0], proj_traj[:, 1], color=colors[type_idx], alpha=0.4, linewidth=1.0)
                    ax_all.scatter(proj_traj[:, 0], proj_traj[:, 1], color=colors[type_idx], s=20, alpha=0.7, zorder=3)
                    # 起点为黑色实心大圆点，终点为黑色空心大圆点
                    ax_all.scatter(proj_traj[0, 0], proj_traj[0, 1], color='black', s=80, marker='o', zorder=10, label='Start' if i == 0 else None)
                    ax_all.scatter(proj_traj[-1, 0], proj_traj[-1, 1], facecolors='none', edgecolors='black', s=80, marker='o', zorder=10, label='End' if i == 0 else None)

                handles, labels_ = ax_all.get_legend_handles_labels()
                by_label = dict(zip(labels_, handles))
                ax_all.legend(by_label.values(), by_label.keys(), loc='best', fontsize=9, frameon=False)
                ax_all.set_title('All types', fontsize=13, fontweight='bold')
                ax_all.set_xlabel('PC1', fontsize=12)
                ax_all.set_ylabel('PC2', fontsize=12)
                ax_all.grid(True, alpha=0.2)

                for idx in range(total_panels, len(axes)):
                    axes[idx].axis('off')
                fig.suptitle(f'Sequence Trajectories by Type ({phase})', fontsize=14, fontweight='bold', y=0.995)
                plt.tight_layout()
                fname = os.path.join(save_dir, f"trajectory_{phase}.svg")
                fig.savefig(fname, bbox_inches='tight', dpi=150)
                plt.close(fig)

                # 3D Trajectory Visualization (Plotly)
                pca3d = PCA(n_components=3)
                pca3d.fit(z_flat)
                colors_hex = ['#%02x%02x%02x' % tuple(int(c*255) for c in colors[i][:3]) for i in range(len(types))]
                fig3d_all = go.Figure()
                seen_types = set()
                for batch_idx in range(len(labels_np)):
                    type_idx = labels_np[batch_idx]
                    traj_points = []
                    for t in range(L_ts):
                        h_t = hidden_np[t, batch_idx, :]
                        traj_points.append(pad_traj_point(t, h_t))
                    traj_points = np.array(traj_points)
                    if np.std(traj_points) < 1e-8:
                        continue
                    proj_traj = pca3d.transform(traj_points)
                    show_leg = type_idx not in seen_types
                    seen_types.add(type_idx)
                    fig3d_all.add_trace(go.Scatter3d(
                        x=proj_traj[:,0], y=proj_traj[:,1], z=proj_traj[:,2],
                        mode='lines+markers',
                        marker=dict(size=3, color=colors_hex[type_idx]),
                        line=dict(color=colors_hex[type_idx], width=1.5),
                        name=f"Type {type_idx}",
                        legendgroup=f"type{type_idx}",
                        showlegend=show_leg
                    ))
                fig3d_all.update_layout(
                    title=f"Trajectory 3D - All Types ({phase})",
                    scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'),
                    margin=dict(l=0, r=0, b=0, t=30),
                    legend=dict(orientation='h', yanchor='top', y=1.05, xanchor='center', x=0.5)
                )
                fname3d_all = os.path.join(save_dir, f"trajectory3d_{phase}.html")
                fig3d_all.write_html(fname3d_all)
            except Exception as e:
                print(f"Sequence trajectory plot failed ({phase}): {e}")
                    
        

    # # smoothed loss/acc curves
    # def smooth(y, window=10):
    #     y = np.array(y)
    #     if len(y) < window:
    #         return y
    #     w = np.ones(window) / window
    #     y_smooth = np.convolve(y, w, mode='valid')
    #     pad_left = (len(y) - len(y_smooth)) // 2
    #     pad_right = len(y) - len(y_smooth) - pad_left
    #     return np.concatenate([y[:pad_left], y_smooth, y[-pad_right:]])
        
    # epochs_range = range(1, len(history['train_loss']) + 1)
    # fig, ax1 = plt.subplots(figsize=(3.5, 3), constrained_layout=True)
    # ax1.plot(epochs_range, smooth(history['train_loss']), label='Train loss', color='b', linewidth=1.5, linestyle='-')
    # ax1.plot(epochs_range, smooth(history['test_loss']), label='Test loss', color='g', linewidth=1.5, linestyle='-')
    # ax1.set_xlabel('Epoch')
    # ax1.set_ylabel('Loss')
    # ax1.grid(True, alpha=0.3)

    # ax2 = ax1.twinx()
    # ax2.plot(epochs_range, smooth(history['train_acc']), label='Train acc', color='b', linewidth=1.5, linestyle='--')
    # ax2.plot(epochs_range, smooth(history['test_acc']), label='Test acc', color='g', linewidth=1.5, linestyle='--')
    # ax2.set_ylabel('acc')

    # lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False)

    # fname = os.path.join(save_dir, f"loss_acc.svg")
    # fig.savefig(fname, bbox_inches='tight')
    # plt.close(fig)


def train(model, X_train, X_test, test_labels, train_labels=None, types=None,
    n_epochs=300, lr=0.001, weight_decay=1e-3, verbose=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {k: [] for k in ['train_loss', 'test_loss', 'train_acc', 'test_acc', 'train_radial', 'test_radial',
                                'train_P', 'train_L', 'train_A', 'test_P', 'test_L', 'test_A']}
 
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        X_batch = X_train
        hidden, output = model(X_batch)

        ce_loss = F.cross_entropy(output.reshape(-1, output.shape[-1]),
                        torch.argmax(X_batch, dim=-1).reshape(-1))
        total_loss = ce_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # evaluation
        with torch.no_grad():
            pred_train = torch.argmax(output, dim=-1)
            target_train = torch.argmax(X_batch, dim=-1)
            train_acc = (pred_train == target_train).all(dim=0).float().mean().item()
            
            # Only compute radial score at the last epoch to save time
            if epoch % 200 == 0:
                if hidden.ndim == 3:
                    hidden_2d = hidden.permute(1, 0, 2).reshape(hidden.shape[1], -1)
                else:
                    hidden_2d = hidden

                hidden_np = hidden_2d.cpu().numpy()
                labels_np = train_labels.cpu().numpy()
                unique_labels = np.unique(labels_np)
                train_radial, train_P, train_L, train_A = radial_pattern_score(
                        hidden_np, labels_np, types, return_components=True)

        model.eval()
        with torch.no_grad():
            hidden_test, test_output = model(X_test)
            test_ce_loss = F.cross_entropy(
                test_output.reshape(-1, test_output.shape[-1]),
                torch.argmax(X_test, dim=-1).reshape(-1)
            )
            test_loss = test_ce_loss

            pred_test = torch.argmax(test_output, dim=-1)
            target_test = torch.argmax(X_test, dim=-1)
            test_acc = (pred_test == target_test).all(dim=0).float().mean().item()

            if hidden_test.ndim == 3:
                hidden_test_2d = hidden_test.permute(1, 0, 2).reshape(hidden_test.shape[1], -1)
            else:
                hidden_test_2d = hidden_test
            # Only compute test radial score at the last epoch to save time
            if epoch % 200 == 0:
                hidden_test_np = hidden_test_2d.cpu().numpy()
                test_labels_np = test_labels.cpu().numpy()
                unique_labels = np.unique(test_labels_np)
                test_radial, test_P, test_L, test_A = radial_pattern_score(
                    hidden_test_np, test_labels_np, types, return_components=True)

        # history - only save final epoch values to reduce memory
        if epoch % 200 == 0:
            history['train_loss'].append(total_loss.item())
            history['test_loss'].append(float(test_loss))
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            history['train_radial'].append(train_radial)
            history['test_radial'].append(test_radial)
            history['train_P'].append(train_P)
            history['train_L'].append(train_L)
            history['train_A'].append(train_A)
            history['test_P'].append(test_P)
            history['test_L'].append(test_L)
            history['test_A'].append(test_A)
        
        # Print at the final epoch if verbose
        if verbose and epoch%200 == 0:
            print(f"\nEpoch {epoch+1}/{n_epochs}:")
            print(f"  Train - Loss: {total_loss.item():.4f}, Acc: {train_acc:.4f}, Radial: {train_radial:.4f} (P={train_P:.4f}, L={train_L:.4f}, A={train_A:.4f})")
            print(f"  Test  - Loss: {test_loss.item():.4f}, Acc: {test_acc:.4f}, Radial: {test_radial:.4f} (P={test_P:.4f}, L={test_L:.4f}, A={test_A:.4f})")

    return history


def compute_metrics(train_metrics, test_metrics, latent_test, test_labels, types):
    print('\n' + '='*60)
    print('='*60)
    print(f"Train acc: {train_metrics['acc']:.4f}")
    print(f"Test acc: {test_metrics['acc']:.4f}")
    
    # Compute radial pattern score (now using hidden_test)
    if latent_test.ndim == 3:
        hidden_np = latent_test.permute(1, 0, 2).reshape(latent_test.shape[1], -1).cpu().numpy()
    else:
        hidden_np = latent_test.cpu().numpy()

    test_labels_np = test_labels if isinstance(test_labels, np.ndarray) else test_labels.cpu().numpy()
    unique_labels = np.unique(test_labels_np)
    if len(unique_labels) > 1:
        radial, P, L, A = radial_pattern_score(hidden_np, test_labels_np, types, return_components=True)
    else:
        radial, P, L, A = 0.0, 0.0, 0.0, 0.0

    return {'radial': float(radial), 'P': float(P), 'L': float(L), 'A': float(A)}


def run_experiment():
    set_seed(seed)

    model = RNN(d_input=alpha, d_hidden=d_hidden, num_layers=num_layers, d_output=alpha,
            output_activation=None, nonlinearity='relu', device=device).to(device)

    seq_train, seq_test, labels_train, labels_test, types = generate_instances(alpha, L, m, frac_train=0.8)
    X_train = sequences_to_tensor(seq_train, alpha).to(device)
    X_test = sequences_to_tensor(seq_test, alpha).to(device)
    test_labels = torch.tensor(labels_test, dtype=torch.long)
    train_labels = torch.tensor(labels_train, dtype=torch.long)

    history = train(model, X_train, X_test, test_labels, train_labels=train_labels, types=types,
        n_epochs=epochs, lr=lr, weight_decay=weight_decay)

    # final evaluation
    model.eval()
    with torch.no_grad():
        hidden_test, test_output = model(X_test)
        pred_test = torch.argmax(test_output, dim=-1)
        target_test = torch.argmax(X_test, dim=-1)
        test_acc = (pred_test == target_test).all(dim=0).float().mean().item()

    train_acc = history['train_acc'][-1] if len(history['train_acc']) > 0 else 0.0
    train_metrics = {'acc': train_acc}
    test_metrics = {'acc': test_acc}

    # compute numeric metrics (silhouette) and print
    try:
        compute_metrics(train_metrics, test_metrics, hidden_test, labels_test, types)
    except Exception as e:
        print(f"compute_metrics failed: {e}")

    # save consolidated final-epoch diagnostics (SVGs)
    try:
        plot_diagnostics(model, X_train, torch.tensor(labels_train, dtype=torch.long), X_test, test_labels, types, save_dir=SAVE_DIR, history=history, seq_train=seq_train, seq_test=seq_test)
    except Exception as e:
        print(f"plot_diagnostics failed: {e}")


if __name__ == '__main__':
    run_experiment()
