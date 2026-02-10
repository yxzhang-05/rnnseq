import random
import collections
import numpy as np
import torch
import torch.nn.functional as F
from model import RNNAutoencoder
from sequences import findStructures, replace_symbols as seq_replace_symbols
import string
import itertools
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
from analysis_utils import radial_pattern_score


seed = 42
L, m, alpha = 4, 2, 6
epochs = 1000 
lr = 1e-3
d_hidden = 16
d_latent = 4
d_latent_hidden = 8
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
        h_tr, enc_lat_tr = model.encoder(X_train)
        h_te, enc_lat_te = model.encoder(X_test)

    datasets = [
        (h_tr, enc_lat_tr, train_labels, 'train', seq_train),
        (h_te, enc_lat_te, test_labels, 'test', seq_test)
    ]

    for hidden, enc_lat, labels, phase, seqs in datasets:
        labels_np = _safe_numpy(labels)
        if enc_lat is not None:
            _, latent_out = model.latent(enc_lat)
        else:
            latent_out = None
        latent_out_np = latent_out.detach().cpu().numpy() if latent_out is not None else None

        # Use latent output sequence for PCA: (Seq_Len, Batch, Dim) -> (Batch, Seq_Len*Dim)
        if latent_out_np is not None and latent_out_np.ndim == 3:
            latent_out_flat = latent_out_np.transpose(1, 0, 2).reshape(latent_out_np.shape[1], -1)
        else:
            latent_out_flat = latent_out_np

        # Extract letter combinations
        letter_combos = None
        if seqs is not None:
            letter_combos = [''.join(sorted(set(seq))) for seq in seqs]

        # Color by type
        unique_types = range(len(types)) if types is not None else np.unique(labels_np)

        # =======================================================
        # 1. Latent output (all timesteps) PCA Spectrum (Global) - Uses latent_out_flat
        # =======================================================
        try:
            n_comp = min(latent_out_flat.shape[1], latent_out_flat.shape[0] - 1, 10)
            n_comp = max(1, n_comp)

            pca_lat = PCA(n_components=n_comp)
            pca_lat.fit(latent_out_flat)
            evr_lat = pca_lat.explained_variance_ratio_

            fig, ax = plt.subplots(figsize=(3, 3))
            ax.bar(range(1, len(evr_lat)+1), evr_lat, color='C1')
            ax.set_xlabel('PC')
            ax.set_ylabel('Explained variance ratio')
            ax.set_title(f'{phase.capitalize()}')
            fname = os.path.join(save_dir, f"latent_output_all_pca_spectrum_{phase}.svg")
            fig.savefig(fname, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"Latent output all-timesteps PCA spectrum failed ({phase}): {e}")

        # =======================================================
        # 2. Latent output (all timesteps) PCA Scatter 2D (PC1-PC2) - Uses latent_out_flat
        # =======================================================
        try:
            n_components = min(3, latent_out_flat.shape[0] - 1, latent_out_flat.shape[1])
            n_components = max(2, n_components)

            pca_lat = PCA(n_components=n_components)
            proj_lat = pca_lat.fit_transform(latent_out_flat)

            fig, ax = plt.subplots(figsize=(4, 4))
            group_count = len(types)
            colors = plt.cm.tab20(np.linspace(0, 1, group_count))
            for tidx in range(len(types)):
                mask = (labels_np == tidx)
                if mask.sum() > 0:
                    ax.scatter(proj_lat[mask, 0], proj_lat[mask, 1],
                                c=[colors[tidx]], s=50, alpha=0.8,
                                label=types[tidx], edgecolors='none')
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=int(np.ceil(len(types)/2)), frameon=False, fontsize=9)

            ax.set_xlabel('PC1', fontsize=11)
            ax.set_ylabel('PC2', fontsize=11)
            ax.grid(True, alpha=0.2)

            fname = os.path.join(save_dir, f"latent_output_all_pca_12_{phase}.svg")
            fig.savefig(fname, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"Latent output all-timesteps PCA PC1-PC2 failed ({phase}): {e}")

        # =======================================================
        # 3. Latent output (all timesteps) PCA Scatter 2D (PC2-PC3) - Uses latent_out_flat
        # =======================================================
        try:
            n_components = min(3, latent_out_flat.shape[0] - 1, latent_out_flat.shape[1])
            n_components = max(3, n_components)

            pca_lat = PCA(n_components=n_components)
            proj_lat = pca_lat.fit_transform(latent_out_flat)

            fig, ax = plt.subplots(figsize=(4, 4))
            group_count = len(types)
            colors = plt.cm.tab20(np.linspace(0, 1, group_count))
            for tidx in range(len(types)):
                mask = (labels_np == tidx)
                if mask.sum() > 0:
                    ax.scatter(proj_lat[mask, 1], proj_lat[mask, 2],
                              c=[colors[tidx]], s=50, alpha=0.8,
                              label=types[tidx], edgecolors='none')
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=int(np.ceil(len(types)/2)), frameon=False, fontsize=9)

            ax.set_xlabel('PC2', fontsize=11)
            ax.set_ylabel('PC3', fontsize=11)
            ax.grid(True, alpha=0.2)

            fname = os.path.join(save_dir, f"latent_output_all_pca_23_{phase}.svg")
            fig.savefig(fname, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"Latent output all-timesteps PCA PC2-PC3 failed ({phase}): {e}")

        # =======================================================
        # 4. Latent output (all timesteps) PCA 3D (PC1-PC2-PC3) by Type - Uses latent_out_flat
        # =======================================================
        try:
            n_components = min(3, latent_out_flat.shape[0], latent_out_flat.shape[1])
            pca_lat = PCA(n_components=n_components)
            proj_lat = pca_lat.fit_transform(latent_out_flat)

            fig = go.Figure()
            group_count = len(types)
            colors = plt.cm.tab20(np.linspace(0, 1, group_count))
            colors_hex = ['#%02x%02x%02x' % tuple(int(c*255) for c in colors[i][:3])
                          for i in range(group_count)]
            for tidx in range(len(types)):
                mask = (labels_np == tidx)
                if mask.sum() > 0:
                    x = proj_lat[mask, 0]
                    y = proj_lat[mask, 1] if n_components > 1 else np.zeros(mask.sum())
                    z = proj_lat[mask, 2] if n_components > 2 else np.zeros(mask.sum())

                    if seqs is not None:
                        seq_labels = [seqs[i] for i in range(len(labels_np)) if labels_np[i] == tidx]
                    else:
                        seq_labels = None

                    fig.add_trace(go.Scatter3d(
                        x=x, y=y, z=z,
                        mode='markers+text',
                        name=types[tidx],
                        text=seq_labels,
                        textposition='top center',
                        textfont=dict(size=8),
                        marker=dict(size=5, color=colors_hex[tidx], opacity=0.8)
                    ))

            fig.update_layout(
                title=f"Latent output (all timesteps) PCA by Type ({phase}) - {n_components} components",
                scene=dict(
                    xaxis_title='PC1',
                    yaxis_title='PC2' if n_components > 1 else 'PC2 (N/A)',
                    zaxis_title='PC3' if n_components > 2 else 'PC3 (N/A)'
                ),
                margin=dict(l=0, r=0, b=0, t=30),
                legend=dict(orientation='h', yanchor='top', y=1.1, xanchor='center', x=0.5)
            )

            fname = os.path.join(save_dir, f"latent_output_all_pca_{phase}.html")
            fig.write_html(fname)
        except Exception as e:
            print(f"Latent output all-timesteps PCA 3D failed ({phase}): {e}")
    


def train(model, X_train, X_test, test_labels, train_labels=None, types=None,
    n_epochs=300, lr=0.001, weight_decay=1e-3, verbose=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {k: [] for k in ['train_loss', 'test_loss', 'train_acc', 'test_acc', 'train_radial', 'test_radial',
                                'train_P', 'train_L', 'train_A', 'test_P', 'test_L', 'test_A']}
 
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        X_batch = X_train
        hidden, latent, output = model(X_batch)

        ce_loss = F.cross_entropy(
            output.reshape(-1, output.shape[-1]),
            torch.argmax(X_batch, dim=-1).reshape(-1)
        )
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
            if train_labels is not None and epoch == n_epochs - 1:
                if latent.ndim == 3:
                    latent_2d = latent.permute(1, 0, 2).reshape(latent.shape[1], -1)
                else:
                    latent_2d = latent

                latent_np = latent_2d.cpu().numpy()
                labels_np = train_labels.cpu().numpy()
                unique_labels = np.unique(labels_np)
                if len(unique_labels) > 1 and types is not None:
                    train_radial, train_P, train_L, train_A = radial_pattern_score(
                        latent_np, labels_np, types, return_components=True)
                else:
                    train_radial, train_P, train_L, train_A = 0.0, 0.0, 0.0, 0.0
            else:
                # Use dummy value for non-final epochs
                train_radial, train_P, train_L, train_A = 0.0, 0.0, 0.0, 0.0

        model.eval()
        with torch.no_grad():
            _, test_latent, test_output = model(X_test)
            test_ce_loss = F.cross_entropy(
                test_output.reshape(-1, test_output.shape[-1]),
                torch.argmax(X_test, dim=-1).reshape(-1)
            )
            test_loss = test_ce_loss

            pred_test = torch.argmax(test_output, dim=-1)
            target_test = torch.argmax(X_test, dim=-1)
            test_acc = (pred_test == target_test).all(dim=0).float().mean().item()

            if test_latent.ndim == 3:
                test_latent_2d = test_latent.permute(1, 0, 2).reshape(test_latent.shape[1], -1)
            else:
                test_latent_2d = test_latent
            
            # Only compute test radial score at the last epoch to save time
            if epoch == n_epochs - 1:
                test_latent_np = test_latent_2d.cpu().numpy()
                test_labels_np = test_labels.cpu().numpy()
                unique_labels = np.unique(test_labels_np)
                if len(unique_labels) > 1 and types is not None:
                    test_radial, test_P, test_L, test_A = radial_pattern_score(
                        test_latent_np, test_labels_np, types, return_components=True)
                else:
                    test_radial, test_P, test_L, test_A = 0.0, 0.0, 0.0, 0.0
            else:
                # Use dummy value for non-final epochs
                test_radial, test_P, test_L, test_A = 0.0, 0.0, 0.0, 0.0

        # history - only save final epoch values to reduce memory
        if epoch == n_epochs - 1:
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
        if verbose and epoch == n_epochs - 1:
            print(f"\nEpoch {epoch+1}/{n_epochs}:")
            print(f"  Train - Loss: {total_loss.item():.4f}, Acc: {train_acc:.4f}, Radial: {train_radial:.4f} (P={train_P:.4f}, L={train_L:.4f}, A={train_A:.4f})")
            print(f"  Test  - Loss: {test_loss.item():.4f}, Acc: {test_acc:.4f}, Radial: {test_radial:.4f} (P={test_P:.4f}, L={test_L:.4f}, A={test_A:.4f})")

    return history


def compute_metrics(train_metrics, test_metrics, latent_test, test_labels, types):
    print('\n' + '='*60)
    print('='*60)
    print(f"Train acc: {train_metrics['acc']:.4f}")
    print(f"Test acc: {test_metrics['acc']:.4f}")
    
    # Compute radial pattern score
    if latent_test.ndim == 3:
        latent_np = latent_test.permute(1, 0, 2).reshape(latent_test.shape[1], -1).cpu().numpy()
    else:
        latent_np = latent_test.cpu().numpy()

    test_labels_np = test_labels if isinstance(test_labels, np.ndarray) else test_labels.cpu().numpy()
    unique_labels = np.unique(test_labels_np)
    if len(unique_labels) > 1:
        radial, P, L, A = radial_pattern_score(latent_np, test_labels_np, types, return_components=True)
    else:
        radial, P, L, A = 0.0, 0.0, 0.0, 0.0

    return {'radial': float(radial), 'P': float(P), 'L': float(L), 'A': float(A)}


def run_experiment():
    set_seed(seed)

    model = RNNAutoencoder(alpha, d_hidden, d_latent_hidden, num_layers, d_latent, L).to(device)

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
        _, z_test, test_output = model(X_test)
        pred_test = torch.argmax(test_output, dim=-1)
        target_test = torch.argmax(X_test, dim=-1)
        test_acc = (pred_test == target_test).all(dim=0).float().mean().item()

    train_acc = history['train_acc'][-1] if len(history['train_acc']) > 0 else 0.0
    train_metrics = {'acc': train_acc}
    test_metrics = {'acc': test_acc}

    # compute numeric metrics (silhouette) and print
    compute_metrics(train_metrics, test_metrics, z_test, labels_test, types)
    plot_diagnostics(model, X_train, torch.tensor(labels_train, dtype=torch.long), X_test, test_labels, types, save_dir=SAVE_DIR, history=history, seq_train=seq_train, seq_test=seq_test)


if __name__ == '__main__':
    run_experiment()
