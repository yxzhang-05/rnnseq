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
from analysis_utils import radial_pattern_score, compute_umcontent, compute_distance


seed = 40
L, m, alpha = 4, 2, 6
epochs = 1000 
lr = 1e-3
d_hidden = 8
d_latent = 2
d_latent_hidden = 4
weight_decay = 1e-3 
num_layers = 1
device = torch.device('cpu')
SAVE_DIR = "results"
TRAIN_PRINT_FINAL = True

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

    for type_idx in range(len(types)):
        t = types[type_idx]
        all_type_seqs = [seq_replace_symbols(t, perm) for perm in all_perms]

        n = len(all_type_seqs)
        split = int(frac_train * n)
        perm_idx = np.random.permutation(n)
        train_idx = perm_idx[:split]
        test_idx = perm_idx[split:]

        train_type_seqs = [all_type_seqs[i] for i in train_idx]
        test_type_seqs = [all_type_seqs[i] for i in test_idx]

        train_seqs.extend(train_type_seqs)
        test_seqs.extend(test_type_seqs)
        train_labels.extend([type_idx] * len(train_type_seqs))
        test_labels.extend([type_idx] * len(test_type_seqs))

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


# -------------------------------------------------------
#                Representation Geometry
# -------------------------------------------------------

def trajectory_pca_latent(model, X_train, seq_train, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        _, enc_lat = model.encoder(X_train)
        lat_hidden, _ = model.latent(enc_lat)

    # lat_hidden: [T, B, H] -> [(T*B), H]
    hidden_np = lat_hidden.detach().cpu().numpy()
    t_steps, batch_size, hidden_dim = hidden_np.shape
    hidden_flat = hidden_np.reshape(t_steps * batch_size, hidden_dim)

    # PCA on all hidden states, then reshape back to [T, B, 2].
    pca = PCA(n_components=2)
    hidden_2d_flat = pca.fit_transform(hidden_flat)
    hidden_2d = hidden_2d_flat.reshape(t_steps, batch_size, 2)

    seq_chars = np.asarray([list(s) for s in np.asarray(seq_train)])
    if seq_chars.shape[1] != t_steps:
        # Keep plotting robust if sequence length and hidden timesteps differ.
        t_plot = min(seq_chars.shape[1], t_steps)
        hidden_2d = hidden_2d[:t_plot]
        t_steps = t_plot
        seq_chars = seq_chars[:, :t_plot]

    letters = np.unique(seq_chars)
    cmap = plt.get_cmap('tab10', max(len(letters), 1))
    color_map = {letter: cmap(i) for i, letter in enumerate(letters)}

    plt.figure(figsize=(3, 3.5))
    for b in range(batch_size):
        xy = hidden_2d[:, b, :]
        if t_steps <= 1:
            plt.plot(xy[:, 0], xy[:, 1], color='0.6', linewidth=1.0, alpha=0.6, zorder=1)
            continue

        # Use a light-to-dark gradient to indicate trajectory direction (start -> end).
        for t in range(t_steps - 1):
            frac = t / max(t_steps - 2, 1)
            alpha = 0.2 + 0.65 * frac
            gray = 0.85 - 0.45 * frac
            plt.plot(xy[t:t + 2, 0], xy[t:t + 2, 1], color=(gray, gray, gray),
                    linewidth=1.0, alpha=alpha, zorder=1)

    for t in range(t_steps):
        letters_t = seq_chars[:, t]
        point_colors = [color_map[ch] for ch in letters_t]
        size = 12 + 2.5 * t
        alpha_t = 0.35 + 0.55 * (t / max(t_steps - 1, 1))
        plt.scatter(
            hidden_2d[t, :, 0],
            hidden_2d[t, :, 1],
            c=point_colors,
            s=size,
            marker='o',
            alpha=alpha_t,
            edgecolors='white',
            linewidths=0.4,
            zorder=3,
        )

    legend_handles = []
    for letter in letters:
        handle = plt.Line2D([0], [0], color=color_map[letter], marker='o', lw=0, markersize=5, label=str(letter))
        legend_handles.append(handle)

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
    plt.grid(True, alpha=0.3)
    plt.legend(
        handles=legend_handles,
        fontsize=8,
        ncol=3,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.22),
        frameon=False,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(os.path.join(save_dir, 'latent_trajectory_pca_train.svg'), dpi=200)
    plt.close()


#  -------------------------------------------------------
#                 Connnectivity Structure
#  -------------------------------------------------------
                   
# Singular Value Curves
def singular_value_curve(model, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)

    recurrent_blocks = [
        ('encoder', model.encoder.rnn.h2h),
        ('latent', model.latent.h2h),
        ('decoder', model.decoder.rnn.h2h),
    ]

    plt.figure(figsize=(3.5, 3))

    for name, recurrent_layer in recurrent_blocks:
        w = recurrent_layer.weight.detach().cpu()
        singular_values = torch.linalg.svdvals(w).numpy()
        x = np.arange(1, singular_values.shape[0] + 1)
        plt.plot(x, singular_values, marker='o', linewidth=1.5, markersize=3, label=name)

    plt.xlabel('Index')
    plt.ylabel('Singular value')
    plt.title('Singular Value Curves')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'singular_value_curves.svg'), dpi=200)
    plt.close()


# Effective Dimensionality 
def hidden_dimensionality(model, X, save_dir=SAVE_DIR, eps=1e-12):
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        enc_hidden, enc_lat = model.encoder(X)
        lat_hidden, lat_out = model.latent(enc_lat)
        dec_hidden, _ = model.decoder.rnn(lat_out)

    block_hidden = {
        'encoder': enc_hidden.detach().cpu().numpy(),
        'latent': lat_hidden.detach().cpu().numpy(),
        'decoder': dec_hidden.detach().cpu().numpy(),
    }

    plt.figure(figsize=(3.5, 3))
    for block_name, hidden_np in block_hidden.items():
        time_steps = hidden_np.shape[0]
        x = np.arange(1, time_steps + 1)

        d90_values = []
        for t in range(time_steps):
            states_t = hidden_np[t, :, :]
            if states_t.shape[0] < 2:
                d90_values.append(np.nan)
                continue

            centered = states_t - states_t.mean(axis=0, keepdims=True)
            singular_values = np.linalg.svd(centered, compute_uv=False)
            explained = singular_values ** 2
            total = explained.sum()
            if total <= eps:
                d90_values.append(0)
                continue

            cumulative = np.cumsum(explained) / total
            d90_values.append(int(np.searchsorted(cumulative, 0.9) + 1))

        plt.plot(x, d90_values, marker='o', linewidth=1.5, markersize=3, label=block_name)

    plt.xlabel('Timestep')
    plt.ylabel('d90')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'hidden_d90.svg'), dpi=200)
    plt.close()


# -------------------------------------------------------
#               Representation Structure
# -------------------------------------------------------
def _latent_hidden_numpy(model, X):
    with torch.no_grad():
        _, enc_lat = model.encoder(X)
        lat_hidden, _ = model.latent(enc_lat)
    return lat_hidden.detach().cpu().numpy().astype(np.float64)


def Euclidean_matrix(model, X_train, seq_train, save_dir=SAVE_DIR, latent_hidden_list=None):
    os.makedirs(save_dir, exist_ok=True)

    if latent_hidden_list is None or len(latent_hidden_list) == 0:
        latent_hidden_list = [_latent_hidden_numpy(model, X_train)]

    seq_train = np.asarray(seq_train)
    seq_chars = np.asarray([list(s) for s in seq_train])
    seq_len = seq_chars.shape[1]
    t_steps = min(latent_hidden_list[0].shape[0], seq_len)

    n_show = min(4, t_steps)
    if n_show == 0:
        return

    panel_data = []
    max_dist = 0.0
    for t in range(n_show):
        dist_acc = None
        for hidden_np in latent_hidden_list:
            z_t = hidden_np[t]
            deltas = z_t[:, None, :] - z_t[None, :, :]
            euclidean_dist = np.linalg.norm(deltas, axis=-1)
            if dist_acc is None:
                dist_acc = euclidean_dist
            else:
                dist_acc = dist_acc + euclidean_dist
        euclidean_dist = dist_acc / float(len(latent_hidden_list))
        max_dist = max(max_dist, float(np.max(euclidean_dist)))

        chars_t = seq_chars[:, t]
        order = np.argsort(chars_t, kind='stable')
        distance_sorted = euclidean_dist[order][:, order]
        chars_sorted = chars_t[order]

        unique_chars = np.unique(chars_sorted)
        boundaries = []
        centers = []
        group_labels = []
        start = 0
        for c in unique_chars:
            idx = np.where(chars_sorted == c)[0]
            end = idx[-1] + 1
            boundaries.append(end)
            centers.append((start + end - 1) / 2.0)

            # Example labels by timestep
            group_labels.append('-' * t + c + '-' * (seq_len - t - 1))
            start = end

        panel_data.append((t, distance_sorted, boundaries, centers, group_labels))

    fig, axes = plt.subplots(1, 4, figsize=(16.8, 5.0))
    axes = np.atleast_1d(axes)
    shared_im = None

    for i, (t, distance_sorted, boundaries, centers, group_labels) in enumerate(panel_data):
        ax = axes[i]
        shared_im = ax.imshow(
            distance_sorted,
            cmap='viridis',
            interpolation='nearest',
            aspect='equal',
            vmin=0,
            vmax=max_dist if max_dist > 0 else 1.0,
        )
        ax.set_box_aspect(1)

        for b in boundaries[:-1]:
            ax.axhline(b - 0.5, color='black', linewidth=0.8, alpha=0.7)
            ax.axvline(b - 0.5, color='black', linewidth=0.8, alpha=0.7)

        ax.set_xticks(centers)
        ax.set_xticklabels(group_labels, rotation=45, ha='right', fontsize=13)
        ax.set_yticks(centers)
        ax.set_yticklabels(group_labels, fontsize=13)
        ax.set_xlabel(f't{t + 1}', fontsize=16, labelpad=10)

    # Hide unused axes if sequence length < 4.
    for j in range(n_show, len(axes)):
        axes[j].axis('off')

    fig.subplots_adjust(right=0.9, wspace=0.3, top=0.92, bottom=0.18)
    cbar_ax = fig.add_axes([0.915, 0.2, 0.009, 0.62])
    cbar = fig.colorbar(shared_im, cax=cbar_ax)
    cbar.set_label('Euclidean distance', fontsize=14)
    cbar.ax.tick_params(labelsize=11)
    fig.savefig(os.path.join(save_dir, 'euclidean_matrix_train.svg'), dpi=200)
    plt.close(fig)


def plot_euclidean_matrix(model, X_train, seq_train, X_test, test_labels, train_labels, types, save_dir=SAVE_DIR, ensemble_seeds=None):
    
    if ensemble_seeds is None:
        Euclidean_matrix(model, X_train, seq_train, save_dir=save_dir, latent_hidden_list=None)
        return

    global TRAIN_PRINT_FINAL

    latent_hidden_list = []
    ensemble_metrics = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': [], 'train_radial': [], 'test_radial': []}
    prev_print_state = TRAIN_PRINT_FINAL
    TRAIN_PRINT_FINAL = False
    for s in ensemble_seeds:
        set_seed(s)
        model_seed = RNNAutoencoder(alpha, d_hidden, d_latent_hidden, num_layers, d_latent, L).to(device)
        history_seed = train(model_seed, X_train, X_test, test_labels, train_labels=train_labels, types=types, n_epochs=epochs, lr=lr, weight_decay=weight_decay)
        for key in ensemble_metrics:
            if len(history_seed[key]) > 0:
                ensemble_metrics[key].append(history_seed[key][-1])
        latent_hidden_list.append(_latent_hidden_numpy(model_seed, X_train))
    TRAIN_PRINT_FINAL = prev_print_state

    if len(ensemble_metrics['train_loss']) > 0:
        print('\nEuclidean ensemble mean over seeds:')
        print(f"  Train - Loss: {np.mean(ensemble_metrics['train_loss']):.4f}, Acc: {np.mean(ensemble_metrics['train_acc']):.4f}, Radial: {np.mean(ensemble_metrics['train_radial']):.4f}")
        print(f"  Test  - Loss: {np.mean(ensemble_metrics['test_loss']):.4f}, Acc: {np.mean(ensemble_metrics['test_acc']):.4f}, Radial: {np.mean(ensemble_metrics['test_radial']):.4f}")

    Euclidean_matrix(model, X_train, seq_train, save_dir=save_dir, latent_hidden_list=latent_hidden_list)


# -------------------------------------------------------
#                    Connectivity 
# -------------------------------------------------------

def _get_latent_hidden_and_weight(model, X):
    with torch.no_grad():
        _, enc_lat = model.encoder(X)
        hidden, _ = model.latent(enc_lat)
        recurrent_layer = model.latent.h2h

    hidden_np = hidden.detach().cpu().numpy().astype(np.float64)
    w = recurrent_layer.weight.detach().cpu().numpy().astype(np.float64)
    return hidden_np, w


def plot_activity_along_modes(model, X, save_dir=SAVE_DIR, top_k=4):
    os.makedirs(save_dir, exist_ok=True)

    hidden_np, w = _get_latent_hidden_and_weight(model, X)
    _, _, vt = np.linalg.svd(w, full_matrices=False)
    k = max(1, min(top_k, vt.shape[0], hidden_np.shape[2]))
    modes = vt[:k].T

    proj = np.tensordot(hidden_np, modes, axes=([2], [0]))
    proj_mean = proj.mean(axis=1)
    proj_std = proj.std(axis=1)

    plt.figure(figsize=(4, 3))
    x = np.arange(1, proj_mean.shape[0] + 1)
    for i in range(k):
        line = plt.plot(x, proj_mean[:, i], marker='o', markersize=3, linewidth=1.4, label=f'mode {i + 1}')
        color = line[0].get_color()
        lower = proj_mean[:, i] - proj_std[:, i]
        upper = proj_mean[:, i] + proj_std[:, i]
        plt.fill_between(x, lower, upper, color=color, alpha=0.2, linewidth=0)
    plt.xlabel('Timestep')
    plt.ylabel('Projected activity')
    plt.title(f'Latent Activity Along Top-{k} Modes')
    plt.xticks(x)
    plt.xlim(1, proj_mean.shape[0])
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'latent_activity_along_modes.svg'), dpi=200)
    plt.close()


def plot_ultrametric_content(model, X, save_dir=SAVE_DIR, num_sv_list=(1, 2, 4, 6)):
    os.makedirs(save_dir, exist_ok=True)

    hidden_np, w = _get_latent_hidden_and_weight(model, X)
    _, _, vt = np.linalg.svd(w, full_matrices=False)
    max_k = min(vt.shape[0], hidden_np.shape[2])
    t_steps = hidden_np.shape[0]
    x = np.arange(1, t_steps + 1)

    plt.figure(figsize=(4, 3))
    for n_sv in num_sv_list:
        if n_sv > max_k:
            continue
        modes = vt[:n_sv].T
        proj = np.tensordot(hidden_np, modes, axes=([2], [0]))
        uc_values = []
        for t in range(t_steps):
            dist_t = compute_distance(proj[None, t, :, :])[0]
            uc_t = compute_umcontent(dist_t[None, :, :], return_triplets=False)[0]
            uc_values.append(uc_t)
        plt.plot(x, uc_values, marker='o', markersize=3, linewidth=1.4, label=f'numSV={n_sv}')

    plt.xlabel('Timestep')
    plt.ylabel('Ultrametric content')
    plt.xticks(x)
    plt.xlim(1, t_steps)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ultrametric_content.svg'), dpi=200)
    plt.close()

    
def train(model, X_train, X_test, test_labels, train_labels=None, types=None,
    n_epochs=300, lr=0.001, weight_decay=1e-3):
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
        if TRAIN_PRINT_FINAL and epoch == n_epochs - 1:
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

    # plotting
    singular_value_curve(model, save_dir=SAVE_DIR)
    hidden_dimensionality(model, X_test, save_dir=SAVE_DIR)
    plot_euclidean_matrix(model, X_train, seq_train, X_test, test_labels, train_labels, types,
        save_dir=SAVE_DIR, ensemble_seeds=range(40, 50))
    trajectory_pca_latent(model, X_train, seq_train, save_dir=SAVE_DIR)
    plot_activity_along_modes(model, X_train, save_dir=SAVE_DIR, top_k=4)
    plot_ultrametric_content(model, X_train, save_dir=SAVE_DIR, num_sv_list=(1, 2, 4, 6))
    
    
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

if __name__ == '__main__':
    run_experiment()