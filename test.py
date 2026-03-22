import random
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from model import RNNAutoencoder
from sequences import findStructures, replace_symbols as seq_replace_symbols
import string
import itertools
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import copy
import matplotlib.pyplot as plt
import os
from analysis_utils import compute_distance


seed = 11
L, m, alpha = 4, 2, 6
epochs = 1000 
lr = 1e-3
d_hidden = 4
d_latent = 2
d_latent_hidden = 2
weight_decay = 1e-3 
num_layers = 1
device = torch.device('cpu')
SAVE_DIR = "results"
PLOT_SIZE = 3
PLOT_FONT = 13
plt.rcParams.update({'font.size': PLOT_FONT})

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
#                  Representation Geometry
# -------------------------------------------------------

def trajectory_pca_latent(model, X_train, seq_train, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        enc_hidden, enc_lat = model.encoder(X_train)
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
        t_plot = min(seq_chars.shape[1], t_steps)
        hidden_2d = hidden_2d[:t_plot]
        t_steps = t_plot
        seq_chars = seq_chars[:, :t_plot]

    letters = np.unique(seq_chars)
    muted_palette = ['#5b8e7d', '#c97c5d', '#6c8ebf',  '#b07aa1', '#9e9d57', '#6fa3a3', ]
    # User-requested remap for first six letters: abcdef -> efacdb.
    remap_order = [0, 1, 2, 3, 4, 5]
    remapped_palette = [muted_palette[i] for i in remap_order] + muted_palette
    color_map = {letter: remapped_palette[i % len(remapped_palette)] for i, letter in enumerate(letters)}

    plt.figure(figsize=(3, 3.5))

    for t in range(t_steps):
        letters_t = seq_chars[:, t]
        point_colors = [color_map[ch] for ch in letters_t]
        frac = t / max(t_steps - 1, 1)
        alpha_t = 0.1 + 0.9 * frac 
        plt.scatter(
            hidden_2d[t, :, 0], hidden_2d[t, :, 1],
            c=point_colors, s=30, marker='o', alpha=alpha_t,
            edgecolors='none', linewidths=0, zorder=3,
        )

    legend_handles = [
        plt.Line2D([0], [0], color=color_map[letter], marker='o', lw=0, markersize=5, label=str(letter))
        for letter in letters
    ]
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
    plt.grid(True, alpha=0.3)
    plt.legend(handles=legend_handles, fontsize=8, ncol=1,
               loc='upper center', bbox_to_anchor=(0.5, 1.22),
               frameon=True, framealpha=0.9, facecolor='#F7F7F7', edgecolor='0.8')
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(os.path.join(save_dir, 'latent_trajectory_pca_train.svg'), dpi=200)
    plt.close()


def pca_latent_residual(model, X_train, seq_train, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)

    # ===== get latent states =====
    with torch.no_grad():
        enc_hidden, enc_lat = model.encoder(X_train)
        lat_hidden, _ = model.latent(enc_lat)

    h = lat_hidden.cpu().numpy()  # [T, B, H]
    T, B, H = h.shape

    seq_chars = np.asarray([list(s) for s in np.asarray(seq_train)])  # [B, T]
    T = min(T, seq_chars.shape[1])
    h, seq_chars = h[:T], seq_chars[:, :T]

    # ===== residual 1: remove token mean =====
    h_token_res = h.copy()
    for tok in np.unique(seq_chars):
        mask = (seq_chars == tok).T
        if mask.any():
            mu = h[mask].mean(0)
            h_token_res[mask] -= mu

    # ===== residual 2: remove timestep drift =====
    drift = h.mean(axis=1, keepdims=True)
    h_minus_drift = h - drift

    # ===== PCA (fit once on original representation) =====
    pca = PCA(n_components=2)
    pca.fit(h.reshape(T * B, H))

    def project(x):
        z = pca.transform(x.reshape(T * B, H))
        return z.reshape(T, B, 2)

    z_token_res = project(h_token_res)
    z_minus_drift = project(h_minus_drift)

    # ===== plot 1: time gradient =====
    def plot_time(z, name):
        cmap = plt.get_cmap("Blues")
        fig, ax = plt.subplots(figsize=(PLOT_SIZE, PLOT_SIZE))

        for t in range(T):
            frac = t / max(T - 1, 1)
            ax.scatter(
                z[t, :, 0], z[t, :, 1],
                color=cmap(0.2 + 0.6 * frac),
                alpha=0.25 + 0.7 * (frac ** 1.8),
                s=22, linewidths=0
            )

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.grid(True, alpha=0.25)
        ax.set_aspect('equal')
        ax.set_box_aspect(1)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, name), dpi=200)
        plt.close(fig)
        

    # ===== plot 2: token + timestep =====
    def plot_token_time(z, name):
        colors = ['#5b8e7d','#c97c5d','#6c8ebf','#b07aa1','#9e9d57','#6fa3a3']
        tokens = np.sort(np.unique(seq_chars))
        tok_color = {t: colors[i % len(colors)] for i, t in enumerate(tokens)}

        fig, ax = plt.subplots(figsize=(3, 3))

        for t in range(T):
            frac = t / max(T - 1, 1)
            alpha = 0.25 + 0.65 * (frac ** 1.5)
            for tok in tokens:
                mask = seq_chars[:, t] == tok
                if mask.any():
                    ax.scatter(
                        z[t, mask, 0], z[t, mask, 1],
                        color=tok_color[tok],
                        alpha=alpha,
                        s=22, linewidths=0
                    )

        # add legend for tokens
        legend_handles = [plt.Line2D([0], [0], color=tok_color[t], marker='o', lw=0, markersize=5, label=str(t)) for t in tokens]
        # arrange token legend in 2 x 3 layout (3 columns)
        ax.legend(handles=legend_handles, title='Token', fontsize=8, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.grid(True, alpha=0.25)
        ax.set_aspect('equal')
        ax.set_box_aspect(1)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, name), dpi=200)
        plt.close(fig)

    plot_time(z_token_res, 'latent_token_residual.svg')
    plot_token_time(z_minus_drift, 'latent_minus_drift.svg')



def train_linear_decoders(model, X_train, seq_train, X_test, seq_test, save_dir=SAVE_DIR,
    n_epochs=500, lr=1e-2, latent_hidden_train_list=None, latent_hidden_test_list=None):
    os.makedirs(save_dir, exist_ok=True)

    if latent_hidden_train_list is None or latent_hidden_test_list is None:
        model.eval()
        with torch.no_grad():
            _, enc_lat_train = model.encoder(X_train)
            lat_hidden_train, _ = model.latent(enc_lat_train)
            _, enc_lat_test = model.encoder(X_test)
            lat_hidden_test, _ = model.latent(enc_lat_test)
        latent_hidden_train_list = [lat_hidden_train.detach().cpu().numpy().astype(np.float64)]
        latent_hidden_test_list = [lat_hidden_test.detach().cpu().numpy().astype(np.float64)]

    if len(latent_hidden_train_list) != len(latent_hidden_test_list):
        raise ValueError('latent_hidden_train_list and latent_hidden_test_list must have the same length.')
    if len(latent_hidden_train_list) == 0:
        raise ValueError('At least one hidden-state set is required for linear decoder training.')

    T, B_train, H = latent_hidden_train_list[0].shape
    B_test = latent_hidden_test_list[0].shape[1]
    
    letter_to_idx = {l: i for i, l in enumerate(string.ascii_lowercase[:alpha])}
    token_labels_train = np.array([[letter_to_idx[c] for c in seq] for seq in seq_train])
    token_labels_flat_train = torch.tensor(token_labels_train.reshape(-1), dtype=torch.long).to(device)
    token_labels_test = np.array([[letter_to_idx[c] for c in seq] for seq in seq_test])
    token_labels_flat_test = torch.tensor(token_labels_test.reshape(-1), dtype=torch.long).to(device)
    
    position_labels_flat_train = torch.tensor(np.tile(np.arange(T), B_train), dtype=torch.long).to(device)
    position_labels_flat_test = torch.tensor(np.tile(np.arange(T), B_test), dtype=torch.long).to(device)

    token_train_histories, token_test_histories = [], []
    position_train_histories, position_test_histories = [], []

    for lat_hidden_train_np, lat_hidden_test_np in zip(latent_hidden_train_list, latent_hidden_test_list):
        hidden_flat_train = torch.tensor(
            lat_hidden_train_np.transpose(1, 0, 2).reshape(T * B_train, H),
            dtype=torch.float,
            device=device,
        )
        hidden_flat_test = torch.tensor(
            lat_hidden_test_np.transpose(1, 0, 2).reshape(T * B_test, H),
            dtype=torch.float,
            device=device,
        )

        token_train_acc_history, token_test_acc_history = [], []
        token_decoder = torch.nn.Linear(H, alpha).to(device)
        optimizer_token = torch.optim.Adam(token_decoder.parameters(), lr=lr)
        for _ in range(n_epochs):
            token_decoder.train()
            optimizer_token.zero_grad()
            logits = token_decoder(hidden_flat_train)
            loss = F.cross_entropy(logits, token_labels_flat_train)
            loss.backward()
            optimizer_token.step()

            token_decoder.eval()
            with torch.no_grad():
                logits_train = token_decoder(hidden_flat_train)
                pred_train = torch.argmax(logits_train, dim=-1)
                train_acc = (pred_train == token_labels_flat_train).float().mean().item()
                token_train_acc_history.append(train_acc)

                logits_test = token_decoder(hidden_flat_test)
                pred_test = torch.argmax(logits_test, dim=-1)
                test_acc = (pred_test == token_labels_flat_test).float().mean().item()
                token_test_acc_history.append(test_acc)

        position_train_acc_history, position_test_acc_history = [], []
        position_decoder = torch.nn.Linear(H, T).to(device)
        optimizer_pos = torch.optim.Adam(position_decoder.parameters(), lr=lr)
        for _ in range(n_epochs):
            position_decoder.train()
            optimizer_pos.zero_grad()
            logits = position_decoder(hidden_flat_train)
            loss = F.cross_entropy(logits, position_labels_flat_train)
            loss.backward()
            optimizer_pos.step()

            position_decoder.eval()
            with torch.no_grad():
                logits_train = position_decoder(hidden_flat_train)
                pred_train = torch.argmax(logits_train, dim=-1)
                train_acc = (pred_train == position_labels_flat_train).float().mean().item()
                position_train_acc_history.append(train_acc)

                logits_test = position_decoder(hidden_flat_test)
                pred_test = torch.argmax(logits_test, dim=-1)
                test_acc = (pred_test == position_labels_flat_test).float().mean().item()
                position_test_acc_history.append(test_acc)

        token_train_histories.append(token_train_acc_history)
        token_test_histories.append(token_test_acc_history)
        position_train_histories.append(position_train_acc_history)
        position_test_histories.append(position_test_acc_history)

    token_train_acc_history = np.mean(np.asarray(token_train_histories), axis=0)
    token_test_acc_history = np.mean(np.asarray(token_test_histories), axis=0)
    position_train_acc_history = np.mean(np.asarray(position_train_histories), axis=0)
    position_test_acc_history = np.mean(np.asarray(position_test_histories), axis=0)

    token_test_final_all = np.asarray([h[-1] for h in token_test_histories], dtype=np.float64)
    position_test_final_all = np.asarray([h[-1] for h in position_test_histories], dtype=np.float64)

    print(f"\nLinear Decoder Results (Test Set, mean over {len(latent_hidden_train_list)} seeds):")
    print(f"  Token prediction accuracy: {token_test_final_all.mean():.4f} +- {token_test_final_all.std(ddof=0):.4f}")
    print(f"  Position prediction accuracy: {position_test_final_all.mean():.4f} +- {position_test_final_all.std(ddof=0):.4f}")
    
    def smooth(y, window=5):
        if len(y) < window:
            return np.array(y)
        smoothed = []
        for i in range(len(y)):
            start = max(0, i - window // 2)
            end = min(len(y), i + window // 2 + 1)
            smoothed.append(np.mean(y[start:end]))
        return np.array(smoothed)
    
    fig, axes = plt.subplots(1, 2, figsize=(2 * PLOT_SIZE, PLOT_SIZE))
    epochs_x = np.arange(1, n_epochs + 1)
    token_chance = 1.0 / alpha
    position_chance = 1.0 / T
    
    axes[0].plot(epochs_x, smooth(token_train_acc_history), label='Train (mean)', linewidth=1.5)
    axes[0].plot(epochs_x, smooth(token_test_acc_history), label='Test (mean)', linewidth=1.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Token')
    axes[0].set_ylim(0, 1)
    axes[0].axhline(token_chance, color='0.5', linestyle='--', linewidth=1.2, label='Chance')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(epochs_x, smooth(position_train_acc_history), label='Train (mean)', linewidth=1.5)
    axes[1].plot(epochs_x, smooth(position_test_acc_history), label='Test (mean)', linewidth=1.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Position')
    axes[1].set_ylim(0, 1)
    axes[1].axhline(position_chance, color='0.5', linestyle='--', linewidth=1.2, label='Chance')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'linear_decoder_accuracy.svg'), dpi=200)
    plt.close()
    
    return {
        'token_acc_mean': float(token_test_final_all.mean()),
        'token_acc_std': float(token_test_final_all.std(ddof=0)),
        'position_acc_mean': float(position_test_final_all.mean()),
        'position_acc_std': float(position_test_final_all.std(ddof=0)),
    }


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

    fig, axes = plt.subplots(1, 4, figsize=(4 * PLOT_SIZE, PLOT_SIZE))
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
        ax.set_xticklabels(group_labels, rotation=45, ha='right', fontsize=PLOT_FONT)
        ax.set_yticks(centers)
        ax.set_yticklabels(group_labels, fontsize=PLOT_FONT)
        ax.set_xlabel(f't{t + 1}', fontsize=PLOT_FONT, labelpad=10)

    # Hide unused axes if sequence length < 4.
    for j in range(n_show, len(axes)):
        axes[j].axis('off')

    fig.subplots_adjust(right=0.9, wspace=0.3, top=0.92, bottom=0.18)
    cbar_ax = fig.add_axes([0.915, 0.2, 0.009, 0.62])
    cbar = fig.colorbar(shared_im, cax=cbar_ax)
    cbar.set_label('Euclidean distance', fontsize=PLOT_FONT)
    cbar.ax.tick_params(labelsize=PLOT_FONT)
    fig.savefig(os.path.join(save_dir, 'euclidean_matrix_train.svg'), dpi=200)
    plt.close(fig)


#  -------------------------------------------------------
#              Dimensionality & Connectivity
#  -------------------------------------------------------
                   
# Singular Value Curves
def singular_value_curve(model=None, save_dir=SAVE_DIR, block_weight_list=None,
    low_pct=5, high_pct=95, matrices=None, block_order=None, titles=None, out_name=None):
    def _orthogonal_similarity_align(weight_list):

        if len(weight_list) == 0:
            return []

        # Reference SVD (use full matrices to get square U and V bases)
        w0 = weight_list[0]
        if w0.ndim != 2:
            raise ValueError('Each weight matrix must be 2D for similarity alignment.')
        U_ref, _, Vh_ref = np.linalg.svd(w0, full_matrices=True)
        V_ref = Vh_ref.T

        aligned = []
        for w in weight_list:
            if w.ndim != 2:
                raise ValueError('Each weight matrix must be 2D for similarity alignment.')
            U_cur, _, Vh_cur = np.linalg.svd(w, full_matrices=True)
            V_cur = Vh_cur.T
            # q_left: aligns left singular vectors (rows), q_right: aligns right singular vectors (cols)
            q_left = U_ref @ U_cur.T
            q_right = V_ref @ V_cur.T
            aligned.append(q_left @ w @ q_right.T)
        return aligned

    # If caller provided a block_weight_list (from ensemble), use it directly.
    if block_weight_list is None or len(block_weight_list) == 0:
        block_weight_list = []
        # We convert matrices[name] into a list of weight arrays per seed
        max_seeds = 0
        for name in block_order:
            val = matrices[name]
            if isinstance(val, (list, tuple)):
                n = len(val)
            else:
                n = 1
            max_seeds = max(max_seeds, n)

        for seed_idx in range(max_seeds):
            bw = {}
            for name in block_order:
                val = matrices[name]
                if isinstance(val, (list, tuple)):
                    arr = val[seed_idx] if seed_idx < len(val) else val[0]
                else:
                    arr = val
                bw[name] = np.asarray(arr, dtype=np.float64)
            block_weight_list.append(bw)

    legend_handles = [
        plt.Line2D([0], [0], color='#2E5CB8', linestyle='--', linewidth=3.2,
            label=r'$\langle SV(W) \rangle_k$'),
        plt.Line2D([0], [0], color='#2E5CB8', linestyle=':', linewidth=3.2,
            label=r'$SV\langle W \rangle_k$'),
        plt.Line2D([0], [0], color='#C23B3B', linestyle='-', linewidth=3.2,
            marker='o', markersize=4,
            label=r'$SV\langle W_{aligned} \rangle_k$'),
    ]

    block_order = list(block_weight_list[0].keys())
    n_blocks = len(block_order)
    fig, axes = plt.subplots(1, n_blocks, figsize=(PLOT_SIZE * n_blocks, PLOT_SIZE), sharey=False)
    axes = np.atleast_1d(axes)
    plt.rcParams['font.size'] = PLOT_FONT

    for ax, block_name in zip(axes, block_order):
        w_list = [bw[block_name] for bw in block_weight_list]
        sv_stack = np.asarray([np.linalg.svd(w, compute_uv=False) for w in w_list], dtype=np.float64)
        aligned_w_list = _orthogonal_similarity_align(w_list)

        sv_mean = np.nanmean(sv_stack, axis=0)
        sv_low = np.nanpercentile(sv_stack, low_pct, axis=0)
        sv_high = np.nanpercentile(sv_stack, high_pct, axis=0)

        w_avg = np.mean(np.stack(w_list, axis=0), axis=0)
        sv_avg = np.linalg.svd(w_avg, compute_uv=False)
        w_aligned_avg = np.mean(np.stack(aligned_w_list, axis=0), axis=0)
        sv_aligned_avg = np.linalg.svd(w_aligned_avg, compute_uv=False)

        max_index = min(10, sv_mean.shape[0])
        sv_mean = sv_mean[:max_index]
        sv_low = sv_low[:max_index]
        sv_high = sv_high[:max_index]
        sv_avg = sv_avg[:max_index]
        sv_aligned_avg = sv_aligned_avg[:max_index]

        x = np.arange(1, max_index + 1)
        ax.fill_between(x, sv_low, sv_high, color='#4A74C9', alpha=0.22, linewidth=0)
        ax.plot(x, sv_mean, color='#2E5CB8', linestyle='--', linewidth=3.2)
        ax.plot(x, sv_avg, color='#2E5CB8', linestyle=':', linewidth=3.2)
        ax.plot(x, sv_aligned_avg, color='#C23B3B', linestyle='-', linewidth=3.0,
            marker='o', markersize=4.2, alpha=0.9, zorder=3)

        if max_index >= 2:
            ax.axvspan(0.0, 2.0, color='#D94848', alpha=0.10)

        ax.set_title(titles.get(block_name, block_name), fontsize=PLOT_FONT, fontweight='bold')
        ax.set_xlabel('index', fontsize=PLOT_FONT)
        ax.set_xlim(0, max_index)
        desired_ticks = [t for t in (2, 5, 10) if t <= max_index]
        if len(desired_ticks) == 0:
            desired_ticks = [max_index]
        ax.set_xticks(desired_ticks)
        ax.tick_params(axis='both', labelsize=PLOT_FONT)
        ax.grid(False)

    axes[0].set_ylabel('singular value', fontsize=PLOT_FONT)
    axes[-1].legend(handles=legend_handles, fontsize=PLOT_FONT, loc='upper right',
        frameon=True, framealpha=0.9, facecolor='#F7F7F7', edgecolor='0.8')
    fig.tight_layout()
    filename = 'singular_value_curves.svg' if out_name is None else f"{out_name}.svg"
    fig.savefig(os.path.join(save_dir, filename), dpi=200)
    plt.close(fig)


# Effective Dimensionality 
def hidden_participation_ratio(model, X, labels=None, types=None, save_dir=SAVE_DIR, eps=1e-12, block_hidden_list=None):
    os.makedirs(save_dir, exist_ok=True)

    if block_hidden_list is None or len(block_hidden_list) == 0:
        with torch.no_grad():
            _, enc_lat = model.encoder(X)
            lat_hidden, _ = model.latent(enc_lat)
        block_hidden_list = [{
            'latent': lat_hidden.detach().cpu().numpy(),
        }]

    sample_count = block_hidden_list[0]['latent'].shape[1]
    if labels is None:
        labels_np = np.zeros(sample_count, dtype=int)
    else:
        labels_np = np.asarray(labels)
        if labels_np.shape[0] != sample_count:
            raise ValueError('labels length must match batch size in X.')

    class_ids = np.unique(labels_np)
    class_indices = [np.where(labels_np == c)[0] for c in class_ids]
    class_indices = [idx for idx in class_indices if idx.size > 0]
    if len(class_indices) == 0:
        class_indices = [np.arange(sample_count)]

    class_names = []
    for c in class_ids:
        if types is not None and int(c) < len(types):
            class_names.append(str(types[int(c)]))
        else:
            class_names.append(f'class {int(c)}')
    if len(class_names) != len(class_indices):
        class_names = [f'class {i}' for i in range(len(class_indices))]

    fig, ax = plt.subplots(figsize=(PLOT_SIZE+0.5, PLOT_SIZE))
    axes = [ax]
    cmap = plt.get_cmap('tab10')
    legend_handles = [
        plt.Line2D([0], [0], color=cmap(i % 10), lw=1.6, alpha=0.9, label=class_names[i])
        for i in range(len(class_indices))
    ]

    for ax, block_name in zip(axes, ['latent']):
        hidden_np = block_hidden_list[0][block_name]
        time_steps, _, _ = hidden_np.shape
        x = np.arange(1, time_steps + 1)

        # Draw one participation-ratio curve per class and summarize with mean.
        pr_by_class = np.full((len(class_indices), time_steps), np.nan, dtype=np.float64)
        for cls_i, idx in enumerate(class_indices):
            pr_seed_values = []
            for block_hidden in block_hidden_list:
                hidden_seed = block_hidden[block_name]
                pr_values = []
                for t in range(time_steps):
                    states_t = hidden_seed[t, idx, :]
                    if states_t.shape[0] < 2:
                        pr_values.append(np.nan)
                        continue

                    centered = states_t - states_t.mean(axis=0, keepdims=True)
                    singular_values = np.linalg.svd(centered, compute_uv=False)
                    lambdas = singular_values ** 2
                    sum_lambda = lambdas.sum()
                    if sum_lambda <= eps:
                        pr_values.append(0.0)
                        continue
                    denom = np.sum(lambdas ** 2)
                    if denom <= eps:
                        pr_values.append(0.0)
                        continue
                    pr_values.append(float((sum_lambda ** 2) / denom))
                pr_seed_values.append(pr_values)

            pr_arr = np.asarray(pr_seed_values, dtype=np.float64)  # (n_seeds, time_steps)
            pr_values_avg = np.nanmean(pr_arr, axis=0)
            pr_by_class[cls_i, :] = pr_values_avg
            color = cmap(cls_i % 10)
            ax.plot(x, pr_values_avg, color=color, linewidth=1.8, alpha=1.0, linestyle='-')

        mean_curve = np.nanmean(pr_by_class, axis=0)
        std_curve = np.nanstd(pr_by_class, axis=0, ddof=0)
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                        color='gray', alpha=0.18, linewidth=0)
        ax.plot(x, mean_curve, color='gray', linestyle='--', linewidth=2.0, alpha=0.95, label='All')

        ax.set_xlabel('Timestep')
        ax.set_xticks(x)
        ax.set_xlim(1, time_steps)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Participation ratio')
    all_handle = plt.Line2D([0], [0], color='gray', linestyle='--', lw=2.0, alpha=0.95, label='All')
    axes[-1].legend(handles=legend_handles + [all_handle], fontsize=PLOT_FONT, loc='upper right',
        frameon=True, framealpha=0.5, edgecolor='0.75')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'participation_ratio.svg'), dpi=200)
    plt.close(fig)



def _get_block_hidden_and_weight(model, X):
    with torch.no_grad():
        enc_hidden, enc_lat = model.encoder(X)
        lat_hidden, lat_out = model.latent(enc_lat)
        dec_hidden, _ = model.decoder.rnn(lat_out)

    return {
        'encoder': {
            'hidden': enc_hidden.detach().cpu().numpy().astype(np.float64),
            'weight': model.encoder.rnn.h2h.weight.detach().cpu().numpy().astype(np.float64),
        },
        'latent': {
            'hidden': lat_hidden.detach().cpu().numpy().astype(np.float64),
            'weight': model.latent.h2h.weight.detach().cpu().numpy().astype(np.float64),
        },
        'decoder': {
            'hidden': dec_hidden.detach().cpu().numpy().astype(np.float64),
            'weight': model.decoder.rnn.h2h.weight.detach().cpu().numpy().astype(np.float64),
        },
    }



def confusion_matrices(pred, target, n_classes=None, labels=None,save_path=None, vmax=None, cmap='cividis'):

    # convert to numpy (ensure ints)
    if hasattr(pred, 'cpu'):
        pred_np = pred.detach().cpu().numpy()
    else:
        pred_np = np.array(pred)
    if hasattr(target, 'cpu'):
        target_np = target.detach().cpu().numpy()
    else:
        target_np = np.array(target)

    pred_np = pred_np.astype(int)
    target_np = target_np.astype(int)

    # flatten over time and batch to aggregate counts
    if pred_np.ndim == 1:
        y_pred_all = pred_np
        y_true_all = target_np
    else:
        y_pred_all = pred_np.reshape(-1)
        y_true_all = target_np.reshape(-1)

    labels = list(labels) if labels is not None else list(range(n_classes))

    counts = np.zeros((n_classes, n_classes), dtype=int)
    for yt, yp in zip(y_true_all, y_pred_all):
        counts[int(yt), int(yp)] += 1

    # Convert to proportions per true class (row-normalized) in [0,1]
    counts_f = counts.astype(float)
    row_sums = counts_f.sum(axis=1, keepdims=True)
    # avoid div-by-zero
    row_sums[row_sums == 0] = 1.0
    props = counts_f / row_sums

    # prepare plot
    if save_path is None:
        save_path = os.path.join(SAVE_DIR, 'confusion_matrix.svg')

    fig, ax = plt.subplots(figsize=(3.2, 3))
    im = ax.imshow(props, interpolation='nearest', cmap=cmap, aspect='equal', vmin=0.0, vmax=1.0)

    Cdim = props.shape[0]
    for r in range(Cdim):
        for c in range(Cdim):
            val = props[r, c]
            txt = f'{val:.2f}' if val > 0 else ''
            color = 'white' if val < 0.7 else 'gray'
            ax.text(c, r, txt, ha='center', va='center', color=color, fontsize=8)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0, fontsize=PLOT_FONT)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=PLOT_FONT)
    ax.set_xlabel('Reconstructed', fontsize=PLOT_FONT)
    ax.set_ylabel('True', fontsize=PLOT_FONT)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)



def whh_ablation(model, X_train, X_test, test_labels, train_labels=None, types=None, n_epochs=None):

    # create fresh model to train under ablated condition
    pert = model(alpha, d_hidden, d_latent_hidden, num_layers, d_latent, L).to(device)

    def _zero_and_freeze_whh(m):
        # zero full-rank weight if present
        with torch.no_grad():
            if hasattr(m, 'weight') and isinstance(getattr(m, 'weight'), torch.Tensor):
                try:
                    m.weight.data.zero_()
                except Exception:
                    pass
            if hasattr(m, 'U') and getattr(m, 'U') is not None:
                try:
                    m.U.data.zero_()
                except Exception:
                    pass
            if hasattr(m, 'V') and getattr(m, 'V') is not None:
                try:
                    m.V.data.zero_()
                except Exception:
                    pass
        # freeze parameters so they remain zero during training
        for p in m.parameters():
            p.requires_grad = False

    # zero & freeze encoder, latent, decoder Whh if available
    try:
        _zero_and_freeze_whh(pert.encoder.rnn.h2h)
    except Exception:
        pass
    try:
        _zero_and_freeze_whh(pert.latent.h2h)
    except Exception:
        pass
    try:
        _zero_and_freeze_whh(pert.decoder.rnn.h2h)
    except Exception:
        pass

    # train perturbed model normally (Whh stay zero because frozen)
    train(pert, X_train, X_test, test_labels, train_labels=train_labels, types=types,
          n_epochs=n_epochs, lr=lr, weight_decay=weight_decay, print_final=False)

    # evaluate
    pert.eval()
    with torch.no_grad():
        _, _, output = pert(X_test)
    pred = torch.argmax(output, dim=-1)
    target = torch.argmax(X_test, dim=-1)
    acc = (pred == target).all(dim=0).float().mean().item()
    confusion_matrices(pred, target, n_classes=alpha,
            labels=list(string.ascii_lowercase[:alpha]))
    print(f'[4] W_hh ablation (trained with W_hh=0): acc={acc:.4f}')
    return acc


def heatmap_whh(seeds, save_dir=SAVE_DIR):
    results = np.zeros((len(seeds), 3), dtype=float)

    for i, s in enumerate(seeds):
        print(f"Running seed {s} ({i+1}/{len(seeds)})")
        set_seed(s)

        # generate data deterministically for this seed
        seq_train, seq_test, labels_train, labels_test, types = generate_instances(alpha, L, m, frac_train=0.8)
        X_train = sequences_to_tensor(seq_train, alpha).to(device)
        X_test = sequences_to_tensor(seq_test, alpha).to(device)
        test_labels = torch.tensor(labels_test, dtype=torch.long)
        train_labels = torch.tensor(labels_train, dtype=torch.long)

        # create and train model
        model = RNNAutoencoder(alpha, d_hidden, d_latent_hidden, num_layers, d_latent, L).to(device)
        train(model, X_train, X_test, test_labels, train_labels=train_labels, types=types,
                        n_epochs=1000, lr=lr, weight_decay=weight_decay, print_final=False)

        # compute accuracies for several conditions via loop
        conditions = [
            ('normal', []),
            ('all', ['encoder', 'latent', 'decoder']),
            ('trained_whh_zero', None),
        ]

        # compute target once
        with torch.no_grad():
            target = torch.argmax(X_test, dim=-1)

        for j, (name, zero_parts) in enumerate(conditions):
            if name == 'trained_whh_zero':
                acc = whh_ablation(model, X_train, X_test, test_labels, train_labels=train_labels, types=types, n_epochs=1000)
                results[i, j] = acc
                continue
            if len(zero_parts) == 0:
                with torch.no_grad():
                    _, _, out = model(X_test)
            else:
                with torch.no_grad():
                    pert = copy.deepcopy(model)
                    if 'encoder' in zero_parts:
                        pert.encoder.rnn.h2h.weight.data.zero_()
                    if 'latent' in zero_parts:
                        pert.latent.h2h.weight.data.zero_()
                    if 'decoder' in zero_parts:
                        pert.decoder.rnn.h2h.weight.data.zero_()
                    _, _, out = pert(X_test)

            pred = torch.argmax(out, dim=-1)
            acc = (pred == target).all(dim=0).float().mean().item()
            results[i, j] = acc

    # sort rows by normal accuracy (column 0) descending, then plot heatmap
    order = np.argsort(results[:, 0])[::-1]
    results_sorted = results[order, :]

    fig, ax = plt.subplots(figsize=(3, 3.5))
    im = ax.imshow(results_sorted, cmap='cividis', aspect='auto', vmin=0.0, vmax=1.0)
    # x ticks: three conditions (baseline, all-zero, retrained Whh=0)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(['normal', 'all', 'trained'], rotation=45, ha='right')
    ax.set_ylabel('Simulation')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plot_path = os.path.join(save_dir, 'whh_seeds_heatmap.svg')
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Saved heatmap to {plot_path}")
    return results_sorted


def Whh_test(model, X_train, X_test, seq_train, seq_test, save_dir=SAVE_DIR):

    pert = copy.deepcopy(model)
    pert.encoder.rnn.h2h.weight.data.zero_()
    pert.latent.h2h.weight.data.zero_()
    pert.decoder.rnn.h2h.weight.data.zero_()

    with torch.no_grad():
        _, _, test_output_pert = pert(X_test)
        pred_test = torch.argmax(test_output_pert, dim=-1)
        target_test = torch.argmax(X_test, dim=-1)
    acc = (pred_test == target_test).all(dim=0).float().mean().item()
    confusion_matrices(pred_test, target_test, n_classes=alpha, labels=list(string.ascii_lowercase[:alpha]))

    # Use test data for downstream analyses when evaluating test-time W=0
    trajectory_pca_latent(pert, X_test, seq_test, save_dir)
    bwp = _get_block_hidden_and_weight(pert, X_test)
    # singular_value_curve(save_dir,
    #                      matrices={'W_enc': bwp['encoder']['weight'], 'W_lat': bwp['latent']['weight'], 'W_dec': bwp['decoder']['weight']},
    #                      block_order=['W_enc', 'W_lat', 'W_dec'],
    #                      titles={'W_enc': r'Encoder $W_{hh}$', 'W_lat': r'Latent $W_{hh}$', 'W_dec': r'Decoder $W_{hh}$'},
    #                      out_name='singular_values')

    print(f'[Whh_test] accuracy: {acc:.4f}')


def train(model, X_train, X_test, test_labels, train_labels=None, types=None,
    n_epochs=300, lr=0.001, weight_decay=1e-3, print_final=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {k: [] for k in ['train_loss', 'test_loss', 'train_acc', 'test_acc']}
 
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
            
            if train_labels is not None and epoch == n_epochs - 1:
                if latent.ndim == 3:
                    latent_2d = latent.permute(1, 0, 2).reshape(latent.shape[1], -1)
                else:
                    latent_2d = latent

                latent_np = latent_2d.cpu().numpy()
                labels_np = train_labels.cpu().numpy()
                unique_labels = np.unique(labels_np)

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

        # history - only save final epoch values to reduce memory
        if epoch == n_epochs - 1:
            history['train_loss'].append(total_loss.item())
            history['test_loss'].append(float(test_loss))
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)

        
        # Print at the final epoch when enabled.
        if print_final and epoch == n_epochs - 1:
            print(f"\nEpoch {epoch+1}/{n_epochs}:")
            print(f"  Train - Loss: {total_loss.item():.4f}, Acc: {train_acc:.4f}")
            print(f"  Test  - Loss: {test_loss.item():.4f}, Acc: {test_acc:.4f}")

    return history


def compute_metrics(train_metrics, test_metrics, latent_test, test_labels, types):
    print('\n' + '='*60)
    print('='*60)
    print(f"Train acc: {train_metrics['acc']:.4f}")
    print(f"Test acc: {test_metrics['acc']:.4f}")


def _build_ensemble_block_lists(ensemble_seeds, X_train, X_test, test_labels, train_labels, types):
    block_hidden_weight_list = []
    block_weight_list = []
    block_hidden_list = []
    latent_hidden_train_list = []
    latent_hidden_test_list = []

    for seed_i in ensemble_seeds:
        set_seed(seed_i)
        model_k = RNNAutoencoder(alpha, d_hidden, d_latent_hidden, num_layers, d_latent, L).to(device)
        train(
            model_k,
            X_train,
            X_test,
            test_labels,
            train_labels=train_labels,
            types=types,
            n_epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            print_final=False,
        )

        block_hidden_weight = _get_block_hidden_and_weight(model_k, X_test)
        block_hidden_weight_list.append(block_hidden_weight)
        block_weight_list.append({
            'encoder': block_hidden_weight['encoder']['weight'],
            'latent': block_hidden_weight['latent']['weight'],
            'decoder': block_hidden_weight['decoder']['weight'],
        })
        block_hidden_list.append({
            'encoder': block_hidden_weight['encoder']['hidden'],
            'latent': block_hidden_weight['latent']['hidden'],
            'decoder': block_hidden_weight['decoder']['hidden'],
        })

        with torch.no_grad():
            _, enc_lat_train = model_k.encoder(X_train)
            lat_hidden_train, _ = model_k.latent(enc_lat_train)
            _, enc_lat_test = model_k.encoder(X_test)
            lat_hidden_test, _ = model_k.latent(enc_lat_test)

        latent_hidden_train_list.append(lat_hidden_train.detach().cpu().numpy().astype(np.float64))
        latent_hidden_test_list.append(lat_hidden_test.detach().cpu().numpy().astype(np.float64))

    return (
        block_hidden_weight_list,
        block_weight_list,
        block_hidden_list,
        latent_hidden_train_list,
        latent_hidden_test_list,
    )


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

    
    # confusion matrices
    with torch.no_grad():
        _, _, train_output = model(X_train)
        pred_train = torch.argmax(train_output, dim=-1)
        target_train = torch.argmax(X_train, dim=-1)
    confusion_matrices(pred_train, target_train, n_classes=alpha, labels=list(string.ascii_lowercase[:alpha]))
    
    # plotting
    
    # ensemble_seeds = range(40, 50)
    # block_hidden_weight_list, block_weight_list, block_hidden_list, latent_hidden_train_list, latent_hidden_test_list = _build_ensemble_block_lists(
    #     ensemble_seeds, X_train, X_test, test_labels, train_labels, types)

    # # Euclidean_matrix(model, X_train, seq_train, save_dir=SAVE_DIR, latent_hidden_list=latent_hidden_train_list)
    # train_linear_decoders(model, X_train, seq_train, X_test, seq_test, save_dir=SAVE_DIR,
    #     n_epochs=400, lr=1e-2, latent_hidden_train_list=latent_hidden_train_list,
    #     latent_hidden_test_list=latent_hidden_test_list)

    # trajectory_pca_latent(model, X_train, seq_train, save_dir=SAVE_DIR)
    # pca_latent_residual(model, X_train, seq_train, save_dir=SAVE_DIR)
    # pca_encoder_input(model, X_train, seq_train, save_dir=SAVE_DIR)

    # Whh_test(model, X_train, X_test, seq_train, seq_test, save_dir=SAVE_DIR)

    # # Whh_ablation
    # seeds = list(range(seed, seed + 20))
    # heatmap_whh(seeds, save_dir=SAVE_DIR)

    # # encoder Win (i2h)
    # W_in = model.encoder.rnn.i2h.weight.detach().cpu().numpy().astype(np.float64)
    # singular_value_curve(matrices={'W_in': W_in}, block_order=['W_in'], titles={'W_in': r'Encoder $W_{in}$'}, save_dir=SAVE_DIR, out_name='singular_values_Win')
    # Whh_enc = model.encoder.rnn.h2h.weight.detach().cpu().numpy().astype(np.float64)
    # singular_value_curve(matrices={'Whh_enc': Whh_enc}, block_order=['Whh_enc'], titles={'Whh_enc': r'Encoder $W_{hh}$'}, save_dir=SAVE_DIR, out_name='singular_values_Whh_encoder')
    # W_out_enc = model.encoder.rnn.h2o.weight.detach().cpu().numpy().astype(np.float64)
    # singular_value_curve(matrices={'W_out_enc': W_out_enc}, block_order=['W_out_enc'], titles={'W_out_enc': r'Encoder $W_{out}$'}, save_dir=SAVE_DIR, out_name='singular_values_Wout_encoder')

    # # latent Win and Wout
    # W_in_lat = model.latent.i2h.weight.detach().cpu().numpy().astype(np.float64)
    # singular_value_curve(matrices={'W_in_lat': W_in_lat}, block_order=['W_in_lat'], titles={'W_in_lat': r'Latent $W_{in}$'}, save_dir=SAVE_DIR, out_name='singular_values_Win_latent')
    # Whh_lat = model.latent.h2h.weight.detach().cpu().numpy().astype(np.float64)
    # singular_value_curve(matrices={'Whh_lat': Whh_lat}, block_order=['Whh_lat'], titles={'Whh_lat': r'Latent $W_{hh}$'}, save_dir=SAVE_DIR, out_name='singular_values_Whh_latent')
    # W_out_lat = model.latent.h2o.weight.detach().cpu().numpy().astype(np.float64)
    # singular_value_curve(matrices={'W_out_lat': W_out_lat}, block_order=['W_out_lat'], titles={'W_out_lat': r'Latent $W_{out}$'}, save_dir=SAVE_DIR, out_name='singular_values_Wout_latent')

    # # decoder
    # W_in_dec = model.decoder.rnn.i2h.weight.detach().cpu().numpy().astype(np.float64)
    # singular_value_curve(matrices={'W_in_dec': W_in_dec}, block_order=['W_in_dec'], titles={'W_in_dec': r'Decoder $W_{in}$'}, save_dir=SAVE_DIR, out_name='singular_values_Win_decoder')
    # Whh_dec = model.decoder.rnn.h2h.weight.detach().cpu().numpy().astype(np.float64)
    # singular_value_curve(matrices={'Whh_dec': Whh_dec}, block_order=['Whh_dec'], titles={'Whh_dec': r'Decoder $W_{hh}$'}, save_dir=SAVE_DIR, out_name='singular_values_Whh_decoder') 
    # W_out = model.decoder.rnn.h2o.weight.detach().cpu().numpy().astype(np.float64)
    # singular_value_curve(matrices={'W_out': W_out}, block_order=['W_out'], titles={'W_out': r'Decoder $W_{out}$'}, save_dir=SAVE_DIR, out_name='singular_values_Wout')
    
    
    # hidden_participation_ratio(model, X_test, labels=labels_test, types=types,
    #     save_dir=SAVE_DIR, block_hidden_list=block_hidden_list)
    
    
    # final evaluation
    model.eval()
    with torch.no_grad():
        _, z_test, test_output = model(X_test)
        pred_test = torch.argmax(test_output, dim=-1)
        target_test = torch.argmax(X_test, dim=-1)
        test_acc = (pred_test == target_test).all(dim=0).float().mean().item()

    # train_acc = history['train_acc'][-1] if len(history['train_acc']) > 0 else 0.0
    # train_metrics = {'acc': train_acc}
    # test_metrics = {'acc': test_acc}

    # compute_metrics(train_metrics, test_metrics, z_test, labels_test, types)

if __name__ == '__main__':
    run_experiment()