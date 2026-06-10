import random
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from model import RNN, RNNAutoencoder
from sequences import findStructures, replace_symbols as seq_replace_symbols
import string
import itertools
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import copy
import matplotlib.pyplot as plt
import os
from analysis_utils import compute_distance


seed = 42
L, m, alpha = 4, 2, 6
epochs = 1000 
lr = 1e-3
d_hidden = 64
d_latent_hidden = 32
d_latent = 32
weight_decay = 1e-3 
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

def pca_h_t(model, X_train, labels=None, types=None, seq_train=None, save_dir=SAVE_DIR):
    with torch.no_grad():
        # Prefer to obtain latent hidden sequence by running encoder then latent RNN
        if hasattr(model, 'encoder') and hasattr(model, 'latent'):
            enc_hidden, enc_latent = model.encoder(X_train)
            lat_hidden, lat_out = model.latent(enc_latent)
            latent_seq = lat_hidden
        else:
            outs = model(X_train)
            if isinstance(outs, tuple) and len(outs) == 3:
                # outs[1] may be latent-out or latent-hidden depending on model; use it
                latent_seq = outs[1]
            else:
                latent_seq = outs[0]
        h_last = latent_seq[-1].detach().cpu().numpy()
        first_token_ids = torch.argmax(X_train[0], dim=-1).detach().cpu().numpy()

    pca = PCA(n_components=min(2, h_last.shape[1]))
    h_2d = pca.fit_transform(h_last)

    labels_np = np.asarray(labels) if labels is not None else np.asarray(first_token_ids)
    unique_labels = np.unique(labels_np)
    muted_palette = ['#5b8e7d', '#c97c5d', '#6c8ebf', '#b07aa1', '#9e9d57', '#6fa3a3']
    colors = muted_palette
    if types is not None:
        token_names = [str(t) for t in types]
    else:
        token_names = list(string.ascii_lowercase[:alpha])

    fig, ax = plt.subplots(figsize=(PLOT_SIZE, PLOT_SIZE))
    for label_idx in unique_labels:
        mask = labels_np == label_idx
        label_name = token_names[int(label_idx)] if int(label_idx) < len(token_names) else f'token {int(label_idx)}'
        ax.scatter(
            h_2d[mask, 0],
            h_2d[mask, 1],
            color=colors[int(label_idx) % len(colors)],
            s=50,
            label=label_name,
            alpha=0.85,
            edgecolors='none',
        )

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)' if h_2d.shape[1] > 1 else 'PC2')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_aspect('equal')
    ax.set_box_aspect(1)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'hidden_last_pca.svg'), dpi=200)
    plt.close(fig)

    if h_last.shape[1] >= 3:
        pca_3d = PCA(n_components=3)
        h_3d = pca_3d.fit_transform(h_last)

        fig = plt.figure(figsize=(PLOT_SIZE + 1, PLOT_SIZE + 1))
        ax3d = fig.add_subplot(111, projection='3d')
        for label_idx in unique_labels:
            mask = labels_np == label_idx
            label_name = token_names[int(label_idx)] if int(label_idx) < len(token_names) else f'type {int(label_idx)}'
            ax3d.scatter(
                h_3d[mask, 0],
                h_3d[mask, 1],
                h_3d[mask, 2],
                color=colors[int(label_idx) % len(colors)],
                s=50,
                label=label_name,
                alpha=0.85,
                edgecolors='none',
            )

        # Add sequence labels to each point
        if seq_train is not None:
            seq_array = np.asarray(seq_train)
            for i in range(h_3d.shape[0]):
                label_text = seq_array[i] if i < len(seq_array) else f'seq{i}'
                ax3d.text(h_3d[i, 0], h_3d[i, 1], h_3d[i, 2], label_text, 
                         fontsize=8, alpha=0.7, ha='center')

        ax3d.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0] * 100:.1f}%)')
        ax3d.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1] * 100:.1f}%)')
        ax3d.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2] * 100:.1f}%)')
        ax3d.legend(fontsize=8, loc='upper right')
        fig.tight_layout()
        plt.show()


def pca_trajectory_level(model, X, labels, types, save_dir=SAVE_DIR):

    with torch.no_grad():
        if hasattr(model, 'encoder') and hasattr(model, 'latent'):
            enc_hidden, enc_latent = model.encoder(X)
            lat_hidden, lat_out = model.latent(enc_latent)
            h = lat_hidden.detach().cpu().numpy()
        else:
            outs = model(X)
            if isinstance(outs, tuple) and len(outs) == 3:
                lat_hidden = outs[1]
            else:
                lat_hidden = outs[0]
            h = lat_hidden.detach().cpu().numpy()  # [T, B, H]
    T, B, H = h.shape

    # Reshape to [B, T*H] - concatenate timesteps for each sequence
    h_trajectory = h.transpose(1, 0, 2).reshape(B, T * H)  # [B, T*H]

    # PCA with 3 components
    pca = PCA(n_components=min(3, h_trajectory.shape[1]))
    h_3d = pca.fit_transform(h_trajectory)  # [B, 3]

    # Color trajectories by type labels (provided via `labels` argument)
    token_labels = np.array(labels) if labels is not None else np.argmax(X[1].detach().cpu().numpy(), axis=-1)
    unique_tokens = np.unique(token_labels)

    fig, ax = plt.subplots(figsize=(PLOT_SIZE, PLOT_SIZE))
    muted_palette = ['#5b8e7d', '#c97c5d', '#6c8ebf', '#b07aa1', '#9e9d57', '#6fa3a3']
    colors = muted_palette
    if types is not None:
        token_names = [str(t) for t in types]
    else:
        token_names = list(string.ascii_lowercase[:alpha])
    for tok in unique_tokens:
        mask = token_labels == tok
        token_name = token_names[int(tok)] if int(tok) < len(token_names) else f'token {int(tok)}'
        ax.scatter(h_3d[mask, 0], h_3d[mask, 1], color=colors[int(tok) % len(colors)], s=50, label=token_name, alpha=0.8, edgecolors='none')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_aspect('equal')
    ax.set_box_aspect(1)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'latent_trajectory_pca.svg'), dpi=200)
    plt.close(fig)

    # 3D visualization (interactive)
    if h_3d.shape[1] >= 3:
        fig = plt.figure(figsize=(PLOT_SIZE + 1, PLOT_SIZE + 1))
        ax3d = fig.add_subplot(111, projection='3d')
        for tok in unique_tokens:
            mask = token_labels == tok
            token_name = token_names[int(tok)] if int(tok) < len(token_names) else f'token {int(tok)}'
            ax3d.scatter(h_3d[mask, 0], h_3d[mask, 1], h_3d[mask, 2],
                        color=colors[int(tok) % len(colors)], s=50, label=token_name,
                        alpha=0.8, edgecolors='none')

        ax3d.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
        ax3d.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
        ax3d.legend(fontsize=8, loc='upper right')
        fig.tight_layout()
        plt.show()


def cosine_similarity(model, X_train, labels, types=None, save_dir=SAVE_DIR):
 
    with torch.no_grad():
        if hasattr(model, 'encoder') and hasattr(model, 'latent'):
            enc_hidden, enc_latent = model.encoder(X_train)
            lat_hidden, lat_out = model.latent(enc_latent)
            h_last = lat_hidden[-1].detach().cpu().numpy()
        else:
            outs = model(X_train)
            if isinstance(outs, tuple) and len(outs) == 3:
                latent = outs[1]
            else:
                latent = outs[0]
            h_last = latent[-1].detach().cpu().numpy()

    labels_np = np.asarray(labels)
       
    sort_idx = np.argsort(labels_np, kind='stable')
    h_sorted = h_last[sort_idx]
    labels_sorted = labels_np[sort_idx]
    unique_labels = np.unique(labels_sorted)

    norms = np.linalg.norm(h_sorted, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    cosine_sim = (h_sorted @ h_sorted.T) / (norms @ norms.T)
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)

    boundaries = []
    centers = []
    type_names = []
    start = 0
    for label_idx in unique_labels:
        count = int(np.sum(labels_sorted == label_idx))
        end = start + count
        boundaries.append(end)
        centers.append((start + end - 1) / 2.0)
        type_names.append(str(types[int(label_idx)]) if types is not None else f'type {int(label_idx)}')
        start = end

    fig, ax = plt.subplots(figsize=(PLOT_SIZE + 0.75, PLOT_SIZE + 0.75))
    im = ax.imshow(cosine_sim, cmap='viridis', vmin=0, vmax=1.0, interpolation='nearest', aspect='equal')

    for boundary in boundaries[:-1]:
        ax.axhline(boundary - 0.5, color='white', linewidth=1.0, alpha=0.9)
        ax.axvline(boundary - 0.5, color='white', linewidth=1.0, alpha=0.9)

    ax.set_xticks(centers)
    ax.set_yticks(centers)
    ax.set_xticklabels(type_names, fontsize=8)
    ax.set_yticklabels(type_names, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Cosine similarity')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'cosine_similarity.svg'), dpi=200)
    plt.close(fig)


def confusion_matrices(pred, target, n_classes=None, labels=None, cmap='cividis'):

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

    labels = list(labels) if labels is not None else list(range(n_classes))
    
    # Handle 3D case: [T, B, C] -> [T, B]
    if pred_np.ndim == 3:
        pred_np = np.argmax(pred_np, axis=-1)
    
    # Ensure target is 2D [T, B]
    if target_np.ndim == 1:
        # If target is 1D, reshape it assuming it comes from 2D
        T = pred_np.shape[0]
        B = pred_np.shape[1]
        target_np = np.tile(target_np, (T, 1))
    
    T, B = pred_np.shape
    
    print(f"Generating {T} confusion matrices in 1x{T} layout...")
    
    # Create 1xT subplot figure
    fig, axes = plt.subplots(1, T, figsize=(4 * T, 3.2))
    if T == 1:
        axes = [axes]  # make it iterable if only 1 subplot
    
    # Create confusion matrix for each timestep
    for t in range(T):
        y_pred_t = pred_np[t, :].reshape(-1)  # [B]
        y_true_t = target_np[t, :].reshape(-1)  # [B]
        
        counts = np.zeros((n_classes, n_classes), dtype=int)
        for yt, yp in zip(y_true_t, y_pred_t):
            counts[int(yt), int(yp)] += 1
        
        # Convert to proportions per true class (row-normalized)
        counts_f = counts.astype(float)
        row_sums = counts_f.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        props = counts_f / row_sums
        
        ax = axes[t]
        im = ax.imshow(props, interpolation='nearest', cmap=cmap, aspect='equal', vmin=0.0, vmax=1.0)
        
        # Add text annotations
        for r in range(n_classes):
            for c in range(n_classes):
                val = props[r, c]
                txt = f'{val:.2f}' if val > 0 else ''
                color = 'white' if val < 0.7 else 'gray'
                ax.text(c, r, txt, ha='center', va='center', color=color, fontsize=8)
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=0, fontsize=PLOT_FONT)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=PLOT_FONT)
        ax.set_xlabel('Reconstructed', fontsize=PLOT_FONT)
        if t == 0:
            ax.set_ylabel('True', fontsize=PLOT_FONT)
        ax.set_title(f'Timestep {t}', fontsize=PLOT_FONT)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    fig.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'confusion_matrices.svg')
    fig.savefig(save_path, dpi=200)
    plt.close(fig)



def train(model, X_train, X_test, test_labels, train_labels=None, types=None,
    n_epochs=300, lr=0.001, weight_decay=1e-3, print_final=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {k: [] for k in ['train_loss', 'test_loss', 'train_acc', 'test_acc']}
 
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        X_batch = X_train
        outs = model(X_batch)
        if isinstance(outs, tuple) and len(outs) == 3:
            hidden, latent, output = outs
        else:
            hidden, output = outs

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

        model.eval()
        with torch.no_grad():
            outs_test = model(X_test)
            if isinstance(outs_test, tuple) and len(outs_test) == 3:
                _, _, test_output = outs_test
            else:
                _, test_output = outs_test

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



def run_experiment():
    set_seed(seed)

    # use RNNAutoencoder for experiments (encode -> latent -> reconstruct)
    model = RNNAutoencoder(d_input=alpha, d_hidden=d_hidden, d_latent_hidden=d_latent_hidden,
                           num_layers=L, d_latent=d_latent, sequence_length=L,
                           nonlinearity='linear', device=device).to(device)

    seq_train, seq_test, labels_train, labels_test, types = generate_instances(alpha, L, m, frac_train=0.8)
    X_train = sequences_to_tensor(seq_train, alpha).to(device)
    X_test = sequences_to_tensor(seq_test, alpha).to(device)
    test_labels = torch.tensor(labels_test, dtype=torch.long)
    train_labels = torch.tensor(labels_train, dtype=torch.long)

    history = train(model, X_train, X_test, test_labels, train_labels=train_labels, types=types,
        n_epochs=epochs, lr=lr, weight_decay=weight_decay)

    
    # confusion matrices
    with torch.no_grad():
        _, _, test_output = model(X_test)
        pred_test = torch.argmax(test_output, dim=-1)
        target_test = torch.argmax(X_test, dim=-1)
    # confusion_matrices(pred_test, target_test, n_classes=alpha, labels=list(string.ascii_lowercase[:alpha]))

    # plotting
    
    # # Euclidean_matrix(model, X_train, seq_train, save_dir=SAVE_DIR, latent_hidden_list=latent_hidden_train_list)
    # train_linear_decoders(model, X_train, seq_train, X_test, seq_test, save_dir=SAVE_DIR,
    #     n_epochs=400, lr=1e-2, latent_hidden_train_list=latent_hidden_train_list,
    #     latent_hidden_test_list=latent_hidden_test_list)

    # cosine_similarity(model, X_train, labels_train, types, save_dir=SAVE_DIR)
    # pca_h_t(model, X_train, labels_train, types, seq_train=seq_train, save_dir=SAVE_DIR)
    # pca_trajectory_level(model, X_train, labels_train, types, save_dir=SAVE_DIR)
    # pca_hidden_by_timestep(model, X_train, labels_train, types, save_dir=SAVE_DIR)


    # final evaluation
    model.eval()
    with torch.no_grad():
        # obtain latent hidden sequence (latent RNN hidden) and reconstructed output
        if hasattr(model, 'encoder') and hasattr(model, 'latent') and hasattr(model, 'decoder'):
            enc_hidden, enc_latent = model.encoder(X_test)
            lat_hidden, lat_out = model.latent(enc_latent)
            z_test = lat_hidden
            test_output = model.decoder(lat_out)
        else:
            outs = model(X_test)
            if isinstance(outs, tuple) and len(outs) == 3:
                _, z_test, test_output = outs
            else:
                z_test, test_output = outs
        pred_test = torch.argmax(test_output, dim=-1)
        target_test = torch.argmax(X_test, dim=-1)
        test_acc = (pred_test == target_test).all(dim=0).float().mean().item()

    train_acc = history['train_acc'][-1] if len(history['train_acc']) > 0 else 0.0
    train_metrics = {'acc': train_acc}
    test_metrics = {'acc': test_acc}

    idx_to_char = list(string.ascii_lowercase[:alpha])
    wrong_indices = []
    print("\nWrong instances on the test set:")
    for b in range(pred_test.shape[1]):
        pred_seq = ''.join(idx_to_char[int(i)] for i in pred_test[:, b].tolist())
        true_seq = seq_test[b]
        if pred_seq != true_seq:
            wrong_indices.append(b)
            print(f"  instance {b}: true={true_seq}, pred={pred_seq}")

    compute_metrics(train_metrics, test_metrics, z_test, labels_test, types)

if __name__ == '__main__':
    run_experiment()