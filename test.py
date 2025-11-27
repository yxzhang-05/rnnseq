import random
import numpy as np
import torch
import torch.nn.functional as F
from model import RNNAutoencoder
from sequences import findStructures, replace_symbols as seq_replace_symbols
import string
import itertools
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import os


# =====================================
ENABLE_SERIALIZE_Z = False
ENABLE_CONTRASTIVE = False      
CONTRASTIVE_WEIGHT = 2.5   
# =====================================

seed = 42
L, m, alpha = 4, 2, 6
epochs = 1000 
lr = 1e-3
d_hidden = 32
d_latent = 8  
weight_decay = 1e-3 
num_layers = 1
device = torch.device('cpu')
SAVE_DIR = "results"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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

    return (np.array(train_seqs), np.array(test_seqs), np.array(train_labels), np.array(test_labels), types,)


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


def contrastive_loss(latent, X_batch):
   
    L, B, alpha = X_batch.shape
    
    # extract position matrix
    seq_indices = torch.argmax(X_batch, dim=-1).T  # (B, L)
    
    # build relation matrix (B, L, L)
    relation_matrices = []
    for b in range(B):
        seq = seq_indices[b]  # (L,)
        # relation[i,j] = 1 if seq[i] == seq[j]
        relation = (seq.unsqueeze(0) == seq.unsqueeze(1)).float()
        relation_matrices.append(relation)
    
    relation_matrices = torch.stack(relation_matrices)  # (B, L, L)
    relation_vectors = relation_matrices.view(B, -1)  # (B, L*L)   
    relation_sim = torch.matmul(relation_vectors, relation_vectors.T)  # (B, B)

    norm = torch.sqrt((relation_vectors ** 2).sum(dim=1, keepdim=True))
    relation_sim = relation_sim / (norm @ norm.T + 1e-8)

    latents_norm = F.normalize(latent, p=2, dim=1)
    latent_sim = torch.matmul(latents_norm, latents_norm.T) / 0.5
    
    exp_sim = torch.exp(latent_sim)
    weighted_positive = (exp_sim * relation_sim).sum(dim=1)
    all_sum = exp_sim.sum(dim=1) - torch.diag(exp_sim)  

    loss = -torch.log(weighted_positive / (all_sum + 1e-8) + 1e-8)
    loss = loss.mean()
    
    return loss


def train(model, X_train, X_test, test_labels,
                               n_epochs=300, lr=0.001, weight_decay=0,
                               use_contrastive=True, contrastive_weight=0.5):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {k: [] for k in ['train_loss', 'ce_loss', 'contrastive_loss', 'test_loss',
                               'train_acc', 'test_acc', 'silhouette']}
 
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

        contrast_loss = torch.tensor(0.0)
        if use_contrastive:
            contrast_loss = contrastive_loss(latent, X_batch)
            total_loss = total_loss + contrastive_weight * contrast_loss

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
            _, test_latent, test_output = model(X_test)
            test_ce_loss = F.cross_entropy(
                test_output.reshape(-1, test_output.shape[-1]),
                torch.argmax(X_test, dim=-1).reshape(-1)
            )
            test_loss = test_ce_loss

            pred_test = torch.argmax(test_output, dim=-1)
            target_test = torch.argmax(X_test, dim=-1)
            test_acc = (pred_test == target_test).all(dim=0).float().mean().item()            

            sil = silhouette_score(test_latent.cpu().numpy(), test_labels.cpu().numpy())

        # history
        history['train_loss'].append(total_loss.item())
        history['ce_loss'].append(ce_loss.item())
        history['contrastive_loss'].append(contrast_loss.item() if isinstance(contrast_loss, torch.Tensor) else 0)
        history['test_loss'].append(float(test_loss))
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['silhouette'].append(sil)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Loss={total_loss.item():.4f} "
                  f"(CE={ce_loss.item():.4f} Contr={contrast_loss.item():.4f} "
                  f"Acc={train_acc:.3f} | "
                  f"Test: Loss={float(test_loss):.4f} "
                  f"Acc={test_acc:.3f} Sil={sil:.3f}")

    return history


def compute_metrics(train_metrics, test_metrics, latent_test, test_labels, types):
    print('\n' + '='*60)
    print('='*60)
    print(f"Train acc: {train_metrics['acc']:.4f}")
    print(f"Test acc: {test_metrics['acc']:.4f}")
    
    # silhouette
    latent_np = latent_test.cpu().numpy()
    sil = silhouette_score(latent_np, test_labels)
    print(f"Silhouette score: {sil:.4f}")

    # PCA (2D)
    try:
        pca = PCA(n_components=2)
        proj = pca.fit_transform(latent_np)
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(types)))
        fig, ax = plt.subplots(figsize=(6, 5))
        for tidx, tname in enumerate(types):
            mask = (test_labels == tidx)
            if mask.sum() > 0:
                ax.scatter(proj[mask, 0], proj[mask, 1],
                         label=tname, color=colors[tidx], alpha=0.7, s=50)
        ax.set_xlabel('PC1', fontsize=12)
        ax.set_ylabel('PC2', fontsize=12)
        ax.set_title('PCA (final latent)', fontsize=14)
        ax.legend(markerscale=1.5, fontsize='small', ncol=2)
        ax.grid(True, alpha=0.3)
        
        os.makedirs(SAVE_DIR, exist_ok=True)
        fname = os.path.join(SAVE_DIR, f"PCA.png")
        fig.savefig(fname, bbox_inches='tight', dpi=150)
        plt.close(fig)
    except Exception as e:
        print('PCA failed:', e)

    try:
        sim = 1 - squareform(pdist(latent_np, metric='cosine'))
        order = np.argsort(test_labels)
        sim_ord = sim[np.ix_(order, order)]
        labels_ord = np.array(test_labels)[order]

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(sim_ord, cmap='viridis', vmin=-1, vmax=1, aspect='equal')
        ax.set_title('Latent Similarity', fontsize=14)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        unique_labels, counts = np.unique(labels_ord, return_counts=True)
        cum = np.cumsum(counts)
        for c in cum[:-1]:
            ax.axhline(c - 0.5, color='k', linewidth=1.5)
            ax.axvline(c - 0.5, color='k', linewidth=1.5)

        tick_pos = np.cumsum(counts) - counts / 2.0
        ax.set_xticks(tick_pos)
        ax.set_yticks(tick_pos)
        ax.set_xticklabels(types, rotation=90, fontsize=10)
        ax.set_yticklabels(types, fontsize=10)

        os.makedirs(SAVE_DIR, exist_ok=True)
        fname = os.path.join(SAVE_DIR, f"latent_sim.png")
        fig.savefig(fname, bbox_inches='tight', dpi=150)
        plt.close(fig)

    except Exception as e:
        print('Similarity plotting failed:', e)


def plot_results(history, model, X_test, test_labels_np, types, save_dir=SAVE_DIR):

    def smooth(y, window):
        y = np.array(y)
        if len(y) < window:
            return y
        # use valid mode to avoid edge artifacts, then pad to original length
        w = np.ones(window) / window
        y_smooth = np.convolve(y, w, mode='valid')
        # pad edges with original values
        pad_left = (len(y) - len(y_smooth)) // 2
        pad_right = len(y) - len(y_smooth) - pad_left
        return np.concatenate([y[:pad_left], y_smooth, y[-pad_right:]])

    # Loss + accuracy
    epochs_range = range(1, len(history['train_loss']) + 1)
    fig, ax1 = plt.subplots(figsize=(3.5, 3), constrained_layout=True)
    # train=blue, test=green; loss=solid, acc=dashed
    ax1.plot(epochs_range, smooth(history['train_loss'], 10), label='Train loss', color='b', linewidth=1.5, linestyle='-')
    ax1.plot(epochs_range, smooth(history['test_loss'], 10), label='Test loss', color='g', linewidth=1.5, linestyle='-')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(epochs_range, smooth(history['train_acc'], 10), label='Train acc', color='b', linewidth=1.5, linestyle='--')
    ax2.plot(epochs_range, smooth(history['test_acc'], 10), label='Test acc', color='g', linewidth=1.5, linestyle='--')
    ax2.set_ylabel('acc')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False)

    os.makedirs(save_dir, exist_ok=True)
    fname = os.path.join(save_dir, f"loss_acc.png")
    fig.savefig(fname, bbox_inches='tight', dpi=150)
    plt.close(fig)

    # Hidden/latent per-timestep similarity
    with torch.no_grad():
        hidden_test, z_test, test_output = model(X_test)

    if hidden_test is not None:
        hidden_np = hidden_test.cpu().numpy()
        order = np.argsort(test_labels_np)
        L_ts = hidden_np.shape[0]
        ncols = min(4, L_ts)
        nrows = int(np.ceil(L_ts / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
        axes = np.array(axes).reshape(-1)
        for t in range(L_ts):
            ax = axes[t]
            sim = 1 - squareform(pdist(hidden_np[t], metric='cosine'))
            sim_ord = sim[np.ix_(order, order)]
            im = ax.imshow(sim_ord, cmap='viridis', vmin=-1, vmax=1, aspect='equal')
            ax.set_title(f'Timestep {t+1}')
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in axes[L_ts:]:
            ax.axis('off')
        fig.colorbar(im, ax=axes.tolist(), fraction=0.02)
        fname = os.path.join(save_dir, f"latent_by_timestep.png")
        fig.savefig(fname, bbox_inches='tight', dpi=150)
        plt.close(fig)

        # PCA branching structure - show how types diverge across timesteps
        try:
            # Get encoder and decoder hidden states
            with torch.no_grad():
                # Encoder hidden states (already have from earlier)
                encoder_hidden = hidden_test.cpu().numpy()  # (L, B, N)
                
                # Decoder hidden states - need to get from decoder RNN
                if model.enable_serialize:
                    latent_s = model.serialize(z_test)
                    B = latent_s.shape[0]
                    latent_seq = latent_s.view(B, model.sequence_length, model.d_latent).permute(1,0,2).contiguous()
                    decoder_hidden, _ = model.decoder.rnn(latent_seq)
                else:
                    latent_expanded = z_test.unsqueeze(0).expand(model.sequence_length, *[-1] * z_test.dim())
                    decoder_hidden, _ = model.decoder.rnn(latent_expanded)
                
                decoder_hidden = decoder_hidden.cpu().numpy()  # (L, B, N)
            
            L_ts = encoder_hidden.shape[0]
            
            # Plot encoder PCA - branching structure
            # Collect all prefix-type combinations at each timestep
            all_hiddens = []
            all_labels = []
            all_timesteps = []
            
            for t in range(L_ts):
                # At timestep t, we have read t+1 characters
                prefix_length = t + 1
                
                # Group types by their prefix of length prefix_length
                prefix_groups = {}
                for tidx, tname in enumerate(types):
                    mask = (test_labels_np == tidx)
                    if mask.sum() > 0:
                        prefix = tname[:prefix_length]
                        if prefix not in prefix_groups:
                            prefix_groups[prefix] = []
                        prefix_groups[prefix].append((tidx, mask))
                
                # For each unique prefix, average hidden states
                for prefix, type_list in prefix_groups.items():
                    # Combine all samples with this prefix
                    all_masks = [mask for _, mask in type_list]
                    combined_mask = np.logical_or.reduce(all_masks)
                    
                    # Average over all samples with this prefix
                    prefix_hidden = encoder_hidden[t, combined_mask, :].mean(axis=0)
                    all_hiddens.append(prefix_hidden)
                    all_labels.append(prefix)
                    all_timesteps.append(t)
            
            # Do PCA on all prefix-timestep combinations
            all_hiddens = np.array(all_hiddens)
            pca = PCA(n_components=2)
            proj = pca.fit_transform(all_hiddens)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot points colored by timestep
            colors = plt.cm.viridis(np.linspace(0, 1, L_ts))
            for i, (label, t) in enumerate(zip(all_labels, all_timesteps)):
                ax.scatter(proj[i, 0], proj[i, 1], c=[colors[t]], s=100, alpha=0.6, edgecolors='black', linewidths=1)
                ax.text(proj[i, 0], proj[i, 1], label, 
                       fontsize=8, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
            
            # Add colorbar for timestep
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=1, vmax=L_ts))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Timestep', fontsize=12)
            
            ax.set_xlabel('PC1', fontsize=12)
            ax.set_ylabel('PC2', fontsize=12)
            ax.set_title('Encoder: Branching structure across timesteps', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            fname = os.path.join(save_dir, f"PCA_encoder_branching.png")
            fig.savefig(fname, bbox_inches='tight', dpi=150)
            plt.close(fig)
            
            # Plot decoder PCA - branching structure
            all_hiddens = []
            all_labels = []
            all_timesteps = []
            
            for t in range(L_ts):
                prefix_length = t + 1
                prefix_groups = {}
                for tidx, tname in enumerate(types):
                    mask = (test_labels_np == tidx)
                    if mask.sum() > 0:
                        prefix = tname[:prefix_length]
                        if prefix not in prefix_groups:
                            prefix_groups[prefix] = []
                        prefix_groups[prefix].append((tidx, mask))
                
                for prefix, type_list in prefix_groups.items():
                    all_masks = [mask for _, mask in type_list]
                    combined_mask = np.logical_or.reduce(all_masks)
                    prefix_hidden = decoder_hidden[t, combined_mask, :].mean(axis=0)
                    all_hiddens.append(prefix_hidden)
                    all_labels.append(prefix)
                    all_timesteps.append(t)
            
            all_hiddens = np.array(all_hiddens)
            pca = PCA(n_components=2)
            proj = pca.fit_transform(all_hiddens)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            for i, (label, t) in enumerate(zip(all_labels, all_timesteps)):
                ax.scatter(proj[i, 0], proj[i, 1], c=[colors[t]], s=100, alpha=0.6, edgecolors='black', linewidths=1)
                ax.text(proj[i, 0], proj[i, 1], label, 
                       fontsize=8, ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
            
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=1, vmax=L_ts))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Timestep', fontsize=12)
            
            ax.set_xlabel('PC1', fontsize=12)
            ax.set_ylabel('PC2', fontsize=12)
            ax.set_title('Decoder: Branching structure across timesteps', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            fname = os.path.join(save_dir, f"PCA_decoder_branching.png")
            fig.savefig(fname, bbox_inches='tight', dpi=150)
            plt.close(fig)
            
        except Exception as e:
            print(f'PCA branching plotting failed: {e}')


def run_experiment():
    set_seed(seed)

    model = RNNAutoencoder(
        alpha, d_hidden, num_layers, d_latent, L,
        enable_serialize=ENABLE_SERIALIZE_Z,
    ).to(device)

    seq_train, seq_test, labels_train, labels_test, types = generate_instances(
        alpha, L, m, frac_train=0.8
    )
    X_train = sequences_to_tensor(seq_train, alpha).to(device)
    X_test = sequences_to_tensor(seq_test, alpha).to(device)
    test_labels = torch.tensor(labels_test, dtype=torch.long)

    history = train(
        model, X_train, X_test, test_labels,
        n_epochs=epochs, lr=lr, weight_decay=weight_decay,
        use_contrastive=ENABLE_CONTRASTIVE,
        contrastive_weight=CONTRASTIVE_WEIGHT
    )


    model.eval()
    with torch.no_grad():
        _, z_test, test_output = model(X_test)
        pred_test = torch.argmax(test_output, dim=-1)
        target_test = torch.argmax(X_test, dim=-1)
        test_acc = (pred_test == target_test).all(dim=0).float().mean().item()

    train_acc = history['train_acc'][-1]
    train_metrics = {'acc': train_acc}
    test_metrics = {'acc': test_acc}

    test_labels_np = test_labels.cpu().numpy()
    compute_metrics(train_metrics, test_metrics, z_test, test_labels_np, types)
    
    # plotting: moved to helper to keep run_experiment concise
    plot_results(history, model, X_test, test_labels_np, types, save_dir=SAVE_DIR)



if __name__ == '__main__':
    run_experiment()