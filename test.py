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
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import os


# =====================================
ENABLE_SERIALIZE_Z = False
ENABLE_CONTRASTIVE = False      
CONTRASTIVE_WEIGHT = 0   
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


def plot_diagnostics(model, X_train, train_labels, X_test, test_labels, types, save_dir=SAVE_DIR, history=None):

    os.makedirs(save_dir, exist_ok=True)

    def _safe_numpy(t):
        return t.cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t)

    try:
        model.eval()
        with torch.no_grad():
            h_tr, z_tr, _ = model(X_train)
            h_te, z_te, _ = model(X_test)

        datasets = [
            (h_tr, z_tr, train_labels, 'train'),
            (h_te, z_te, test_labels, 'test')
        ]

        for hidden, z, labels, phase in datasets:
            labels_np = _safe_numpy(labels)
            hidden_np = hidden.cpu().numpy() if hidden is not None else None
            z_np = z.cpu().numpy()

            # Encoder PCA spectrum (using final timestep hidden state)
            if hidden_np is not None:
                try:
                    enc_final = hidden_np[-1]
                    n_comp = min(enc_final.shape[1], enc_final.shape[0])
                    pca_enc = PCA(n_components=n_comp)
                    pca_enc.fit(enc_final)
                    evr_enc = pca_enc.explained_variance_ratio_
                    
                    # Show only first 10 components for better visualization
                    evr_plot = evr_enc[:min(10, len(evr_enc))]

                    fig, ax = plt.subplots(figsize=(4, 2.5))
                    ax.bar(range(1, len(evr_plot)+1), evr_plot, color='C0')
                    ax.set_xlabel('PC')
                    ax.set_ylabel('Explained variance ratio')
                    fname = os.path.join(save_dir, f"encoder_pca_spectrum_{phase}.svg")
                    fig.savefig(fname, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    print(f"Encoder PCA spectrum failed ({phase}): {e}")
            
            # Latent PCA spectrum
            try:
                n_comp = min(z_np.shape[1], z_np.shape[0])
                pca_lat = PCA(n_components=n_comp)
                pca_lat.fit(z_np)
                evr_lat = pca_lat.explained_variance_ratio_

                fig, ax = plt.subplots(figsize=(4, 2.5))
                ax.bar(range(1, len(evr_lat)+1), evr_lat, color='C1')
                ax.set_xlabel('PC')
                ax.set_ylabel('Explained variance ratio')
                fname = os.path.join(save_dir, f"latent_pca_spectrum_{phase}.svg")
                fig.savefig(fname, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"Latent PCA spectrum failed ({phase}): {e}")

            # latent similarity heatmap
            try:
                sim = 1 - squareform(pdist(z_np, metric='cosine'))
                order = np.argsort(labels_np)
                sim_ord = sim[np.ix_(order, order)]
                fig, ax = plt.subplots(figsize=(5, 5))
                im = ax.imshow(sim_ord, cmap='viridis', vmin=-1, vmax=1, aspect='equal')
                ax.set_title('Latent Similarity', fontsize=12)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                unique_labels, counts = np.unique(labels_np[order], return_counts=True)
                cum = np.cumsum(counts)
                for c in cum[:-1]:
                    ax.axhline(c - 0.5, color='k', linewidth=1.2)
                    ax.axvline(c - 0.5, color='k', linewidth=1.2)
                tick_pos = np.cumsum(counts) - counts / 2.0
                ax.set_xticks(tick_pos)
                ax.set_yticks(tick_pos)
                ax.set_xticklabels(types, rotation=90, fontsize=9)
                ax.set_yticklabels(types, fontsize=9)
                fname = os.path.join(save_dir, f"latent_sim_{phase}.svg")
                fig.savefig(fname, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"Latent similarity plot failed ({phase}): {e}")

            # per-timestep similarity for encoder hidden states (grid layout with type boundaries)
            if hidden_np is not None:
                try:
                    L_ts = hidden_np.shape[0]
                    order = np.argsort(labels_np)
                    ncols = min(4, L_ts)
                    nrows = int(np.ceil(L_ts / ncols))
                    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
                    axes = np.array(axes).reshape(-1)
                    
                    for t in range(L_ts):
                        ax = axes[t]
                        sim = 1 - squareform(pdist(hidden_np[t], metric='cosine'))
                        sim_ord = sim[np.ix_(order, order)]
                        im = ax.imshow(sim_ord, cmap='viridis', vmin=0, vmax=1, aspect='equal')
                        ax.set_title(f'Timestep {t+1}')
                        ax.set_xticks([])
                        ax.set_yticks([])
                        
                        # draw type boundary separators
                        labels_ordered = labels_np[order]
                        uniq, counts = np.unique(labels_ordered, return_counts=True)
                        cumsum = np.cumsum(counts)
                        for bound in cumsum[:-1]:
                            pos = bound - 0.5
                            ax.axhline(pos, color='white', linewidth=1.2, alpha=0.9)
                            ax.axvline(pos, color='white', linewidth=1.2, alpha=0.9)
                    
                    for ax in axes[L_ts:]:
                        ax.axis('off')
                    fig.colorbar(im, ax=axes.tolist(), fraction=0.02)
                    fname = os.path.join(save_dir, f"encoder_by_timestep_{phase}.svg")
                    fig.savefig(fname, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    print(f"Encoder hidden timestep similarity failed ({phase}): {e}")
            
            # per-timestep similarity for serialized latent (if enabled)
            if hasattr(model, 'decoder') and getattr(model.decoder, 'enable_serialize', False):
                try:
                    with torch.no_grad():
                        latent_s = model.decoder.serialize(z)
                    B = latent_s.shape[0]
                    # reshape to (L, B, d_latent)
                    latent_seq = latent_s.view(B, model.sequence_length, model.d_latent).permute(1, 0, 2).cpu().numpy()
                    L_ts = latent_seq.shape[0]
                    order = np.argsort(labels_np)
                    ncols = min(4, L_ts)
                    nrows = int(np.ceil(L_ts / ncols))
                    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
                    axes = np.array(axes).reshape(-1)
                    
                    for t in range(L_ts):
                        ax = axes[t]
                        sim = 1 - squareform(pdist(latent_seq[t], metric='cosine'))
                        sim_ord = sim[np.ix_(order, order)]
                        im = ax.imshow(sim_ord, cmap='viridis', vmin=0, vmax=1, aspect='equal')
                        ax.set_title(f'Timestep {t+1}')
                        ax.set_xticks([])
                        ax.set_yticks([])
                        
                        # draw type boundary separators
                        labels_ordered = labels_np[order]
                        uniq, counts = np.unique(labels_ordered, return_counts=True)
                        cumsum = np.cumsum(counts)
                        for bound in cumsum[:-1]:
                            pos = bound - 0.5
                            ax.axhline(pos, color='white', linewidth=1.2, alpha=0.9)
                            ax.axvline(pos, color='white', linewidth=1.2, alpha=0.9)
                    
                    for ax in axes[L_ts:]:
                        ax.axis('off')
                    fig.colorbar(im, ax=axes.tolist(), fraction=0.02)
                    fname = os.path.join(save_dir, f"latent_by_timestep_{phase}.svg")
                    fig.savefig(fname, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    print(f"Serialized latent timestep similarity failed ({phase}): {e}")

            # Encoder PCA (separate plot)
            try:
                if hidden_np is not None:
                    enc_final = hidden_np[-1]
                    pca_enc = PCA(n_components=2)
                    proj_enc = pca_enc.fit_transform(enc_final)
                    colors = plt.cm.tab20(np.linspace(0, 1, len(types)))
                    
                    fig, ax = plt.subplots(figsize=(4, 4))
                    for tidx in range(len(types)):
                        mask = (labels_np == tidx)
                        if mask.sum() > 0:
                            ax.scatter(proj_enc[mask, 0], proj_enc[mask, 1], label=types[tidx], color=colors[tidx], s=30, alpha=0.8)
                    ax.set_xlabel('PC1')
                    ax.set_ylabel('PC2')
                    # legend in 2 rows (ncol = half of types, rounded up)
                    ncol_legend = max(1, (len(types) + 1) // 2)
                    ax.legend(fontsize='small', ncol=ncol_legend, loc='upper center', bbox_to_anchor=(0.5, 1.17), frameon=False)
                    ax.grid(True, alpha=0.2)
                    
                    fname = os.path.join(save_dir, f"encoder_pca_{phase}.svg")
                    fig.savefig(fname, bbox_inches='tight')
                    plt.close(fig)
            except Exception as e:
                print(f"Encoder PCA failed ({phase}): {e}")
            
            # Latent PCA (separate plot, using encoder output z)
            try:
                pca_lat = PCA(n_components=2)
                proj_lat = pca_lat.fit_transform(z_np)
                colors = plt.cm.tab20(np.linspace(0, 1, len(types)))
                
                fig, ax = plt.subplots(figsize=(4, 4))
                for tidx in range(len(types)):
                    mask = (labels_np == tidx)
                    if mask.sum() > 0:
                        ax.scatter(proj_lat[mask, 0], proj_lat[mask, 1], label=types[tidx], color=colors[tidx], s=30, alpha=0.8)
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                # legend in 2 rows (ncol = half of types, rounded up)
                ncol_legend = max(1, (len(types) + 1) // 2)
                ax.legend(fontsize='small', ncol=ncol_legend, loc='upper center', bbox_to_anchor=(0.5, 1.17), frameon=False)
                ax.grid(True, alpha=0.2)
                
                fname = os.path.join(save_dir, f"latent_pca_{phase}.svg")
                fig.savefig(fname, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"Latent PCA failed ({phase}): {e}")
            
            # Branching PCA (prefix aggregation for encoder
            # if phase == 'test' and hidden_np is not None:
            #     L_ts = hidden_np.shape[0]
            #     batch_labels_str = [types[idx] for idx in labels_np]
                
                # def plot_branching_pca(hidden_states, title_prefix, filename_suffix):
                #     all_hiddens = []
                #     all_labels = []
                #     all_timesteps = []
                #     for t in range(L_ts):
                #         prefix_length = t + 1
                #         groups = collections.defaultdict(list)
                #         for i, full_str in enumerate(batch_labels_str):
                #             if len(full_str) >= prefix_length:
                #                 p = full_str[:prefix_length]
                #                 groups[p].append(i)
                        
                #         for prefix, indices in groups.items():
                #             mean_vec = hidden_states[t, indices, :].mean(axis=0)
                #             all_hiddens.append(mean_vec)
                #             all_labels.append(prefix)
                #             all_timesteps.append(t)
                    
                #     if len(all_hiddens) == 0:
                #         print(f"Skipping {title_prefix} PCA: No data collected.")
                #         return
                    
                #     all_hiddens_np = np.array(all_hiddens)
                #     pca = PCA(n_components=2)
                #     proj = pca.fit_transform(all_hiddens_np)
                    
                #     fig, ax = plt.subplots(figsize=(4, 3.5))
                #     ax.scatter(proj[:, 0], proj[:, 1], alpha=0)
                #     colors = plt.cm.plasma(np.linspace(0, 0.85, L_ts))
                    
                #     for i, (label, t) in enumerate(zip(all_labels, all_timesteps)):
                #         x, y = proj[i, 0], proj[i, 1]
                #         c = colors[t]
                #         ax.text(x, y, label, color=c, fontsize=12, ha='right', va='bottom', alpha=0.8)
                    
                #     ax.set_xlabel('PC1')
                #     ax.set_ylabel('PC2')
                #     ax.set_title(f'$\\lambda ={CONTRASTIVE_WEIGHT}$', fontsize=14)
                #     ax.grid(True, alpha=0.2)
                    
                #     fname = os.path.join(save_dir, f"PCA_{filename_suffix}_branching.svg")
                #     fig.savefig(fname, bbox_inches='tight')
                #     plt.close(fig)
                
                # # Encoder branching PCA
                # plot_branching_pca(hidden_np, "Encoder", "encoder")
                        
    except Exception as e:
        print(f"plot_all_diagnostics failed: {e}")
    
    # smoothed loss/acc curves
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


def train(model, X_train, X_test, test_labels, train_labels=None,
    n_epochs=300, lr=0.001, weight_decay=1e-3,
    use_contrastive=True, contrastive_weight=0.5, types=None, save_dir=SAVE_DIR):
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

        # no per-epoch diagnostics; plotting done once after training

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
    
    # silhouette only; plotting moved to `plot_all_diagnostics` (final-epoch single function)
    latent_np = latent_test.cpu().numpy()
    sil = silhouette_score(latent_np, test_labels)
    print(f"Silhouette score: {sil:.4f}")
    return {'silhouette': float(sil)}


def run_experiment():
    set_seed(seed)

    model = RNNAutoencoder(alpha, d_hidden, num_layers, d_latent, L,
        enable_serialize=ENABLE_SERIALIZE_Z,
    ).to(device)

    seq_train, seq_test, labels_train, labels_test, types = generate_instances(
        alpha, L, m, frac_train=0.8
    )
    X_train = sequences_to_tensor(seq_train, alpha).to(device)
    X_test = sequences_to_tensor(seq_test, alpha).to(device)
    test_labels = torch.tensor(labels_test, dtype=torch.long)

    history = train(
        model, X_train, X_test, test_labels, train_labels=torch.tensor(labels_train, dtype=torch.long),
        n_epochs=epochs, lr=lr, weight_decay=weight_decay,
        use_contrastive=ENABLE_CONTRASTIVE,
        contrastive_weight=CONTRASTIVE_WEIGHT, types=types, save_dir=SAVE_DIR
    )

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
    try:
        compute_metrics(train_metrics, test_metrics, z_test, labels_test, types)
    except Exception as e:
        print(f"compute_metrics failed: {e}")

    # save consolidated final-epoch diagnostics (SVGs)
    try:
        plot_all_diagnostics(model, X_train, torch.tensor(labels_train, dtype=torch.long), X_test, test_labels, types, save_dir=SAVE_DIR, history=history)
    except Exception as e:
        print(f"plot_all_diagnostics failed: {e}")


if __name__ == '__main__':
    run_experiment()
