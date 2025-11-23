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
ENABLE_AUGMENT = True
ENABLE_SERIALIZE_Z = True
ENABLE_CONTRASTIVE = True      
CONTRASTIVE_WEIGHT = 0.5      
# =====================================

seed = 42
L, m, alpha = 4, 2, 6
epochs = 250  
lr = 1e-3
d_hidden = 128
d_latent = 32  
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
    types_nested = findStructures(alphabet, L, m)
    types = [t for sub in types_nested for t in sub]

    all_perms = list(itertools.permutations(alphabet, m))
    seqs = []
    labels = []
    for type_idx, t in enumerate(types):
        for perm in all_perms:
            seqs.append(seq_replace_symbols(t, perm))
            labels.append(type_idx)

    seqs = np.array(seqs)
    labels = np.array(labels)
    N = len(seqs)
    idx = np.random.permutation(N)
    split = int(N * frac_train)
    return seqs[idx[:split]], seqs[idx[split:]], labels[idx[:split]], labels[idx[split:]], types


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


def augment_sequences(X_onehot, alpha, prob=0.5):
    L, B, _ = X_onehot.shape
    X_aug = X_onehot.clone()
    for b in range(B):
        if np.random.rand() < prob:
            seq_indices = torch.argmax(X_onehot[:, b, :], dim=-1)
            unique_letters = torch.unique(seq_indices).tolist()
            n_unique = len(unique_letters)
            new_letters = np.random.choice(alpha, n_unique, replace=False)
            mapping = {old: new for old, new in zip(unique_letters, new_letters)}
            new_indices = torch.tensor([mapping[idx.item()] for idx in seq_indices])
            new_onehot = torch.zeros(L, alpha)
            new_onehot[torch.arange(L), new_indices] = 1
            X_aug[:, b, :] = new_onehot
    return X_aug


def contrastive_loss(latent, labels, temperature=0.5):
    """   
    Args:
        latent: (B, d_latent) 
        labels: (B,) type labels
        temperature
    """
    B = latent.shape[0]
    # L2 normalize
    latent_norm = F.normalize(latent, p=2, dim=1)

    similarity = torch.matmul(latent_norm, latent_norm.T) / temperature  # (B, B)
    
    # positive mask 
    labels_expanded = labels.unsqueeze(0)  # (1, B)
    labels_T = labels.unsqueeze(1)  # (B, 1)
    positive_mask = (labels_expanded == labels_T).float()  # (B, B)
    
    # Remove diagonal 
    positive_mask = positive_mask * (1 - torch.eye(B, device=latent.device))
    
    # Calculate loss (InfoNCE)
    exp_sim = torch.exp(similarity)
    
    # sum of positive samples / sum of all samples
    positive_sum = (exp_sim * positive_mask).sum(dim=1)
    all_sum = exp_sim.sum(dim=1) - torch.diag(exp_sim) 
    
    loss = -torch.log(positive_sum / (all_sum + 1e-8) + 1e-8)
    
    # Only compute loss for samples with positive examples
    valid_mask = positive_mask.sum(dim=1) > 0
    if valid_mask.sum() > 0:
        loss = loss[valid_mask].mean()
    else:
        loss = torch.tensor(0.0, device=latent.device)
    
    return loss


def train_autoencoder_improved(model, X_train, X_test, train_labels, test_labels,
                               n_epochs=300, lr=0.001, weight_decay=0,
                               use_augment=True, use_contrastive=True,
                               contrastive_weight=0.5):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {k: [] for k in ['train_loss', 'ce_loss', 'contrastive_loss', 'test_loss',
                               'train_token_acc', 'test_token_acc',
                               'train_seq_acc', 'test_seq_acc', 'silhouette']}

    alpha = X_train.shape[2]
 
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        if use_augment:
            X_batch = augment_sequences(X_train, alpha, prob=0.5)
        else:
            X_batch = X_train

        hidden, latent, output = model(X_batch)

        ce_loss = F.cross_entropy(
            output.reshape(-1, output.shape[-1]),
            torch.argmax(X_batch, dim=-1).reshape(-1)
        )
        total_loss = ce_loss

        contrast_loss = torch.tensor(0.0)
        if use_contrastive:
            contrast_loss = contrastive_loss(latent, train_labels, temperature=0.5)
            total_loss = total_loss + contrastive_weight * contrast_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # evaluation
        with torch.no_grad():
            pred_train = torch.argmax(output, dim=-1)
            target_train = torch.argmax(X_batch, dim=-1)
            train_token_acc = (pred_train == target_train).float().mean().item()
            train_seq_acc = (pred_train == target_train).all(dim=0).float().mean().item()

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
            test_token_acc = (pred_test == target_test).float().mean().item()
            test_seq_acc = (pred_test == target_test).all(dim=0).float().mean().item()            

            sil = silhouette_score(test_latent.cpu().numpy(), test_labels.cpu().numpy())

        # history
        history['train_loss'].append(total_loss.item())
        history['ce_loss'].append(ce_loss.item())
        history['contrastive_loss'].append(contrast_loss.item() if isinstance(contrast_loss, torch.Tensor) else 0)
        history['test_loss'].append(float(test_loss))
        history['train_token_acc'].append(train_token_acc)
        history['test_token_acc'].append(test_token_acc)
        history['train_seq_acc'].append(train_seq_acc)
        history['test_seq_acc'].append(test_seq_acc)
        history['silhouette'].append(sil)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Loss={total_loss.item():.4f} "
                  f"(CE={ce_loss.item():.4f} Contr={contrast_loss.item():.4f} "
                  f"Token={train_token_acc:.3f} Seq={train_seq_acc:.3f} | "
                  f"Test: Loss={float(test_loss):.4f} Token={test_token_acc:.3f} "
                  f"Seq={test_seq_acc:.3f} Sil={sil:.3f}")

    return history


def compute_metrics_and_print(name, train_metrics, test_metrics, latent_test, test_labels, types):
    print('\n' + '='*60)
    print(f'Experiment: {name}')
    print('='*60)
    print(f"Train token acc: {train_metrics['token']:.4f}, Train seq acc: {train_metrics['seq']:.4f}")
    print(f"Test token acc: {test_metrics['token']:.4f}, Test seq acc: {test_metrics['seq']:.4f}")
    
    # silhouette
    latent_np = latent_test.cpu().numpy()
    sil = silhouette_score(latent_np, test_labels)
    print(f"Silhouette score (test latent): {sil:.4f}")

    # PCA
    try:
        pca = PCA(n_components=min(3, latent_np.shape[1]))
        proj = pca.fit_transform(latent_np)
        print(f"PCA explained variance ratio (first {proj.shape[1]}): {pca.explained_variance_ratio_}")
        
        for tidx, tname in enumerate(types):
            mask = (test_labels == tidx)
            if mask.sum() > 0:
                mean_coord = proj[mask].mean(axis=0)
                print(f" Type {tname} mean proj (first 2): {mean_coord[:2]}")

        # 3D PCA 
        colors = plt.cm.tab20(np.linspace(0, 1, len(types)))
        if proj.shape[1] >= 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            for tidx, tname in enumerate(types):
                mask = (test_labels == tidx)
                if mask.sum() > 0:
                    ax.scatter(proj[mask, 0], proj[mask, 1], proj[mask, 2],
                             label=tname, color=colors[tidx], alpha=0.7, s=50)
            ax.set_xlabel('PC1', fontsize=12)
            ax.set_ylabel('PC2', fontsize=12)
            ax.set_zlabel('PC3', fontsize=12)
            ax.set_title('PCA 3D Visualization', fontsize=14)
            ax.legend(markerscale=1.5, fontsize='medium')
            
            os.makedirs(SAVE_DIR, exist_ok=True)
            fname = os.path.join(SAVE_DIR, f"{name}_pca3d_seed{seed}.png")
            fig.savefig(fname, bbox_inches='tight', dpi=150)
            plt.close(fig)
    except Exception as e:
        print('PCA failed:', e)

    try:
        sim = 1 - squareform(pdist(latent_np, metric='cosine'))
        order = np.argsort(test_labels)
        sim_ord = sim[np.ix_(order, order)]
        labels_ord = np.array(test_labels)[order]

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(sim_ord, cmap='coolwarm', vmin=-1, vmax=1, aspect='equal')
        ax.set_title('Latent Similarity Matrix (by type)', fontsize=14)
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
        fname = os.path.join(SAVE_DIR, f"{name}_sim_by_label_seed{seed}.png")
        fig.savefig(fname, bbox_inches='tight', dpi=150)
        plt.close(fig)

        print("\nIntra-type similarity:")
        for tidx, tname in enumerate(types):
            mask = (np.array(test_labels) == tidx)
            if mask.sum() > 1:
                type_lat = latent_np[mask]
                sims = 1 - pdist(type_lat, metric='cosine')
                print(f"Type {tname}: mean={sims.mean():.4f}, std={sims.std():.4f}")
    except Exception as e:
        print('Similarity plotting failed:', e)


def run_experiment():
    set_seed(seed)

    model = RNNAutoencoder(
        alpha, d_hidden, num_layers, d_latent, L,
        enable_serialize=ENABLE_SERIALIZE_Z,
        serialize_type='linear'
    ).to(device)

    seq_train, seq_test, labels_train, labels_test, types = generate_instances(
        alpha, L, m, frac_train=0.8
    )
    X_train = sequences_to_tensor(seq_train, alpha).to(device)
    X_test = sequences_to_tensor(seq_test, alpha).to(device)
    train_labels = torch.tensor(labels_train, dtype=torch.long)
    test_labels = torch.tensor(labels_test, dtype=torch.long)

    history = train_autoencoder_improved(
        model, X_train, X_test, train_labels, test_labels,
        n_epochs=epochs, lr=lr, weight_decay=weight_decay,
        use_augment=ENABLE_AUGMENT,
        use_contrastive=ENABLE_CONTRASTIVE,
        contrastive_weight=CONTRASTIVE_WEIGHT
    )


    model.eval()
    with torch.no_grad():
        _, z_test, test_output = model(X_test)
        pred_test = torch.argmax(test_output, dim=-1)
        target_test = torch.argmax(X_test, dim=-1)
        test_token_acc = (pred_test == target_test).float().mean().item()
        test_seq_acc = (pred_test == target_test).all(dim=0).float().mean().item()

    train_token_acc = history['train_token_acc'][-1]
    train_seq_acc = history['train_seq_acc'][-1]
    train_metrics = {'token': train_token_acc, 'seq': train_seq_acc}
    test_metrics = {'token': test_token_acc, 'seq': test_seq_acc}

    test_labels_np = test_labels.cpu().numpy()
    compute_metrics_and_print('improved_ablation', train_metrics, test_metrics,
                            z_test, test_labels_np, types)
    
    # Silhouette score over epochs
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history['silhouette'], linewidth=2, color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score Over Epochs')
    ax.grid(True, alpha=0.3)
    # reference line at 0
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    os.makedirs(SAVE_DIR, exist_ok=True)
    fname = os.path.join(SAVE_DIR, f"silhouette_seed{seed}.png")
    fig.savefig(fname, bbox_inches='tight', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    run_experiment()