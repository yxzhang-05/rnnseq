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


ENABLE_AUGMENT = True          
ENABLE_SERIALIZE_Z = True       

# ----------------------
# 实验超参与随机性控制
# ----------------------
seed = 42
L, m, alpha = 4, 2, 6
epochs = 250
lr = 1e-3
d_hidden = 32
d_latent = 8
weight_decay = 0.0
num_layers = 1
device = torch.device('cpu')
SAVE_DIR = "results"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_instances(alpha, L, m, frac_train=0.8):
    """
    Generate train/test instances using the canonical templates from `sequences.findStructures`.
    If L_param or m_param are None, fall back to globals `L` and `m`.
    """

    alphabet = list(string.ascii_lowercase[:alpha])

    # findStructures returns a nested list; flatten to get the types for the requested m
    types_nested = findStructures(alphabet, L, m)
    # flatten
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
    # also return the canonical type/templates (one item per type index)
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
    return one_hot.permute(1, 0, 2)  # (L, B, alpha)


def augment_sequences(X_onehot, alpha, prob=0.3):
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


# ---------- Inline training utilities (from eval_autoencoder_generalization) ----------
def train_autoencoder_improved(model, X_train, X_test, train_labels, test_labels,
                               n_epochs=300, lr=0.001, weight_decay=0,
                               use_augment=True):
    """A compact training loop with optional augmentation and contrastive loss.

    This is a simplified, self-contained version adapted for the ablation script.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=False)
    history = {k: [] for k in ['train_loss','ce_loss','test_loss','train_token_acc','test_token_acc','train_seq_acc','test_seq_acc']}

    alpha = X_train.shape[2]

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()

        if use_augment:
            X_batch = augment_sequences(X_train, alpha, prob=0.5)
        else:
            X_batch = X_train

        hidden, latent, output = model(X_batch)

        ce_loss = F.cross_entropy(output.reshape(-1, output.shape[-1]), torch.argmax(X_batch, dim=-1).reshape(-1))
        total_loss = ce_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            pred_train = torch.argmax(output, dim=-1)
            target_train = torch.argmax(X_batch, dim=-1)
            train_token_acc = (pred_train == target_train).float().mean().item()
            train_seq_acc = (pred_train == target_train).all(dim=0).float().mean().item()

        model.eval()
        with torch.no_grad():
            _, test_latent, test_output = model(X_test)
            test_ce_loss = F.cross_entropy(test_output.reshape(-1, test_output.shape[-1]), torch.argmax(X_test, dim=-1).reshape(-1))
            test_loss = test_ce_loss

            pred_test = torch.argmax(test_output, dim=-1)
            target_test = torch.argmax(X_test, dim=-1)
            test_token_acc = (pred_test == target_test).float().mean().item()
            test_seq_acc = (pred_test == target_test).all(dim=0).float().mean().item()

        scheduler.step(test_loss)

        history['train_loss'].append(total_loss.item())
        history['ce_loss'].append(ce_loss.item())
        history['test_loss'].append(float(test_loss if isinstance(test_loss, torch.Tensor) else test_loss))
        history['train_token_acc'].append(train_token_acc)
        history['test_token_acc'].append(test_token_acc)
        history['train_seq_acc'].append(train_seq_acc)
        history['test_seq_acc'].append(test_seq_acc)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Train Loss={total_loss.item():.4f} (CE={ce_loss.item():.4f}) Token={train_token_acc:.3f} Seq={train_seq_acc:.3f} | Test Loss={float(test_loss):.4f} Token={test_token_acc:.3f} Seq={test_seq_acc:.3f}")

    return history

# ---------- end inlined training utilities ----------

def compute_metrics_and_print(name, train_metrics, test_metrics, latent_test, test_labels, types):
    print('\n' + '='*60)
    print(f'Experiment: {name}')
    print('='*60)
    print(f"Train token acc: {train_metrics['token']:.4f}, Train seq acc: {train_metrics['seq']:.4f}")
    print(f"Test  token acc: {test_metrics['token']:.4f}, Test  seq acc: {test_metrics['seq']:.4f}")
    # silhouette
    latent_np = latent_test.cpu().numpy()
    try:
        sil = silhouette_score(latent_np, test_labels)
    except Exception:
        sil = float('nan')
    print(f"Silhouette score (test latent): {sil}")

    # PCA 简介 + 绘图
    try:
        pca = PCA(n_components=min(3, latent_np.shape[1]))
        proj = pca.fit_transform(latent_np)
        print(f"PCA explained variance ratio (first {proj.shape[1]}): {pca.explained_variance_ratio_}")
        # 打印每个 type 的均值坐标（前2维）
        for tidx, tname in enumerate(types):
            mask = (test_labels == tidx)
            if mask.sum() > 0:
                mean_coord = proj[mask].mean(axis=0)
                print(f" Type {tname} mean proj (first 2): {mean_coord[:2]}")

        # 绘制 PCA 散点：优先 3D（若有 3 个主成分），否则退回 2D
        colors = plt.cm.tab10(np.linspace(0,1,len(types)))
        if proj.shape[1] >= 3:
            fig = plt.figure(figsize=(7,6))
            ax = fig.add_subplot(111, projection='3d')
            for tidx, tname in enumerate(types):
                mask = (test_labels == tidx)
                if mask.sum() > 0:
                    ax.scatter(proj[mask,0], proj[mask,1], proj[mask,2], label=tname, color=colors[tidx], alpha=0.8)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            ax.set_title(f'PCA')
            ax.legend(markerscale=1, fontsize='small')

            os.makedirs(SAVE_DIR, exist_ok=True)
            fname = os.path.join(SAVE_DIR, f"{name}_pca3d_seed{seed}.png")
            fig.savefig(fname, bbox_inches='tight')
            plt.close(fig)
        else:
            fig, ax = plt.subplots(figsize=(6,5))
            for tidx, tname in enumerate(types):
                mask = (test_labels == tidx)
                if mask.sum() > 0:
                    ax.scatter(proj[mask,0], proj[mask,1] if proj.shape[1] > 1 else np.zeros_like(proj[mask,0]),
                               label=tname, color=colors[tidx], alpha=0.8)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title(f'PCA 2D - {name}')
            ax.legend(markerscale=1, fontsize='small')

            os.makedirs(SAVE_DIR, exist_ok=True)
            fname = os.path.join(SAVE_DIR, f"{name}_pca2d_seed{seed}.png")
            fig.savefig(fname, bbox_inches='tight')
            plt.close(fig)
    except Exception as e:
        print('PCA failed:', e)

    # type 内相似度并绘制相似度热图（按 label 排序，并生成 group-mean 矩阵）
    try:
        # 相似度矩阵（cosine similarity）
        sim = 1 - squareform(pdist(latent_np, metric='cosine'))  # (B,B)

        # 按标签排序，使同类连续
        order = np.argsort(test_labels)
        sim_ord = sim[np.ix_(order, order)]
        labels_ord = np.array(test_labels)[order]

        # 绘制按标签排序的样本级相似度热图
        fig, ax = plt.subplots(figsize=(6,6))
        im = ax.imshow(sim_ord, cmap='coolwarm', vmin=-1, vmax=1, aspect='equal')
        ax.set_title(f'Latent Similarity')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 在组边界处画线分割
        unique_labels, counts = np.unique(labels_ord, return_counts=True)
        cum = np.cumsum(counts)
        for c in cum[:-1]:
            ax.axhline(c - 0.5, color='k', linewidth=0.8)
            ax.axvline(c - 0.5, color='k', linewidth=0.8)

        # 将 ticks 放在每个 block 中心并显示类型名
        tick_pos = np.cumsum(counts) - counts/2.0
        ax.set_xticks(tick_pos)
        ax.set_yticks(tick_pos)
        ax.set_xticklabels(types, rotation=90, fontsize='small')
        ax.set_yticklabels(types, fontsize='small')

        os.makedirs(SAVE_DIR, exist_ok=True)
        fname = os.path.join(SAVE_DIR, f"{name}_sim_by_label_seed{seed}.png")
        fig.savefig(fname, bbox_inches='tight')
        plt.close(fig)

        for tidx, tname in enumerate(types):
            mask = (np.array(test_labels) == tidx)
            if mask.sum() > 1:
                type_lat = latent_np[mask]
                sims = 1 - pdist(type_lat, metric='cosine')
                print(f"Type {tname} mean intra-similarity: {sims.mean():.4f}")
    except Exception as e:
        print('Similarity plotting failed:', e)


def run_experiment():
    
    set_seed(seed)

    model = RNNAutoencoder(
        alpha, d_hidden, num_layers, d_latent, L,
        enable_position_encoding=False,
        pos_enc_scale=0.02,
        enable_serialize=ENABLE_SERIALIZE_Z,
        serialize_type='linear',
        latent_dropout_p=0.0
    ).to(device)
    # create dataset (train/test splits) and obtain canonical type/templates
    seq_train, seq_test, labels_train, labels_test, types = generate_instances(alpha, L, m, frac_train=0.8)
    X_train = sequences_to_tensor(seq_train, alpha).to(device)
    X_test = sequences_to_tensor(seq_test, alpha).to(device)
    train_labels = torch.tensor(labels_train, dtype=torch.long)
    test_labels = torch.tensor(labels_test, dtype=torch.long)

    # in-model options behave as no-ops that match the earlier improved pipeline.

    history = train_autoencoder_improved(
        model, X_train, X_test, train_labels, test_labels,
        n_epochs=epochs, lr=lr, weight_decay=weight_decay,
        use_augment=ENABLE_AUGMENT
    )

    # after training, get final outputs
    model.eval()
    with torch.no_grad():
        _, z_test, test_output = model(X_test)
        _, z_train_full, _ = model(X_train)
        pred_test = torch.argmax(test_output, dim=-1)
        target_test = torch.argmax(X_test, dim=-1)
        test_token_acc = (pred_test == target_test).float().mean().item()
        test_seq_acc = (pred_test == target_test).all(dim=0).float().mean().item()

    # training metrics from history (last epoch)
    train_token_acc = history['train_token_acc'][-1] if 'train_token_acc' in history else 0.0
    train_seq_acc = history['train_seq_acc'][-1] if 'train_seq_acc' in history else 0.0
    train_metrics = {'token': train_token_acc, 'seq': train_seq_acc}
    test_metrics = {'token': test_token_acc, 'seq': test_seq_acc}

    # 打印并计算 PCA/silhouette/相似度
    test_labels_np = test_labels.cpu().numpy() if isinstance(test_labels, torch.Tensor) else np.array(test_labels)
    compute_metrics_and_print('ablation_test', train_metrics, test_metrics, z_test, test_labels_np, types)


if __name__ == '__main__':
    # 使用说明：在顶部开关打开你想测试的改动，确保一次只启用一个改动（或只启用 baseline）
    run_experiment()
