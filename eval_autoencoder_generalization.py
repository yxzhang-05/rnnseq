import numpy as np
import torch
from collections import defaultdict
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pre_model import RNNAutoencoder
import string
import itertools
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D  
from scipy.spatial.distance import pdist, squareform


def replace_symbols(sequence, symbols):
    newseq = np.array(list(sequence))
    old_symbols = np.unique(newseq)
    assert len(old_symbols) == len(symbols)
    for old, new in zip(old_symbols, symbols):
        newseq[newseq == old] = new
    return "".join(newseq)

def generate_types(L=4, m=2):
    """
    生成所有可能的序列类型，使用正好m个字母的L长度序列
    """
    import itertools
    alphabet = list(string.ascii_lowercase[:m])
    all_types = [''.join(p) for p in itertools.product(alphabet, repeat=L)]
    # 只保留使用正好m个唯一字母的类型
    types = [t for t in all_types if len(set(t)) == m]
    print(f"生成了 {len(types)} 种序列类型")
    return types


def generate_instances(alpha, frac_train=0.8):
    # 所有长度4、m=2 的唯一结构类型（共 6 个）
    types = ["aaab", "aaba", "abaa", "aabb", "abba", "abab"]

    alphabet = list(string.ascii_lowercase[:alpha])
    m = 2   # 固定两个不同符号

    # 所有符号替换组合，例如 (a,b), (a,c), ...
    all_perms = list(itertools.permutations(alphabet, m))

    seqs = []
    labels = []

    for type_idx, t in enumerate(types):
        for perm in all_perms:
            seqs.append(replace_symbols(t, perm))
            labels.append(type_idx)

    seqs = np.array(seqs)
    labels = np.array(labels)

    # train/test split
    N = len(seqs)
    idx = np.random.permutation(N)
    split = int(N * frac_train)

    return seqs[idx[:split]], seqs[idx[split:]], labels[idx[:split]], labels[idx[split:]]

def sequences_to_tensor(sequences, alpha):
    # Back to one-hot for pre_model
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


# =========================================================
# 训练 autoencoder
# =========================================================
def latent_regularizer(z, std_coeff=1.0, cov_coeff=1.0, eps=1e-4):

    # ---------- (1) Variance term: 防止 collapse ----------
    # 每一维的 std 应该 >= 1，这能拉开 latent
    std = torch.sqrt(z.var(dim=0) + eps)
    var_loss = torch.mean(F.relu(1.0 - std))   # std 太小 → 惩罚

    # ---------- (2) Covariance term: 减少维度相关性 ----------
    z = z - z.mean(dim=0, keepdim=True)
    N, D = z.shape
    cov = (z.T @ z) / (N - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag ** 2).sum() / D

    return std_coeff * var_loss + cov_coeff * cov_loss

def evaluate(model, X, y_target):

    model.eval()
    with torch.no_grad():
        _, _, logits = model(X)         # (seq_len, batch, vocab)
        preds = logits.argmax(dim=-1)   # (seq_len, batch)

        correct = (preds == y_target).float()
        token_acc = correct.mean().item()

        # sequence-level acc：整个 seq 全对
        seq_correct = correct.prod(dim=0)  # (batch,)
        seq_acc = seq_correct.mean().item()

    return token_acc, seq_acc


def train_autoencoder(model, X_train, X_test, n_epochs=200, lr=0.001, weight_decay=1e-3):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = defaultdict(list)

    for epoch in range(n_epochs):
        # ===== 训练阶段 =====
        model.train()
        optimizer.zero_grad()
        
        # 前向传播
        hidden, latent, output = model(X_train)  # output shape: (batch, seq_len, vocab_size)
        
        # Cross-entropy loss
        ce_loss = F.cross_entropy(
            output.reshape(-1, output.shape[-1]),             # (batch*seq_len, vocab_size)
            torch.argmax(X_train, dim=-1).reshape(-1)         # (batch*seq_len,)
        )
        
        # 潜在空间正则
        latent_reg = latent_regularizer(latent, std_coeff=5, cov_coeff=10)
        train_loss = ce_loss
        
        # 反向传播
        train_loss.backward()
        optimizer.step()
        
        # ===== 训练准确率 =====
        with torch.no_grad():
            pred_train = torch.argmax(output, dim=-1)         # (batch, seq_len)
            target_train = torch.argmax(X_train, dim=-1)     # (batch, seq_len)
            
            train_token_acc = (pred_train.reshape(-1) == target_train.reshape(-1)).float().mean().item()
            # sequences are along dim=0 (seq_len, batch), so check all tokens per sequence via dim=0
            train_seq_acc = (pred_train == target_train).all(dim=0).float().mean().item()
        
        # ===== 测试阶段 =====
        model.eval()
        with torch.no_grad():
            _, _, test_output = model(X_test)  # shape: (batch, seq_len, vocab_size)
            
            test_loss = F.cross_entropy(
                test_output.reshape(-1, test_output.shape[-1]),
                torch.argmax(X_test, dim=-1).reshape(-1)
            )
            
            pred_test = torch.argmax(test_output, dim=-1)
            target_test = torch.argmax(X_test, dim=-1)
            
            test_token_acc = (pred_test.reshape(-1) == target_test.reshape(-1)).float().mean().item()
            test_seq_acc = (pred_test == target_test).all(dim=0).float().mean().item()
        
        # ===== 保存历史 =====
        history['train_loss'].append(train_loss.item())
        history['ce_loss'].append(ce_loss.item())
        history['latent_reg'].append(latent_reg.item())
        history['test_loss'].append(test_loss.item())
        history['train_token_acc'].append(train_token_acc)
        history['test_token_acc'].append(test_token_acc)
        history['train_seq_acc'].append(train_seq_acc)
        history['test_seq_acc'].append(test_seq_acc)
        
        # 打印进度
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: "
                  f"Train Loss={train_loss.item():.4f} (CE={ce_loss.item():.4f}, Reg={latent_reg.item():.4f}) "
                  f"Token={train_token_acc:.3f} Seq={train_seq_acc:.3f} | "
                  f"Test Loss={test_loss.item():.4f} Token={test_token_acc:.3f} Seq={test_seq_acc:.3f}")
    
    return history

# =========================================================
# 泛化分析 + latent similarity
# =========================================================
def analyze_generalization(model, X_train, X_test,
                           train_labels, test_labels, types, alpha):

    model.eval()

    # -----------------------------------------
    # Forward pass
    # -----------------------------------------
    with torch.no_grad():
        _, latent_train, out_train = model(X_train)
        _, latent_test, out_test = model(X_test)

    # -----------------------------------------
    # Accuracy
    # -----------------------------------------
    results = {}

    target_train = torch.argmax(X_train, dim=-1)
    target_test = torch.argmax(X_test, dim=-1)

    results["train_token_acc"] = (torch.argmax(out_train, -1) == target_train).float().mean().item()
    results["test_token_acc"] = (torch.argmax(out_test, -1) == target_test).float().mean().item()

    results["train_seq_acc"] = (torch.argmax(out_train, -1) == target_train).all(dim=0).float().mean().item()
    results["test_seq_acc"] = (torch.argmax(out_test, -1) == target_test).all(dim=0).float().mean().item()

    print("\n=== Accuracy Summary ===")
    print(f"Token Acc: train={results['train_token_acc']:.3f}, test={results['test_token_acc']:.3f}")
    print(f"Seq Acc:   train={results['train_seq_acc']:.3f}, test={results['test_seq_acc']:.3f}")

    # -----------------------------------------
    # Latent similarity
    # -----------------------------------------
    unique_types = sorted(set(train_labels))

    lt_train = latent_train.cpu().numpy()
    lt_test = latent_test.cpu().numpy()

    type_means_train = np.array([lt_train[train_labels == t].mean(0) for t in unique_types])
    type_means_test = np.array([lt_test[test_labels == t].mean(0) for t in unique_types])

    sim_train = 1 - squareform(pdist(type_means_train, "cosine"))
    sim_test = 1 - squareform(pdist(type_means_test, "cosine"))

    print("\nTrain latent similarity:")
    print(sim_train)
    print("\nTest latent similarity:")
    print(sim_test)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, sim, title in zip(axes, [sim_train, sim_test], ["Train", "Test"]):
        im = ax.imshow(sim, cmap="RdYlBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(types)))
        ax.set_yticks(range(len(types)))
        ax.set_xticklabels(types, rotation=45)
        ax.set_yticklabels(types)
        ax.set_title(title)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cbar_ax)
    plt.show()

    # -----------------------------------------
    # PCA
    # -----------------------------------------
    Z = np.vstack([lt_train, lt_test])
    Z_pca = PCA(n_components=3).fit_transform(Z)
    Ztrain, Ztest = Z_pca[:len(lt_train)], Z_pca[len(lt_train):]

    fig = plt.figure(figsize=(9, 4))

    for i, (Zp, labels, title) in enumerate(
            [(Ztrain, train_labels, "Train PCA 3D"), (Ztest, test_labels, "Test PCA 3D")]
    ):
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        # labels are numeric indices (0..K-1); `types` contains string names for those indices
        for label_idx in sorted(np.unique(labels)):
            mask = labels == label_idx
            # Use provided `types` list to get human-readable name if available
            try:
                lname = types[int(label_idx)]
            except Exception:
                lname = str(label_idx)
            if mask.sum() == 0:
                continue
            ax.scatter(Zp[mask, 0], Zp[mask, 1], Zp[mask, 2], s=25, label=str(lname))
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.legend()

    plt.tight_layout()
    plt.show()

    # -----------------------------------------
    # Silhouette & cluster purity
    # -----------------------------------------
    print("\n=== Cluster Metrics ===")
    silhouette_train = silhouette_score(lt_train, train_labels)
    silhouette_test = silhouette_score(lt_test, test_labels)

    print(f"Silhouette (train): {silhouette_train:.3f}")
    print(f"Silhouette (test) : {silhouette_test:.3f}")

    results["silhouette_train"] = silhouette_train
    results["silhouette_test"] = silhouette_test

    # -----------------------------------------
    # Latent norm distribution
    # -----------------------------------------
    # norms_train = np.linalg.norm(lt_train, axis=1)
    # norms_test = np.linalg.norm(lt_test, axis=1)

    # plt.hist(norms_train, bins=30, alpha=0.6, label="Train")
    # plt.hist(norms_test, bins=30, alpha=0.6, label="Test")
    # plt.title("Latent Norm Distribution")
    # plt.legend()
    # plt.show()

    # -----------------------------------------
    # Decoder confidence
    # -----------------------------------------
    # conf_train = out_train.softmax(-1).max(-1).values.cpu().numpy().flatten()
    # conf_test = out_test.softmax(-1).max(-1).values.cpu().numpy().flatten()

    # plt.hist(conf_train, bins=30, alpha=0.6, label="Train")
    # plt.hist(conf_test, bins=30, alpha=0.6, label="Test")
    # plt.title("Decoder Confidence Histogram")
    # plt.legend()
    # plt.show()

    # return results


# =========================================================
# 主 pipeline
# =========================================================
def run_pipeline():
    L, m, alpha = 4, 2, 6  
    d_embed, d_hidden, d_latent = 4, 32, 8
    frac_train = 0.8  
    n_epochs = 200  
    lr = 0.01
    weight_decay = 0

    print(f"配置: L={L}, m={m}, alpha={alpha}, d_latent={d_latent}")
    print(f"训练: epochs={n_epochs}, lr={lr}, weight_decay={weight_decay}")

    train_seqs, test_seqs, train_labels, test_labels = generate_instances(
        alpha, frac_train
    )
    print(f"训练序列数: {len(train_seqs)}, 测试序列数: {len(test_seqs)}")

    types_used = ["aaab", "aaba", "abaa", "aabb", "abba", "abab"]

    X_train = sequences_to_tensor(train_seqs, alpha)
    X_test = sequences_to_tensor(test_seqs, alpha)
    print(f"输入形状: train {X_train.shape}, test {X_test.shape}")

    model = RNNAutoencoder(alpha, d_hidden, 1, d_latent, L)
    print(f"模型参数: {sum(p.numel() for p in model.parameters())}")

    history = train_autoencoder(model, X_train, X_test, n_epochs, lr, weight_decay)

    results = analyze_generalization(
        model, X_train, X_test,
        train_labels, test_labels,
        types_used, alpha
    )

    return model, history, results


if __name__ == "__main__":
    run_pipeline()
