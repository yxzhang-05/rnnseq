import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
import os
from test import generate_instances, sequences_to_tensor

# ============================================================================
# 核心发现：Serialization让Decoder承担了大部分工作
# ============================================================================

class ExperimentalAutoencoder(nn.Module):
    """
    可配置的autoencoder，用于实验不同的设计选择
    """
    def __init__(self, d_input, d_hidden, d_latent, seq_length,
                 enable_serialize=False,
                 decoder_type='rnn',  # 'rnn', 'linear', 'mlp'
                 share_decoder_weights=False,
                 device='cpu'):
        super().__init__()
        
        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_latent = d_latent
        self.seq_length = seq_length
        self.enable_serialize = enable_serialize
        self.decoder_type = decoder_type
        self.device = device
        
        # Encoder (固定为RNN)
        self.encoder_rnn = nn.LSTM(d_input, d_hidden, 1, batch_first=False)
        self.encoder_proj = nn.Linear(d_hidden, d_latent)
        
        # Serialization (可选)
        if enable_serialize:
            self.serialize = nn.Linear(d_latent, seq_length * d_latent)
        
        # Decoder (可变)
        if decoder_type == 'rnn':
            # 标准RNN decoder
            if enable_serialize:
                self.decoder_rnn = nn.LSTM(d_latent, d_hidden, 1, batch_first=False)
            else:
                self.decoder_rnn = nn.LSTM(d_latent, d_hidden, 1, batch_first=False)
            self.decoder_output = nn.Linear(d_hidden, d_input)
        
        elif decoder_type == 'linear':
            # 纯linear decoder（测试你的发现）
            if enable_serialize:
                # 对每个位置独立的linear
                self.decoder_linears = nn.ModuleList([
                    nn.Linear(d_latent, d_input) for _ in range(seq_length)
                ])
            else:
                # 单个shared linear
                self.decoder_linear = nn.Linear(d_latent, d_input)
        
        elif decoder_type == 'mlp':
            # MLP decoder（介于linear和RNN之间）
            if enable_serialize:
                self.decoder_mlps = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(d_latent, d_hidden),
                        nn.ReLU(),
                        nn.Linear(d_hidden, d_input)
                    ) for _ in range(seq_length)
                ])
            else:
                self.decoder_mlp = nn.Sequential(
                    nn.Linear(d_latent, d_hidden),
                    nn.ReLU(),
                    nn.Linear(d_hidden, d_input)
                )
    
    def forward(self, x):
        # x: (L, B, d_input)
        L, B, _ = x.shape
        
        # Encode
        _, (h, _) = self.encoder_rnn(x)
        latent = self.encoder_proj(h[-1])  # (B, d_latent)
        
        # Serialize (可选)
        if self.enable_serialize:
            latent_serialized = self.serialize(latent)  # (B, L*d_latent)
            latent_seq = latent_serialized.view(B, L, self.d_latent).permute(1, 0, 2)
        else:
            latent_seq = latent.unsqueeze(0).expand(L, B, -1)
        
        # Decode
        if self.decoder_type == 'rnn':
            decoder_out, _ = self.decoder_rnn(latent_seq)
            output = self.decoder_output(decoder_out)
        
        elif self.decoder_type == 'linear':
            if self.enable_serialize:
                # 对每个位置用不同的linear
                outputs = []
                for t in range(L):
                    out_t = self.decoder_linears[t](latent_seq[t])
                    outputs.append(out_t)
                output = torch.stack(outputs, dim=0)
            else:
                # 所有位置共享同一个linear
                output = self.decoder_linear(latent_seq)
        
        elif self.decoder_type == 'mlp':
            if self.enable_serialize:
                outputs = []
                for t in range(L):
                    out_t = self.decoder_mlps[t](latent_seq[t])
                    outputs.append(out_t)
                output = torch.stack(outputs, dim=0)
            else:
                output = self.decoder_mlp(latent_seq)
        
        return latent, output


# ============================================================================
# 实验1：不同decoder配置的性能对比
# ============================================================================

def experiment_decoder_comparison(X_train, X_test, train_labels, test_labels,
                                 d_input=8, d_hidden=128, d_latent=32,
                                 n_epochs=1000, lr=1e-3):
    """
    对比实验：
    1. RNN decoder + No serialize
    2. RNN decoder + Serialize
    3. Linear decoder + No serialize
    4. Linear decoder + Serialize
    5. MLP decoder + Serialize
    """
    
    seq_length = X_train.shape[0]
    device = X_train.device
    
    configs = [
        ('RNN-NoSerialize', 'rnn', False),
        ('RNN-Serialize', 'rnn', True),
        ('Linear-NoSerialize', 'linear', False),
        ('Linear-Serialize', 'linear', True),
        ('MLP-Serialize', 'mlp', True),
    ]
    
    results = {}
    
    for name, decoder_type, enable_serialize in configs:
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")
        
        model = ExperimentalAutoencoder(
            d_input, d_hidden, d_latent, seq_length,
            enable_serialize=enable_serialize,
            decoder_type=decoder_type,
            device=device
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        
        # 训练
        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            
            latent, output = model(X_train)
            loss = F.cross_entropy(
                output.reshape(-1, output.shape[-1]),
                torch.argmax(X_train, dim=-1).reshape(-1)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if (epoch + 1) % 200 == 0:
                model.eval()
                with torch.no_grad():
                    test_latent, test_output = model(X_test)
                    pred = torch.argmax(test_output, dim=-1)
                    target = torch.argmax(X_test, dim=-1)
                    test_acc = (pred == target).all(dim=0).float().mean().item()
                    
                    try:
                        sil = silhouette_score(
                            test_latent.cpu().numpy(),
                            test_labels.cpu().numpy()
                        )
                    except:
                        sil = -999
                
                print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, "
                      f"TestAcc={test_acc:.3f}, Sil={sil:.3f}")
        
        # 最终评估
        model.eval()
        with torch.no_grad():
            test_latent, test_output = model(X_test)
            pred = torch.argmax(test_output, dim=-1)
            target = torch.argmax(X_test, dim=-1)
            test_acc = (pred == target).all(dim=0).float().mean().item()
            
            latent_np = test_latent.cpu().numpy()
            sil = silhouette_score(latent_np, test_labels.cpu().numpy())
            
            # 计算latent的统计特性
            latent_std = latent_np.std(axis=0).mean()
            latent_range = latent_np.max() - latent_np.min()
        
        results[name] = {
            'model': model,
            'test_acc': test_acc,
            'silhouette': sil,
            'latent': test_latent,
            'latent_std': latent_std,
            'latent_range': latent_range
        }
        
        print(f"\nFinal Results:")
        print(f"  Test Acc: {test_acc:.4f}")
        print(f"  Silhouette: {sil:.4f}")
        print(f"  Latent Std: {latent_std:.4f}")
        print(f"  Latent Range: {latent_range:.4f}")
    
    return results


# ============================================================================
# 实验2：分析latent的"信息熵"和"利用率"
# ============================================================================

def analyze_latent_information(results, test_labels, save_dir='results'):
    """
    分析不同配置下latent的信息含量
    
    关键问题：
    - Latent维度是否都被有效利用？
    - 哪些维度携带了类型信息？
    - 是否存在"坍塌"现象（所有样本映射到相似的latent）？
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (name, data) in enumerate(results.items()):
        if idx >= 6:
            break
        
        ax = axes[idx]
        latent_np = data['latent'].cpu().numpy()
        
        # 1. 计算每个维度的方差
        dim_variances = latent_np.var(axis=0)
        
        # 2. 计算有效秩（effective rank）
        # 类似于PCA，看需要多少维才能解释90%的方差
        pca = PCA()
        pca.fit(latent_np)
        explained_var_cumsum = np.cumsum(pca.explained_variance_ratio_)
        effective_rank = np.argmax(explained_var_cumsum >= 0.9) + 1
        
        # 绘制
        ax.bar(range(len(dim_variances)), dim_variances, color='steelblue', alpha=0.7)
        ax.set_title(f'{name}\nSil={data["silhouette"]:.3f}, Acc={data["test_acc"]:.3f}\n'
                    f'Effective Rank={effective_rank}/{len(dim_variances)}',
                    fontsize=10)
        ax.set_xlabel('Latent Dimension')
        ax.set_ylabel('Variance')
        ax.axhline(y=dim_variances.mean(), color='r', linestyle='--', 
                  label=f'Mean={dim_variances.mean():.3f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/latent_dimension_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_dir}/latent_dimension_analysis.png")
    plt.close()
    
    # 生成报告
    print("\n" + "="*70)
    print("Latent Information Analysis")
    print("="*70)
    
    for name, data in results.items():
        latent_np = data['latent'].cpu().numpy()
        
        # 有效秩
        pca = PCA()
        pca.fit(latent_np)
        explained_var_cumsum = np.cumsum(pca.explained_variance_ratio_)
        effective_rank = np.argmax(explained_var_cumsum >= 0.9) + 1
        
        # 维度利用率（方差>阈值的维度数）
        dim_variances = latent_np.var(axis=0)
        threshold = dim_variances.mean() * 0.1
        active_dims = (dim_variances > threshold).sum()
        
        print(f"\n{name}:")
        print(f"  Test Acc: {data['test_acc']:.4f}")
        print(f"  Silhouette: {data['silhouette']:.4f}")
        print(f"  Effective Rank: {effective_rank}/{len(dim_variances)}")
        print(f"  Active Dimensions: {active_dims}/{len(dim_variances)}")
        print(f"  Latent Std: {data['latent_std']:.4f}")


# ============================================================================
# 实验3：可视化decoder的"记忆"能力
# ============================================================================

def visualize_decoder_memory(results, X_test, test_labels, save_dir='results'):
    """
    关键实验：测试decoder是否"记住"了每个位置应该输出什么
    
    方法：
    1. 给decoder输入随机latent
    2. 看它能否仍然产生合理的输出
    3. 如果能 → decoder记住了位置信息（不好）
       如果不能 → decoder依赖latent（好）
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("Decoder Memory Test")
    print("="*70)
    print("\n给decoder喂随机latent，看它能否仍然重建序列...")
    
    for name, data in results.items():
        model = data['model']
        model.eval()
        
        with torch.no_grad():
            # 1. 正常前向（真实latent）
            real_latent, real_output = model(X_test)
            real_pred = torch.argmax(real_output, dim=-1)
            real_target = torch.argmax(X_test, dim=-1)
            real_acc = (real_pred == real_target).all(dim=0).float().mean().item()
            
            # 2. 随机latent前向
            B = X_test.shape[1]
            random_latent = torch.randn_like(real_latent)
            
            # 手动调用decoder
            L = model.seq_length
            if model.enable_serialize:
                latent_serialized = model.serialize(random_latent)
                latent_seq = latent_serialized.view(B, L, model.d_latent).permute(1, 0, 2)
            else:
                latent_seq = random_latent.unsqueeze(0).expand(L, B, -1)
            
            if model.decoder_type == 'rnn':
                decoder_out, _ = model.decoder_rnn(latent_seq)
                random_output = model.decoder_output(decoder_out)
            elif model.decoder_type == 'linear':
                if model.enable_serialize:
                    outputs = []
                    for t in range(L):
                        outputs.append(model.decoder_linears[t](latent_seq[t]))
                    random_output = torch.stack(outputs, dim=0)
                else:
                    random_output = model.decoder_linear(latent_seq)
            elif model.decoder_type == 'mlp':
                if model.enable_serialize:
                    outputs = []
                    for t in range(L):
                        outputs.append(model.decoder_mlps[t](latent_seq[t]))
                    random_output = torch.stack(outputs, dim=0)
                else:
                    random_output = model.decoder_mlp(latent_seq)
            
            random_pred = torch.argmax(random_output, dim=-1)
            random_acc = (random_pred == real_target).all(dim=0).float().mean().item()
            
            # 3. 计算"记忆比例"
            # 如果random latent也能得到高准确率，说明decoder记住了位置
            memory_ratio = random_acc / (real_acc + 1e-8)
        
        print(f"\n{name}:")
        print(f"  Real Latent Acc: {real_acc:.4f}")
        print(f"  Random Latent Acc: {random_acc:.4f}")
        print(f"  Memory Ratio: {memory_ratio:.4f}")
        
        if memory_ratio > 0.5:
            print(f"  ⚠️  WARNING: Decoder记住了大量位置信息！")
        else:
            print(f"  ✓  Decoder properly依赖latent")


# ============================================================================
# 实验4：梯度流分析
# ============================================================================

def analyze_gradient_flow(model, X_sample):
    """
    分析梯度如何从loss反向传播到latent
    
    关键问题：
    - Serialization是否阻断了对latent的梯度？
    - Decoder的参数量是否让它"接管"了学习？
    """
    
    model.train()
    latent, output = model(X_sample)
    
    # 计算loss
    loss = F.cross_entropy(
        output.reshape(-1, output.shape[-1]),
        torch.argmax(X_sample, dim=-1).reshape(-1)
    )
    
    # 反向传播
    loss.backward()
    
    # 收集梯度统计
    grad_stats = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_mean = param.grad.abs().mean().item()
            grad_stats[name] = {
                'norm': grad_norm,
                'mean': grad_mean,
                'shape': param.shape
            }
    
    return grad_stats


def compare_gradient_flows(results, X_sample):
    """对比不同配置的梯度流"""
    
    print("\n" + "="*70)
    print("Gradient Flow Analysis")
    print("="*70)
    
    for name, data in results.items():
        model = data['model']
        model.zero_grad()
        
        grad_stats = analyze_gradient_flow(model, X_sample)
        
        print(f"\n{name}:")
        
        # Encoder部分的梯度
        encoder_grad = grad_stats.get('encoder_proj.weight', {}).get('norm', 0)
        
        # Serialization的梯度
        serialize_grad = grad_stats.get('serialize.weight', {}).get('norm', 0) if 'serialize.weight' in grad_stats else 0
        
        # Decoder部分的梯度
        decoder_grads = [v['norm'] for k, v in grad_stats.items() if 'decoder' in k]
        decoder_grad_total = sum(decoder_grads)
        
        print(f"  Encoder Projection Grad: {encoder_grad:.6f}")
        print(f"  Serialize Grad: {serialize_grad:.6f}")
        print(f"  Decoder Total Grad: {decoder_grad_total:.6f}")
        print(f"  Decoder/Encoder Ratio: {decoder_grad_total/(encoder_grad+1e-8):.2f}")


# ============================================================================
# 实验5：信息流可视化
# ============================================================================

def visualize_information_flow(results, X_test, test_labels, save_dir='results'):
    """
    可视化信息如何从input流向output
    
    通过计算：
    - Latent与Input的互信息
    - Latent与Output的互信息
    - 判断信息是否通过latent传递，还是decoder"作弊"
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 5))
    if len(results) == 1:
        axes = [axes]
    
    for idx, (name, data) in enumerate(results.items()):
        ax = axes[idx]
        latent_np = data['latent'].cpu().numpy()
        labels_np = test_labels.cpu().numpy()
        
        # PCA投影到2D
        if latent_np.shape[1] > 2:
            pca = PCA(n_components=2)
            latent_2d = pca.fit_transform(latent_np)
            title_suffix = f"(PCA: {pca.explained_variance_ratio_.sum():.2f})"
        else:
            latent_2d = latent_np[:, :2]
            title_suffix = ""
        
        # 按类型着色
        n_types = len(np.unique(labels_np))
        colors = plt.cm.tab10(np.linspace(0, 1, n_types))
        
        for type_idx in range(n_types):
            mask = labels_np == type_idx
            ax.scatter(latent_2d[mask, 0], latent_2d[mask, 1],
                      c=[colors[type_idx]], alpha=0.6, s=30,
                      label=f'Type {type_idx}')
        
        ax.set_title(f'{name}\n{title_suffix}\nSil={data["silhouette"]:.3f}',
                    fontsize=10)
        ax.set_xlabel('Latent Dim 1')
        ax.set_ylabel('Latent Dim 2')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/latent_visualization.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_dir}/latent_visualization.png")
    plt.close()


# ============================================================================
# 主实验函数
# ============================================================================

def run_comprehensive_analysis(X_train, X_test, train_labels, test_labels,
                               d_input=8, d_hidden=32, d_latent=8,
                               n_epochs=1000, save_dir='results'):
    """
    运行完整的分析实验
    """
    
    print("="*70)
    print("重建准确性 vs 表征质量：综合分析实验")
    print("="*70)
    
    # 实验1: 对比不同decoder配置
    results = experiment_decoder_comparison(
        X_train, X_test, train_labels, test_labels,
        d_input, d_hidden, d_latent, n_epochs
    )
    
    # 实验2: 分析latent信息
    analyze_latent_information(results, test_labels, save_dir)
    
    # 实验3: 测试decoder记忆
    visualize_decoder_memory(results, X_test, test_labels, save_dir)
    
    # 实验4: 梯度流分析
    X_sample = X_train[:, :10, :]  # 取小batch用于分析
    compare_gradient_flows(results, X_sample)
    
    # 实验5: 可视化信息流
    visualize_information_flow(results, X_test, test_labels, save_dir)
    
    # 生成总结报告
    print("\n" + "="*70)
    print("总结报告")
    print("="*70)
    
    for name, data in results.items():
        print(f"\n{name}:")
        print(f"  ✓ Test Accuracy: {data['test_acc']:.4f}")
        print(f"  ✓ Silhouette Score: {data['silhouette']:.4f}")
        print(f"  ✓ Latent Std: {data['latent_std']:.4f}")
        
        # 判断
        if data['test_acc'] > 0.9 and data['silhouette'] < 0.1:
            print(f"  ⚠️  高重建但差表征：Decoder可能记住了位置信息")
        elif data['test_acc'] > 0.9 and data['silhouette'] > 0.3:
            print(f"  ✓  理想情况：好重建+好表征")
        elif data['test_acc'] < 0.8:
            print(f"  ⚠️  重建质量差：模型容量不足或训练不够")
    
    return results


if __name__ == "__main__":

    L, m, alpha = 4, 2, 6
    device = torch.device('cpu')
    seq_train, seq_test, labels_train, labels_test, types = generate_instances(
        alpha, L, m, frac_train=0.8)
    X_train = sequences_to_tensor(seq_train, alpha).to(device)
    X_test = sequences_to_tensor(seq_test, alpha).to(device)
    test_labels = torch.tensor(labels_test, dtype=torch.long)
    train_labels = torch.tensor(labels_train, dtype=torch.long)
    run_comprehensive_analysis(X_train, X_test, train_labels, test_labels, d_input=alpha)