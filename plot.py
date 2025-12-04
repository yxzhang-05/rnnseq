import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from model import RNNAutoencoder
from test import generate_instances, sequences_to_tensor, train, plot_all_diagnostics
from sklearn.metrics import silhouette_score

def run_acc_vs_weight_experiment():
    # 1. 实验配置
    # 权重范围：从 0 到 4
    contrastive_weights = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
    serialize_opts = [False, True]
    
    # 固定参数 (保持实验公平)
    seed = 42
    L, m, alpha = 4, 2, 6
    epochs = 1000   # 保证收敛
    lr = 1e-3
    d_hidden = 32
    d_latent = 8
    device = torch.device('cpu')
    
    # We'll generate data per run after creating the model so RNG call order
    # matches `test.run_experiment` (which creates the model before data).
    
    # 存储结果: {Serialize_Bool: [acc_w0, acc_w1, ...]}
    results = {False: [], True: []}
    results_sil = {False: [], True: []}

    print(f"Start Experiment: Weights={contrastive_weights}, Serialize={serialize_opts}")

    # 2. 循环运行实验
    for ser in serialize_opts:
        ser_str = "Serialize" if ser else "No Serialize"
        print(f"\n=== Testing {ser_str} ===")
        
        for w in contrastive_weights:
            print(f"  Running Weight = {w} ...", end=" ")
            # reset RNGs and create model first (match test.run_experiment ordering)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            model = RNNAutoencoder(
                alpha, d_hidden, 1, d_latent, L,
                enable_serialize=ser
            ).to(device)

            # now generate data (same seed -> same data across runs)
            seq_train, seq_test, labels_train, labels_test, types = generate_instances(alpha, L, m, frac_train=0.8)
            X_train = sequences_to_tensor(seq_train, alpha).to(device)
            X_test = sequences_to_tensor(seq_test, alpha).to(device)
            test_labels = torch.tensor(labels_test, dtype=torch.long)

            # 训练
            history = train(
                model, X_train, X_test, test_labels, train_labels=torch.tensor(labels_train, dtype=torch.long),
                n_epochs=epochs, lr=lr,
                use_contrastive=True,
                contrastive_weight=float(w), types=types, save_dir='results'
            )

            # final-epoch diagnostics (SVG)
            try:
                plot_all_diagnostics(model, X_train, torch.tensor(labels_train, dtype=torch.long), X_test, test_labels, types, save_dir='results', history=history)
            except Exception:
                pass
            
            # 获取最终 Test Accuracy
            model.eval()
            with torch.no_grad():
                _, _, test_output = model(X_test)
                pred = torch.argmax(test_output, dim=-1)
                target = torch.argmax(X_test, dim=-1)
                acc = (pred == target).all(dim=0).float().mean().item()
            
            results[ser].append(acc)
            # compute silhouette on test latents
            model.eval()
            with torch.no_grad():
                _, z_test, _ = model(X_test)
            try:
                sil = float(silhouette_score(z_test.cpu().numpy(), labels_test))
            except Exception:
                sil = float('nan')
            results_sil[ser].append(sil)
            print(f"Done. Acc: {acc:.2%}")
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    fig, ax = plt.subplots(figsize=(4, 4))

    # 绘制两条线
    # Line 1: No Serialize
    ax.plot(contrastive_weights, results[False], 
            marker='o', markersize=6, linewidth=2.5, linestyle='-', 
            color='#1f77b4', label='No Serialize')

    # Line 2: Serialize
    ax.plot(contrastive_weights, results[True], 
            marker='s', markersize=6, linewidth=2.5, linestyle='--', 
            color="#317826", label='Serialize')   

    # 图表装饰
    ax.set_xlabel('Contrastive Loss Weight', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test Accuracy', fontsize=14, fontweight='bold')
    
    # 设置坐标轴范围和刻度
    ax.set_xticks(contrastive_weights)
    ax.set_ylim(-0.05, 1.05) # 留一点余地
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Place legend above the plot, centered, no frame
    ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False)

    # 保存图片
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "acc_vs_weight.svg")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {save_path}")

    # --- Silhouette vs weight ---
    print("Plotting silhouette vs weight...")
    fig, ax = plt.subplots(figsize=(4, 4))

    ax.plot(contrastive_weights, results_sil[False],
            marker='o', markersize=6, linewidth=2.5, linestyle='-',
            color='#1f77b4', label='No Serialize')

    ax.plot(contrastive_weights, results_sil[True],
            marker='s', markersize=6, linewidth=2.5, linestyle='--',
            color='#317826', label='Serialize')

    ax.set_xlabel('Contrastive Loss Weight', fontsize=14, fontweight='bold')
    ax.set_ylabel('Silhouette Score', fontsize=14, fontweight='bold')
    ax.set_xticks(contrastive_weights)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, linestyle=':', alpha=0.6)
    # Place legend above the plot, centered, no frame
    ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False)

    save_path = os.path.join(save_dir, "silhouette_vs_weight.svg")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Silhouette plot saved to: {save_path}")

if __name__ == '__main__':
    run_acc_vs_weight_experiment()