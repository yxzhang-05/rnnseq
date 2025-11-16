"""
自编码器诊断脚本
用于快速理解模型结构、训练过程和性能瓶颈
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import RNNAutoencoder
from functions import make_dicts, make_tokens
import string

def diagnose_autoencoder():
    """诊断自编码器的基本功能和性能"""
    
    print("="*60)
    print("自编码器诊断报告")
    print("="*60)
    
    # ===== 1. 设置基本参数 =====
    print("\n1. 初始化参数...")
    L = 4  # 序列长度
    alpha = 10  # 字母表大小
    m = 2  # 每个序列的唯一字母数
    d_hidden = 64
    d_latent = 10
    n_types = 3
    frac_train = 0.8
    
    # ===== 2. 生成测试数据 =====
    print("\n2. 生成测试数据...")
    letter_to_index, index_to_letter = make_dicts(alpha)
    
    # 创建简单的类型 - 修复：使用正确的格式
    # 类型格式应该是长度为L的字符串，包含m个不同的字母
    # 例如 L=4, m=2: "aabb", "abab", "abba" 等
    types = []
    alphabet = list(string.ascii_lowercase)[:alpha]
    
    # 生成几个简单的模式
    patterns = [
        lambda a, b, L: (a * (L//2) + b * (L//2))[:L],  # aabb
        lambda a, b, L: ''.join([a if i%2==0 else b for i in range(L)]),  # abab
        lambda a, b, L: (a + b * (L-1))[:L],  # abbb
    ]
    
    for i in range(min(n_types, len(patterns))):
        a, b = alphabet[i*2 % alpha], alphabet[(i*2+1) % alpha]
        types.append(patterns[i](a, b, L))
    
    print(f"   类型示例: {types}")
    print(f"   每个类型的长度: {[len(t) for t in types]}")
    print(f"   每个类型的唯一字母数: {[len(set(t)) for t in types]}")
    
    try:
        X_train, X_test, y_train, y_test, tokens_train, tokens_test, labels_train, labels_test = \
            make_tokens(types, alpha, m, frac_train, letter_to_index, 
                       'Overlapping', 'Random', noise_level=0.0)
        
        print(f"   训练集大小: {len(tokens_train)} 序列")
        print(f"   测试集大小: {len(tokens_test)} 序列")
        print(f"   输入形状: {X_train.shape}")
        print(f"   训练序列示例: {[''.join(t) for t in tokens_train[:3]]}")
    except Exception as e:
        print(f"   ⚠️ 数据生成失败: {e}")
        return
    
    # ===== 3. 创建模型 =====
    print("\n3. 创建自编码器模型...")
    try:
        model = RNNAutoencoder(
            d_input=alpha,
            d_hidden=d_hidden,
            num_layers=1,
            d_latent=d_latent,
            sequence_length=L,
            nonlinearity='relu'
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   ✓ 模型创建成功")
        print(f"   总参数量: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        print(f"\n   模型结构:")
        print(f"   - 编码器: {alpha} → {d_hidden} → {d_latent}")
        print(f"   - 解码器: {d_latent} → {d_hidden} → {alpha}")
    except Exception as e:
        print(f"   ⚠️ 模型创建失败: {e}")
        return
    
    # ===== 4. 前向传播测试 =====
    print("\n4. 测试前向传播...")
    try:
        sample_input = X_train[:, :3, :]  # 取3个样本
        print(f"   输入形状: {sample_input.shape}")
        
        hidden, latent, reconstructed = model(sample_input)
        
        print(f"   ✓ 前向传播成功")
        print(f"   隐藏层形状: {hidden.shape}")
        print(f"   潜在层形状: {latent.shape}")
        print(f"   重建输出形状: {reconstructed.shape}")
        
        # 计算重建误差
        reconstruction_loss = F.mse_loss(reconstructed, sample_input).item()
        print(f"   初始重建误差 (MSE): {reconstruction_loss:.4f}")
        
    except Exception as e:
        print(f"   ⚠️ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ===== 5. 分析信息瓶颈 =====
    print("\n5. 分析信息瓶颈...")
    
    compression_ratio = (L * alpha) / d_latent
    print(f"   输入维度: {L} × {alpha} = {L * alpha}")
    print(f"   潜在维度: {d_latent}")
    print(f"   压缩比: {compression_ratio:.2f}x")
    
    if compression_ratio < 2:
        print("   ⚠️ 压缩比较低，可能导致欠拟合")
    elif compression_ratio > 10:
        print("   ⚠️ 压缩比很高，信息瓶颈可能太严重")
    else:
        print("   ✓ 压缩比适中")
    
    # ===== 6. 简单训练测试 =====
    print("\n6. 快速训练测试 (10个epoch)...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    test_losses = []
    
    for epoch in range(10):
        # 训练
        model.train()
        optimizer.zero_grad()
        _, _, output = model(X_train[:, :10, :])  # 只用10个样本快速测试
        train_loss = F.mse_loss(output, X_train[:, :10, :])
        train_loss.backward()
        optimizer.step()
        train_losses.append(train_loss.item())
        
        # 测试
        model.eval()
        with torch.no_grad():
            if X_test.shape[1] > 0:
                _, _, test_output = model(X_test[:, :5, :])
                test_loss = F.mse_loss(test_output, X_test[:, :5, :])
                test_losses.append(test_loss.item())
        
        if epoch % 3 == 0:
            test_loss_str = f"{test_losses[-1]:.4f}" if test_losses else "N/A"
            print(f"   Epoch {epoch}: Train Loss={train_losses[-1]:.4f}, "
                  f"Test Loss={test_loss_str}")
    
    # ===== 7. 诊断结论 =====
    print("\n" + "="*60)
    print("诊断结论")
    print("="*60)
    
    if len(train_losses) > 5:
        initial_loss = train_losses[0]
        final_loss = train_losses[-1]
        improvement = (initial_loss - final_loss) / initial_loss * 100
        
        print(f"\n训练改进: {improvement:.1f}%")
        
        if improvement < 10:
            print("⚠️ 问题: 训练几乎没有改进")
            print("   可能原因:")
            print("   1. 学习率过小")
            print("   2. 潜在维度太小（信息瓶颈）")
            print("   3. 网络初始化问题")
            
        elif improvement < 50:
            print("⚠️ 问题: 训练改进有限")
            print("   可能原因:")
            print("   1. 模型容量不足")
            print("   2. 训练时间不够")
            
        else:
            print("✓ 训练表现正常")
    
    # 泛化能力分析
    if test_losses and len(test_losses) > 0:
        train_test_gap = test_losses[-1] - train_losses[-1]
        print(f"\n训练-测试差距: {train_test_gap:.4f}")
        
        if train_test_gap > 0.5:
            print("⚠️ 问题: 泛化能力差")
            print("   可能原因:")
            print("   1. 过拟合训练数据")
            print("   2. 训练测试集分布差异大")
            print("   3. 模型记忆而非学习模式")
        elif train_test_gap > 0.1:
            print("⚠️ 存在一定的泛化问题")
        else:
            print("✓ 泛化能力良好")
    
    print("\n" + "="*60)
    print("建议的改进方向:")
    print("="*60)
    print("1. 调整潜在层维度 (当前: {})".format(d_latent))
    print("2. 尝试变分自编码器 (VAE) 架构")
    print("3. 添加正则化 (dropout, weight decay)")
    print("4. 使用更复杂的编码器/解码器")
    print("5. 增加训练数据或数据增强")
    
    return model, train_losses, test_losses


if __name__ == "__main__":
    diagnose_autoencoder()