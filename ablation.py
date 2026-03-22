

# def generate_single_letter_instances(alpha):
#     letters = list(string.ascii_lowercase[:alpha])
#     seqs = np.asarray([[c] for c in letters])
#     labels = np.arange(alpha, dtype=np.int64)
#     return seqs, labels





# def token_train_full_test():
#     set_seed(seed)
#     model = RNNAutoencoder(alpha, d_hidden, d_latent_hidden, num_layers, d_latent, L).to(device)
#     # 训练集：每个token单独成一个序列
#     alphabet = list(string.ascii_lowercase[:alpha])
#     seq_train = np.array([[c] for c in alphabet])
#     X_train = sequences_to_tensor(seq_train, alpha).to(device)
#     train_labels = torch.arange(alpha, dtype=torch.long)
#     # 测试集：所有合法全长序列
#     _, seq_test, _, labels_test, _ = generate_instances(alpha, L, m, frac_train=0.0)
#     X_test = sequences_to_tensor(seq_test, alpha).to(device)
#     test_labels = torch.tensor(labels_test, dtype=torch.long)
#     print('\n=== Experiment: train on single letters, test on full sequences ===')
#     history = train(model, X_train, X_test, test_labels, train_labels=train_labels, types=None, n_epochs=epochs, lr=lr, weight_decay=weight_decay)
#     model.eval()
#     with torch.no_grad():
#         _, _, train_output = model(X_train)
#         pred_train = torch.argmax(train_output, dim=-1)
#         target_train = torch.argmax(X_train, dim=-1)
#         train_acc = (pred_train == target_train).all(dim=0).float().mean().item()
#         _, z_test, test_output = model(X_test)
#         pred_test = torch.argmax(test_output, dim=-1)
#         target_test = torch.argmax(X_test, dim=-1)
#         test_acc = (pred_test == target_test).all(dim=0).float().mean().item()
#     print(f'Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}')
#     return history, train_acc, test_acc

# def alpha4_train_rest2_test():
#     set_seed(seed)

#     alpha_train = 4
#     alpha_total = 6

#     # 用alpha=4生成训练集
#     seq_train, _, labels_train, _, types = generate_instances(alpha_train, L, m, frac_train=1.0)
#     X_train = sequences_to_tensor(seq_train, alpha_total).to(device)
#     train_labels = torch.tensor(labels_train, dtype=torch.long)

#     # 剩下两个字母
#     all_letters = list(string.ascii_lowercase[:alpha_total])
#     test_letters = all_letters[alpha_train:alpha_total]
#     # 构造所有可能的测试序列
#     test_seqs = [''.join(seq) for seq in itertools.product(test_letters, repeat=L)]
#     X_test = sequences_to_tensor(test_seqs, alpha_total).to(device)
#     test_labels = torch.zeros(len(test_seqs), dtype=torch.long)  # dummy

#     model = RNNAutoencoder(alpha_total, d_hidden, d_latent_hidden, num_layers, d_latent, L).to(device)
#     print('\n=== Experiment: train on alpha=4, test on rest 2 ===')
#     history = train(model, X_train, X_test, test_labels, train_labels=train_labels,
#         types=None, n_epochs=epochs, lr=lr, weight_decay=weight_decay, print_final=False)
#     return history

# def plot_two_test_acc_curves(hist1, hist2, label1, label2, save_path):
#     def smooth_curve(y, window=15):
#         y = np.array(y)
#         if len(y) < window:
#             return y
#         return np.convolve(y, np.ones(window)/window, mode='valid')

#     plt.figure(figsize=(3.2, 3))
#     y1 = smooth_curve(hist1['test_acc'], window=20)
#     y2 = smooth_curve(hist2['test_acc'], window=20)
#     x1 = np.arange(len(y1)) + (len(hist1['test_acc']) - len(y1))
#     x2 = np.arange(len(y2)) + (len(hist2['test_acc']) - len(y2))
#     plt.plot(x1, y1, label=label1, lw=2, color="#17761C")  
#     plt.plot(x2, y2, label=label2, lw=2, color="#243c91")  
#     plt.xlabel('Epoch')
#     plt.ylabel('Test acc')
#     plt.legend()
#     plt.tight_layout()
#     base, _ = os.path.splitext(save_path)
#     plt.savefig(base + '.svg')
#     plt.close()

# def run_experiment():
#     set_seed(seed)
#     model = RNNAutoencoder(alpha, d_hidden, d_latent_hidden, num_layers, d_latent, L).to(device)
#     seq_train, seq_test, labels_train, labels_test, types = generate_instances(alpha, L, m, frac_train=0.8)
#     X_train = sequences_to_tensor(seq_train, alpha).to(device)
#     X_test = sequences_to_tensor(seq_test, alpha).to(device)
#     train_labels = torch.tensor(labels_train, dtype=torch.long)
#     test_labels = torch.tensor(labels_test, dtype=torch.long)
    
   
#     # # 实验1：单token训练，全序列测试
#     # hist1, _, _ = token_train_full_test()
#     # # 实验2：alpha=4训练，剩余2字母全序列测试
#     # hist2 = alpha4_train_rest2_test()
#     # plot_two_test_acc_curves(
#     #     hist1, hist2,
#     #     label1='Single token train, full seq test',
#     #     label2='Alpha=4 train, rest2 test',
#     #     save_path=os.path.join(SAVE_DIR, 'test_acc_compare.svg')
    
#     # # Minimal Ablation & Decoding Experiments
#     # print('\n=== Minimal Ablation & Decoding Experiments (on full train/test) ===')
#     # train(model, X_train, X_test, test_labels, train_labels=train_labels, types=types, n_epochs=epochs, lr=lr, weight_decay=weight_decay, print_final=True)
#     # linear_probe_token_decoding(model, X_train, X_test, seq_train, seq_test, save_dir=SAVE_DIR)
#     # input_ablation(model, X_test, save_dir=SAVE_DIR)
#     # history_ablation(model, X_test, save_dir=SAVE_DIR)
#     # whh_ablation(model, X_test, save_dir=SAVE_DIR)

#     # PCA分析
#     pca_latent_residual(model, X_train, seq_train, save_dir=SAVE_DIR)



