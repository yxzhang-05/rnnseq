import functools
import os
import string
import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from model import LowRankLinear, RNNAutoencoder
from test import generate_instances, sequences_to_tensor, set_seed


CFG = dict(seed=45, L=4, m=2, alpha=6, frac_train=0.8, epochs=1000, lr=1e-3, weight_decay=1e-3, 
           d_hidden=12, d_latent=3, d_latent_hidden=6, num_layers=1, ranks=(1, 2, 3), heatmap_rank=3,
           save_dir=os.path.join('results', 'rank_perturbation'))


def train_rank_model(rank, X_train, X_test, cfg, device):
    layer_type = functools.partial(LowRankLinear, max_rank=rank)
    model = RNNAutoencoder(cfg['alpha'], cfg['d_hidden'], cfg['d_latent_hidden'], cfg['num_layers'], cfg['d_latent'], cfg['L'], layer_type=layer_type).to(device)
    model.latent.h2h = LowRankLinear(cfg['d_latent_hidden'], cfg['d_latent_hidden'], max_rank=rank, bias=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    test_acc_history = []
    pred_test_last, target_test_last = None, None
    for _ in range(cfg['epochs']):
        model.train()
        optimizer.zero_grad()
        _, _, output = model(X_train)
        target_train = torch.argmax(X_train, dim=-1)
        loss = F.cross_entropy(output.reshape(-1, output.shape[-1]), target_train.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            _, _, test_output = model(X_test)
            target_test = torch.argmax(X_test, dim=-1)
            pred_test = torch.argmax(test_output, dim=-1)
            test_seq_acc = (pred_test == target_test).all(dim=0).float().mean().item()
            test_acc_history.append(test_seq_acc)
            pred_test_last = pred_test.detach().cpu().numpy()
            target_test_last = target_test.detach().cpu().numpy()

    return model, np.asarray(test_acc_history, dtype=np.float64), pred_test_last, target_test_last


def _remove_sv1_from_lowrank_layer(layer):
    with torch.no_grad():
        w = layer.weight.detach().cpu().numpy().astype(np.float64)
        u, s, vh = np.linalg.svd(w, full_matrices=False)
        if s.shape[0] > 0:
            s[0] = 0.0
        r = layer.U.shape[0]
        sqrt_s = np.sqrt(np.maximum(s[:r], 0.0))
        v_new = (u[:, :r] * sqrt_s[None, :]).astype(np.float32)
        u_new = (sqrt_s[:, None] * vh[:r, :]).astype(np.float32)
        layer.V.copy_(torch.from_numpy(v_new).to(layer.V.device))
        layer.U.copy_(torch.from_numpy(u_new).to(layer.U.device))


def _remove_sv1_from_model_recurrent(model):
    _remove_sv1_from_lowrank_layer(model.encoder.rnn.h2h)
    _remove_sv1_from_lowrank_layer(model.latent.h2h)
    _remove_sv1_from_lowrank_layer(model.decoder.rnn.h2h)


def _remove_sv2_from_lowrank_layer(layer):
    with torch.no_grad():
        w = layer.weight.detach().cpu().numpy().astype(np.float64)
        u, s, vh = np.linalg.svd(w, full_matrices=False)
        if s.shape[0] > 1:
            s[1] = 0.0
        r = layer.U.shape[0]
        sqrt_s = np.sqrt(np.maximum(s[:r], 0.0))
        v_new = (u[:, :r] * sqrt_s[None, :]).astype(np.float32)
        u_new = (sqrt_s[:, None] * vh[:r, :]).astype(np.float32)
        layer.V.copy_(torch.from_numpy(v_new).to(layer.V.device))
        layer.U.copy_(torch.from_numpy(u_new).to(layer.U.device))


def _remove_sv2_from_model_recurrent(model):
    _remove_sv2_from_lowrank_layer(model.encoder.rnn.h2h)
    _remove_sv2_from_lowrank_layer(model.latent.h2h)
    _remove_sv2_from_lowrank_layer(model.decoder.rnn.h2h)


def _zero_weights_from_lowrank_layer(layer):
    """Set LowRankLinear weights to zero."""
    with torch.no_grad():
        layer.V.zero_()
        layer.U.zero_()


def _zero_weights_from_model_recurrent(model):
    _zero_weights_from_lowrank_layer(model.encoder.rnn.h2h)
    _zero_weights_from_lowrank_layer(model.latent.h2h)
    _zero_weights_from_lowrank_layer(model.decoder.rnn.h2h)


def _predict_test_tokens(model, X_test):
    model.eval()
    with torch.no_grad():
        _, _, test_output = model(X_test)
        target_test = torch.argmax(X_test, dim=-1)
        pred_test = torch.argmax(test_output, dim=-1)
    return pred_test.detach().cpu().numpy(), target_test.detach().cpu().numpy()


def compute_overall_accuracy(pred_test, target_test):
    """Compute sequence-level accuracy: all tokens match."""
    return float((pred_test == target_test).all(axis=0).mean())


def token_timestep_accuracy_matrix(pred_test, target_test, alpha):
    t_steps = target_test.shape[0]
    acc_mat = np.full((alpha, t_steps), np.nan, dtype=np.float64)
    for token_id in range(alpha):
        token_mask = target_test == token_id
        for t in range(t_steps):
            mask_t = token_mask[t]
            if int(mask_t.sum()) == 0:
                continue
            acc_mat[token_id, t] = float((pred_test[t, mask_t] == token_id).mean())
    return acc_mat


def plot_acc_vs_epoch(rank_histories, ranks, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    epochs_x = np.arange(1, len(next(iter(rank_histories.values()))) + 1)
    colors = ['#2563eb', '#f59e0b', '#16a34a']

    def smooth(y, window=15):
        y = np.asarray(y, dtype=np.float64)
        if y.size == 0:
            return y
        if y.size < window:
            return y
        out = np.empty_like(y)
        half = window // 2
        for i in range(y.size):
            lo = max(0, i - half)
            hi = min(y.size, i + half + 1)
            out[i] = y[lo:hi].mean()
        return out

    fig, ax = plt.subplots(figsize=(3.5, 3))
    for i, rank in enumerate(ranks):
        ax.plot(epochs_x, smooth(rank_histories[rank], window=15),
            color=colors[i % len(colors)], linewidth=2.0, label=f'rank {rank}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test sequence accuracy')
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path = os.path.join(save_dir, 'acc_vs_epoch_rank123.svg')
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_token_timestep_heatmap(acc_mat, alpha, L, save_dir, title='rank 3, remove SV1', filename='token_timestep_acc_heatmap_rank3_remove_sv1.svg'):
    os.makedirs(save_dir, exist_ok=True)
    token_labels = list(string.ascii_lowercase[:alpha])

    fig, ax = plt.subplots(1, 1, figsize=(3.4, 3.2))
    im = ax.imshow(acc_mat, aspect='auto', cmap='viridis', vmin=0.0, vmax=1.0, origin='lower')
    ax.set_title(title)
    ax.set_xlabel('Timestep')
    ax.set_xticks(np.arange(L))
    ax.set_xticklabels(np.arange(1, L + 1))
    ax.set_ylabel('Token')
    ax.set_yticks(np.arange(alpha))
    ax.set_yticklabels(token_labels)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Accuracy')
    fig.tight_layout()
    out_path = os.path.join(save_dir, filename)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def run_rank_perturbation_experiment(cfg=None):
    cfg = CFG if cfg is None else cfg
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(cfg['save_dir'], exist_ok=True)

    set_seed(cfg['seed'])
    seq_train, seq_test, _, _, _ = generate_instances(cfg['alpha'], cfg['L'], cfg['m'], frac_train=cfg['frac_train'])
    X_train = sequences_to_tensor(seq_train, cfg['alpha']).to(device)
    X_test = sequences_to_tensor(seq_test, cfg['alpha']).to(device)

    rank_histories, trained_models = {}, {}
    for rank in cfg['ranks']:
        set_seed(cfg['seed'] + rank)
        model, test_history, _, _ = train_rank_model(rank, X_train, X_test, cfg, device)
        trained_models[rank] = model
        rank_histories[rank] = test_history

    curve_path = plot_acc_vs_epoch(rank_histories, cfg['ranks'], cfg['save_dir'])
    heatmap_rank = int(cfg.get('heatmap_rank', 3))
    if heatmap_rank not in trained_models:
        raise ValueError(f'heatmap_rank={heatmap_rank} is not in ranks={cfg["ranks"]}.')
    
    print('\\n=== Overall Accuracy (rank {}) ==='.format(heatmap_rank))
    
    # perturbed_model_sv1 = copy.deepcopy(trained_models[heatmap_rank])
    # _remove_sv1_from_model_recurrent(perturbed_model_sv1)
    # pred_sv1_removed, target_sv1_removed = _predict_test_tokens(perturbed_model_sv1, X_test)
    # acc_sv1 = compute_overall_accuracy(pred_sv1_removed, target_sv1_removed)
    # print(f'rank {heatmap_rank} + remove SV1: {acc_sv1:.4f}')
    # heatmap_mat_sv1 = token_timestep_accuracy_matrix(pred_sv1_removed, target_sv1_removed, cfg['alpha'])
    # heatmap_path_sv1 = plot_token_timestep_heatmap(heatmap_mat_sv1, cfg['alpha'], cfg['L'], cfg['save_dir'],
    #     title=f'rank {heatmap_rank}, remove SV1', filename=f'token_timestep_acc_heatmap_rank{heatmap_rank}_remove_sv1.svg')
    
    # perturbed_model_sv2 = copy.deepcopy(trained_models[heatmap_rank])
    # _remove_sv2_from_model_recurrent(perturbed_model_sv2)
    # pred_sv2_removed, target_sv2_removed = _predict_test_tokens(perturbed_model_sv2, X_test)
    # acc_sv2 = compute_overall_accuracy(pred_sv2_removed, target_sv2_removed)
    # print(f'rank {heatmap_rank} + remove SV2: {acc_sv2:.4f}')
    # heatmap_mat_sv2 = token_timestep_accuracy_matrix(pred_sv2_removed, target_sv2_removed, cfg['alpha'])
    # heatmap_path_sv2 = plot_token_timestep_heatmap(heatmap_mat_sv2, cfg['alpha'], cfg['L'], cfg['save_dir'],
    #     title=f'rank {heatmap_rank}, remove SV2', filename=f'token_timestep_acc_heatmap_rank{heatmap_rank}_remove_sv2.svg')
    
    # perturbed_model_rank1_sv1 = copy.deepcopy(trained_models[1])
    # _remove_sv1_from_model_recurrent(perturbed_model_rank1_sv1)
    # pred_rank1_sv1, target_rank1_sv1 = _predict_test_tokens(perturbed_model_rank1_sv1, X_test)
    # acc_rank1_sv1 = compute_overall_accuracy(pred_rank1_sv1, target_rank1_sv1)
    # print(f'rank 1 + remove SV1: {acc_rank1_sv1:.4f}')
    # heatmap_mat_rank1_sv1 = token_timestep_accuracy_matrix(pred_rank1_sv1, target_rank1_sv1, cfg['alpha'])
    # heatmap_path_rank1_sv1 = plot_token_timestep_heatmap(heatmap_mat_rank1_sv1, cfg['alpha'], cfg['L'], cfg['save_dir'],
    #     title='rank 1, remove SV1', filename='token_timestep_acc_heatmap_rank1_remove_sv1.svg')
    
    perturbed_model_zero = copy.deepcopy(trained_models[heatmap_rank])
    _zero_weights_from_model_recurrent(perturbed_model_zero)
    pred_zero, target_zero = _predict_test_tokens(perturbed_model_zero, X_test)
    acc_zero = compute_overall_accuracy(pred_zero, target_zero)
    print(f'rank {heatmap_rank} + W=0: {acc_zero:.4f}')
    
    print(f'\\nSaved curve: {curve_path}')
    # print(f'Saved heatmap rank 3 SV1: {heatmap_path_sv1}')
    # print(f'Saved heatmap rank 3 SV2: {heatmap_path_sv2}')
    # print(f'Saved heatmap rank 1 SV1: {heatmap_path_rank1_sv1}')
    # return dict(curve_path=curve_path, heatmap_path_rank1_sv1=heatmap_path_rank1_sv1)


if __name__ == '__main__':
    run_rank_perturbation_experiment()