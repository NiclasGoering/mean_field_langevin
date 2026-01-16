#!/usr/bin/env python3
# parity_microscope_h100.py
# ------------------------------------------------------------
# "Microscope" instrumentation for k-sparse parity learning in a 1-hidden-layer ReLU net
# trained with the EXACT LangevinGD (SGLD) update you pasted.
#
# What this script gives you:
# 1) Per-neuron "alignment with support S" over time (init -> final) and correlation init vs final.
# 2) Per-neuron decomposition of the w-gradient into:
#      - teacher term  (coupling to -y)
#      - self term     (coupling to its own contribution to f)
#      - others term   (coupling to all other neurons' contributions)
#    and how each term contributes to the drift of log(u/v) where u=||w_S||^2, v=||w_off||^2.
# 3) A "multiplicative feedback" metric:
#      - corr_j(|w_ij|, |grad_ij|) within each neuron (positive feedback / rich-get-richer)
#      - gate-residual concentration index: E[g r^2] / (E[g] E[r^2])
#      - gate switchiness proxy via smoothed gate derivative
# 4) Huge neuron-by-time heatmaps (N=512) for selected quantities + init/final weight heatmaps.
#
# Defaults are set for H100-ish runs but are configurable via args.
# ------------------------------------------------------------

import os
import json
import math
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# CUDA graph marker (keep exact style)
# -----------------------------
def _cudagraph_mark_step_begin():
    try:
        torch.compiler.cudagraph_mark_step_begin()
    except Exception:
        pass


# -----------------------------
# Data: k-sparse parity
# -----------------------------
def generate_k_sparse_parity_data(P, d, k, device='cpu'):
    """
    X in {-1,+1}^d, y = product of first k features.
    """
    if k > d:
        raise ValueError("k cannot be greater than d.")
    X = torch.randint(2, size=(P, d), device=device, dtype=torch.float32) * 2 - 1
    y = torch.prod(X[:, :k], dim=1, keepdim=True)
    return X, y


# -----------------------------
# Model (same as yours)
# -----------------------------
class TwoLayerNet(nn.Module):
    """
    f(x) = (phi(xW) a) / N^gamma
    w shape: (d,N), a shape (N,1)
    """
    def __init__(self, d, N, g_w, g_a, gamma_scaling_exponent: float, activation='relu'):
        super().__init__()
        self.d = d
        self.N = N
        self.gamma = float(gamma_scaling_exponent)

        sigma_w_sq = float(g_w) / d
        sigma_a_sq = float(g_a)
        self.sigma_w = math.sqrt(sigma_w_sq)
        self.sigma_a = math.sqrt(sigma_a_sq)

        self.w = nn.Parameter(torch.randn(d, N) * self.sigma_w)
        self.a = nn.Parameter(torch.randn(N, 1) * self.sigma_a)

        if activation == 'relu':
            self.phi = F.relu
        elif activation == 'sigmoid':
            self.phi = torch.sigmoid
        else:
            raise ValueError("activation must be relu or sigmoid")

    def forward(self, x):
        return (self.phi(x @ self.w) @ self.a) / (self.N ** self.gamma)

    def homogeneity_loss(self):
        w_abs = self.w.abs()
        var_per_neuron = torch.var(w_abs, dim=0, unbiased=False)
        return var_per_neuron.sum()


# -----------------------------
# LR schedule helper (same style)
# -----------------------------
def poly_decay_lr(epoch: int, eta_start: float, eta_final: float,
                  decay_steps: int, power: float = 2.0) -> float:
    if decay_steps <= 0:
        return eta_final
    tau = min(1.0, epoch / float(decay_steps))
    return eta_final + (eta_start - eta_final) * (1.0 - tau) ** power


# -----------------------------
# EXACT LangevinGD from your snippet
# -----------------------------
class LangevinGD(torch.optim.Optimizer):
    """
    Param groups must include 'sigma_sq' in each group.
    Update: p <- p - lr * ( (T/sigma_sq) * p + grad ) + sqrt(2*T*lr) * N(0,I)
    """
    def __init__(self, params, lr, T):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, T=T)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group['lr'])
            T = float(group['T'])
            sigma_sq = group.get('sigma_sq', None)
            if sigma_sq is None:
                raise ValueError("Each param group must have 'sigma_sq' set.")

            ps = [p for p in group['params'] if p.grad is not None]
            if not ps:
                continue

            decay_coeff = T / float(sigma_sq)
            decays = [p.mul(decay_coeff) for p in ps]
            drift_updates = [-(lr) * (d + p.grad) for p, d in zip(ps, decays)]

            noise_std = math.sqrt(2.0 * T * lr)
            noises = [torch.randn_like(p, dtype=torch.float32).mul_(noise_std).to(p.dtype) for p in ps]

            updates = [du + nz for du, nz in zip(drift_updates, noises)]
            torch._foreach_add_(ps, updates)

        return loss


# -----------------------------
# Utility: rank correlation (Spearman) without scipy
# -----------------------------
def spearman_corr_torch(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    x,y: (d,) float tensors on GPU ok.
    """
    rx = torch.argsort(torch.argsort(x))
    ry = torch.argsort(torch.argsort(y))
    rx = rx.to(torch.float32)
    ry = ry.to(torch.float32)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = (rx.std(unbiased=False) * ry.std(unbiased=False)) + eps
    return (rx * ry).mean() / denom


# -----------------------------
# Microscope logger using memmap (keeps RAM sane)
# -----------------------------
class MemmapLogger:
    def __init__(self, out_dir: Path, n_steps: int, N: int):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir
        self.n_steps = n_steps
        self.N = N
        self.idx = 0

        # time axis
        self.epoch = np.memmap(out_dir / "epoch_i64.dat", mode="w+", dtype=np.int64, shape=(n_steps,))

        # Per-neuron time series (store float16; enough for plotting)
        def mm(name):
            return np.memmap(out_dir / f"{name}_f16.dat", mode="w+", dtype=np.float16, shape=(n_steps, N))

        # Alignment / support discovery
        self.log_uv = mm("log_uv")                    # log(||w_S||^2 / ||w_off||^2)
        self.align_energy = mm("align_energy")        # ||w_S||^2 / (||w||^2)
        self.A_on_abs = mm("A_on_abs")                # per-neuron mean_{j in S} |A_ij|
        self.A_off_abs = mm("A_off_abs")
        self.A_ratio = mm("A_ratio")                  # A_on_abs / A_off_abs
        self.gate_p = mm("gate_p")                    # E[g]
        self.gate_entropy = mm("gate_entropy")        # per-neuron entropy of gate
        self.gate_y_corr = mm("gate_y_corr")          # E[y g] (simple gate-label coupling)

        # Gradient decomposition norms (full-batch instantaneous)
        self.grad_norm_total = mm("grad_norm_total")
        self.grad_norm_teacher = mm("grad_norm_teacher")
        self.grad_norm_self = mm("grad_norm_self")
        self.grad_norm_others = mm("grad_norm_others")

        # Drift contributions to log_uv (approx one-step contribution from each term)
        self.dloguv_teacher = mm("dloguv_teacher")
        self.dloguv_self = mm("dloguv_self")
        self.dloguv_others = mm("dloguv_others")

        # Multiplicative feedback metrics
        self.corr_w_grad = mm("corr_w_grad")          # Spearman corr within neuron: |w_j| vs |grad_j|
        self.gate_resid_index = mm("gate_resid_index")# E[g r^2] / (E[g]E[r^2])
        self.gate_switchiness = mm("gate_switchiness")# E[sig'(z/tau)] proxy

    def write(self, epoch: int, data: dict):
        i = self.idx
        if i >= self.n_steps:
            return
        self.epoch[i] = int(epoch)
        for k, v in data.items():
            arr = getattr(self, k)
            arr[i, :] = v.astype(np.float16, copy=False)
        self.idx += 1

    def flush(self):
        for name, v in self.__dict__.items():
            if isinstance(v, np.memmap):
                v.flush()


# -----------------------------
# Core microscope computation (per-neuron, full-batch)
# -----------------------------
@torch.no_grad()
def compute_neuron_microscope(
    model: TwoLayerNet,
    X: torch.Tensor,
    y: torch.Tensor,
    support_idx: torch.Tensor,
    activation: str,
    eta_now: float,
    T: float,
    tau_switch: float = 1.0,
    eps: float = 1e-12
):
    """
    Returns a dict of numpy arrays (N,) for the memmap logger.

    IMPORTANT: This does NOT change training; it just measures.
    """
    device = X.device
    P, d = X.shape
    N = model.N

    w = model.w.float()        # (d,N)
    a = model.a.float()        # (N,1)
    gamma = model.gamma

    # mask
    mask = torch.zeros(d, dtype=torch.bool, device=device)
    mask[support_idx] = True
    not_mask = ~mask

    # forward pieces
    z = (X @ w).float()  # (P,N)

    if activation == "relu":
        g = (z > 0).to(torch.float32)      # (P,N)
        phi = F.relu(z)                    # (P,N)
    elif activation == "sigmoid":
        sig = torch.sigmoid(z)
        g = (sig * (1.0 - sig)).to(torch.float32)
        phi = sig
    else:
        raise ValueError("activation must be relu or sigmoid")

    f = (phi @ a) / (N ** gamma)     # (P,1)
    r = f - y.float()                # (P,1)

    # --- alignment in weight-space (support energy)
    wS = w[mask, :]        # (k,N)
    wO = w[not_mask, :]    # (d-k,N)
    u = (wS * wS).sum(dim=0)                 # (N,)
    v = (wO * wO).sum(dim=0)                 # (N,)
    log_uv = torch.log((u + eps) / (v + eps))
    align_energy = u / (u + v + eps)

    # --- A_{i,j} = E[g_i * y * x_j], per neuron on/off abs means + ratio
    # Compute A as (N,d) to then take per-neuron means over j in S / off
    A = (g.T @ (y.float() * X.float())) / float(P)    # (N,d)
    A_on_abs = A[:, mask].abs().mean(dim=1)
    A_off_abs = A[:, not_mask].abs().mean(dim=1)
    A_ratio = A_on_abs / (A_off_abs + eps)

    # --- gate stats
    p_i = g.mean(dim=0)  # (N,)
    p_clamped = torch.clamp(p_i, min=eps, max=1.0 - eps)
    gate_entropy = -(p_clamped * torch.log(p_clamped) + (1.0 - p_clamped) * torch.log(1.0 - p_clamped))
    gate_y_corr = (g * y.float()).mean(dim=0)  # (N,)

    # --- Decompose residual into:
    # f = sum_l a_l phi_l / N^gamma
    # r = f - y = (others_only_i) + (self_i) - y
    # self_mat[:,i] = a_i phi_i / N^gamma
    self_mat = (phi * a.view(1, N)) / (N ** gamma)          # (P,N)
    f_rep = f.repeat(1, N)                                   # (P,N)
    others_only_mat = f_rep - self_mat                       # (P,N)

    # gradient components for w:
    # G_teacher = a_i * E[ g_i * (-y) * x ]
    # G_self    = a_i * E[ g_i * self_i * x ]
    # G_others  = a_i * E[ g_i * others_only_i * x ]
    # G_total   = a_i * E[ g_i * r * x ]
    teacher_scalar = (-y.float())                             # (P,1)
    WX_teacher = teacher_scalar * X.float()                   # (P,d)
    G_teacher_pre = (g.T @ WX_teacher) / float(P)             # (N,d)
    G_teacher = G_teacher_pre * a.view(N, 1)                  # (N,d)

    scalar_self = g * self_mat                                # (P,N)
    G_self_pre = (scalar_self.T @ X.float()) / float(P)       # (N,d)
    G_self = G_self_pre * a.view(N, 1)

    scalar_others = g * others_only_mat                        # (P,N)
    G_others_pre = (scalar_others.T @ X.float()) / float(P)    # (N,d)
    G_others = G_others_pre * a.view(N, 1)

    scalar_total = g * r.float()                               # (P,N) via broadcast r (P,1)
    G_total_pre = (scalar_total.T @ X.float()) / float(P)      # (N,d)
    G_total = G_total_pre * a.view(N, 1)

    # gradient norms (per neuron)
    grad_norm_teacher = torch.linalg.norm(G_teacher, dim=1)
    grad_norm_self = torch.linalg.norm(G_self, dim=1)
    grad_norm_others = torch.linalg.norm(G_others, dim=1)
    grad_norm_total = torch.linalg.norm(G_total, dim=1)

    # --- Drift contributions to log_uv (one-step approx) from each component
    # Δw_comp = -eta_now * (G_comp^T) in w-coordinates (d,N)
    # Δu = 2 Σ_{j in S} w_{j,i} Δw_{j,i}, Δv similarly
    def dloguv_from_G(G_comp_Nd: torch.Tensor) -> torch.Tensor:
        # G_comp_Nd shape (N,d) -> transpose to (d,N) to match w
        GdN = G_comp_Nd.T  # (d,N)
        dw = -eta_now * GdN
        du = 2.0 * (w[mask, :] * dw[mask, :]).sum(dim=0)
        dv = 2.0 * (w[not_mask, :] * dw[not_mask, :]).sum(dim=0)
        return du / (u + eps) - dv / (v + eps)

    dloguv_teacher = dloguv_from_G(G_teacher)
    dloguv_self = dloguv_from_G(G_self)
    dloguv_others = dloguv_from_G(G_others)

    # --- Multiplicative feedback
    # (a) "rich-get-richer": within each neuron, do large |w_j| get large |grad_j|?
    corr_list = []
    abs_w = w.abs()        # (d,N)
    abs_gt = G_total.abs().T  # (d,N)
    for i in range(N):
        corr_list.append(spearman_corr_torch(abs_w[:, i], abs_gt[:, i]).clamp(-1.0, 1.0))
    corr_w_grad = torch.stack(corr_list, dim=0)

    # (b) residual energy concentrated on gated samples:
    r2 = (r.float().squeeze(1) ** 2)                   # (P,)
    Eg_r2 = r2.mean()
    Eg = p_i
    Eg_gr2 = (g * r2.view(P, 1)).mean(dim=0)
    gate_resid_index = Eg_gr2 / (Eg * Eg_r2 + eps)

    # (c) gate "switchiness" via smoothed gate derivative (proxy for how gating can flip)
    # g_tilde = sigmoid(z/tau); switchiness = E[g_tilde(1-g_tilde)]/tau
    z_scaled = z / float(tau_switch)
    g_tilde = torch.sigmoid(z_scaled)
    switchiness = (g_tilde * (1.0 - g_tilde)).mean(dim=0) / float(tau_switch)

    # pack to cpu numpy float32
    out = {
        "log_uv": log_uv.detach().cpu().numpy().astype(np.float32),
        "align_energy": align_energy.detach().cpu().numpy().astype(np.float32),
        "A_on_abs": A_on_abs.detach().cpu().numpy().astype(np.float32),
        "A_off_abs": A_off_abs.detach().cpu().numpy().astype(np.float32),
        "A_ratio": A_ratio.detach().cpu().numpy().astype(np.float32),
        "gate_p": p_i.detach().cpu().numpy().astype(np.float32),
        "gate_entropy": gate_entropy.detach().cpu().numpy().astype(np.float32),
        "gate_y_corr": gate_y_corr.detach().cpu().numpy().astype(np.float32),
        "grad_norm_total": grad_norm_total.detach().cpu().numpy().astype(np.float32),
        "grad_norm_teacher": grad_norm_teacher.detach().cpu().numpy().astype(np.float32),
        "grad_norm_self": grad_norm_self.detach().cpu().numpy().astype(np.float32),
        "grad_norm_others": grad_norm_others.detach().cpu().numpy().astype(np.float32),
        "dloguv_teacher": dloguv_teacher.detach().cpu().numpy().astype(np.float32),
        "dloguv_self": dloguv_self.detach().cpu().numpy().astype(np.float32),
        "dloguv_others": dloguv_others.detach().cpu().numpy().astype(np.float32),
        "corr_w_grad": corr_w_grad.detach().cpu().numpy().astype(np.float32),
        "gate_resid_index": gate_resid_index.detach().cpu().numpy().astype(np.float32),
        "gate_switchiness": switchiness.detach().cpu().numpy().astype(np.float32),
    }
    return out


# -----------------------------
# Plot helpers
# -----------------------------
def robust_vmin_vmax(arr2d: np.ndarray, lo=1, hi=99):
    vmin = np.percentile(arr2d, lo)
    vmax = np.percentile(arr2d, hi)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(arr2d))
        vmax = float(np.nanmax(arr2d))
        if vmin == vmax:
            vmin -= 1.0
            vmax += 1.0
    return vmin, vmax


def plot_heatmap(out_path: Path, H: np.ndarray, t: np.ndarray, title: str, ylabel="neuron (sorted)", cmap="viridis"):
    """
    H: (T,N) or (T,N) float
    We'll plot as neurons on Y (N) and time on X (T).
    """
    # Expect H as (T,N). We plot imshow with shape (N,T).
    HnT = H.T
    vmin, vmax = robust_vmin_vmax(HnT)
    plt.figure(figsize=(14, 6))
    plt.imshow(HnT, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("time index (diag step)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_weight_heatmap(out_path: Path, W: np.ndarray, title: str, support_k: int):
    """
    W: (d,N) float
    """
    vmin, vmax = robust_vmin_vmax(W, lo=1, hi=99)
    plt.figure(figsize=(14, 5))
    plt.imshow(W, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="coolwarm")
    plt.colorbar()
    plt.title(title)
    plt.ylabel("input coord j")
    plt.xlabel("neuron i")
    # support boundary line
    plt.axhline(support_k - 0.5, color="black", linewidth=1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def scatter_init_final(out_path: Path, init_vec: np.ndarray, final_vec: np.ndarray, title: str, xlabel="init", ylabel="final"):
    mask = np.isfinite(init_vec) & np.isfinite(final_vec)
    x = init_vec[mask]
    y = final_vec[mask]
    if x.size < 2:
        return
    corr = np.corrcoef(x, y)[0, 1]
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=10, alpha=0.6)
    plt.title(f"{title}\nPearson r={corr:.3f}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# Training loop (keeps your structure; adds microscope snapshots)
# -----------------------------
def train_with_microscope(
    out_dir: Path,
    d: int,
    k: int,
    P_train: int,
    P_test: int,
    N: int,
    g_w: float,
    g_a: float,
    gamma: float,
    kappa_0: float,
    eta_start: float,
    eta_final: float,
    lr_decay_steps: int,
    lr_power: float,
    epochs: int,
    log_interval: int,
    diag_interval: int,
    activation: str,
    homogeneity_penalty_weight: float = 0.0,
    tau_switch: float = 1.0,
    seed: int = 12345,
    compile_model: bool = True,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    # data
    X_train, y_train = generate_k_sparse_parity_data(P_train, d, k, device=device)
    X_test, y_test = generate_k_sparse_parity_data(P_test, d, k, device=device)
    support_idx = torch.arange(k, device=device)

    # model
    model = TwoLayerNet(d=d, N=N, g_w=g_w, g_a=g_a, gamma_scaling_exponent=gamma, activation=activation).to(device)
    if compile_model:
        try:
            model = torch.compile(model, mode="max-autotune")
        except Exception:
            pass

    # exact temperature mapping
    kappa = float(kappa_0)
    T = 2.0 * (kappa ** 2)

    # optimizer
    sigma_a = float(model.sigma_a)
    sigma_w = float(model.sigma_w)
    optimizer = LangevinGD(
        params=[
            {'params': [model.a], 'sigma_sq': sigma_a ** 2},
            {'params': [model.w], 'sigma_sq': sigma_w ** 2},
        ],
        lr=eta_start,
        T=T,
    )
    loss_fn = nn.MSELoss(reduction="mean")

    # microscope logger
    n_diag_steps = (epochs // diag_interval) + 1
    mm_dir = out_dir / "microscope_memmap"
    logger = MemmapLogger(mm_dir, n_diag_steps, N)

    # save init weights (for heatmap)
    with torch.no_grad():
        W_init = model.w.detach().float().cpu().numpy().copy()

    metrics_rows = []
    t0 = time.time()

    # ---- init microscope snapshot
    with torch.no_grad():
        snap = compute_neuron_microscope(
            model=model, X=X_train, y=y_train,
            support_idx=support_idx, activation=activation,
            eta_now=float(eta_start), T=T, tau_switch=tau_switch
        )
    logger.write(epoch=0, data=snap)

    for epoch in range(epochs + 1):
        # LR schedule
        lr_t = poly_decay_lr(epoch, eta_start, eta_final, lr_decay_steps, lr_power)
        for g in optimizer.param_groups:
            g['lr'] = float(lr_t)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", dtype=torch.bfloat16):
            _cudagraph_mark_step_begin()
            y_pred = model(X_train)
            base_loss = loss_fn(y_pred, y_train)
            if homogeneity_penalty_weight > 0.0:
                reg = homogeneity_penalty_weight * model.homogeneity_loss()
                loss = base_loss + reg
            else:
                loss = base_loss

        loss.backward()
        optimizer.step()

        # logging (train/test)
        if epoch % log_interval == 0:
            model.eval()
            with torch.no_grad():
                with autocast("cuda", dtype=torch.bfloat16):
                    _cudagraph_mark_step_begin()
                    y_pred_tr = model(X_train).clone()
                    _cudagraph_mark_step_begin()
                    y_pred_te = model(X_test).clone()
                train_mse = loss_fn(y_pred_tr, y_train).item()
                test_mse = loss_fn(y_pred_te, y_test).item()
                train_err = (torch.sign(y_pred_tr) != y_train).float().mean().item()
                test_err = (torch.sign(y_pred_te) != y_test).float().mean().item()
            metrics_rows.append({
                "epoch": int(epoch),
                "lr": float(lr_t),
                "train_mse": float(train_mse),
                "test_mse": float(test_mse),
                "train_error_01": float(train_err),
                "test_error_01": float(test_err),
            })
            dt = time.time() - t0
            print(f"[{epoch:>10}] lr={lr_t:.3e} train_err={train_err:.4f} test_err={test_err:.4f} "
                  f"train_mse={train_mse:.4e} test_mse={test_mse:.4e}  t={dt/60:.1f}m")

        # microscope snapshots
        if diag_interval > 0 and epoch % diag_interval == 0 and epoch > 0:
            model.eval()
            with torch.no_grad():
                snap = compute_neuron_microscope(
                    model=model, X=X_train, y=y_train,
                    support_idx=support_idx, activation=activation,
                    eta_now=float(lr_t), T=T, tau_switch=tau_switch
                )
            logger.write(epoch=epoch, data=snap)

    logger.flush()

    # final weights
    with torch.no_grad():
        W_final = model.w.detach().float().cpu().numpy().copy()

    # save metrics CSV + config
    with open(out_dir / "config.json", "w") as f:
        json.dump({
            "d": d, "k": k, "P_train": P_train, "P_test": P_test,
            "N": N, "g_w": g_w, "g_a": g_a, "gamma": gamma,
            "kappa_0": kappa_0, "T": T,
            "eta_start": eta_start, "eta_final": eta_final,
            "lr_decay_steps": lr_decay_steps, "lr_power": lr_power,
            "epochs": epochs, "log_interval": log_interval, "diag_interval": diag_interval,
            "activation": activation,
            "homogeneity_penalty_weight": homogeneity_penalty_weight,
            "tau_switch": tau_switch,
            "seed": seed,
        }, f, indent=2)

    # save metrics
    import csv
    if metrics_rows:
        with open(out_dir / "metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
            w.writeheader()
            w.writerows(metrics_rows)

    # save weights
    np.save(out_dir / "W_init.npy", W_init)
    np.save(out_dir / "W_final.npy", W_final)

    # produce plots
    make_microscope_plots(out_dir, W_init, W_final, k)

    return out_dir


# -----------------------------
# Plotting from memmaps
# -----------------------------
def load_memmap_series(mm_dir: Path, n_steps: int, N: int, name: str) -> np.ndarray:
    return np.memmap(mm_dir / f"{name}_f16.dat", mode="r", dtype=np.float16, shape=(n_steps, N)).astype(np.float32)


def make_microscope_plots(out_dir: Path, W_init: np.ndarray, W_final: np.ndarray, k: int):
    mm_dir = out_dir / "microscope_memmap"
    epoch_path = mm_dir / "epoch_i64.dat"
    if not epoch_path.exists():
        print("No memmaps found; skipping plots.")
        return

    # infer shapes
    # We stored epoch as (n_steps,), so infer n_steps from file size
    epoch_mm = np.memmap(epoch_path, mode="r", dtype=np.int64)
    n_steps = epoch_mm.shape[0]

    # N inferred from any f16 file
    # (pick log_uv)
    log_uv_mm = np.memmap(mm_dir / "log_uv_f16.dat", mode="r", dtype=np.float16)
    # log_uv stored as (n_steps,N)
    N = int(log_uv_mm.size // n_steps)

    epoch = np.memmap(epoch_path, mode="r", dtype=np.int64, shape=(n_steps,)).copy()

    # load a few series
    log_uv = load_memmap_series(mm_dir, n_steps, N, "log_uv")
    A_ratio = load_memmap_series(mm_dir, n_steps, N, "A_ratio")
    gate_p = load_memmap_series(mm_dir, n_steps, N, "gate_p")
    gate_entropy = load_memmap_series(mm_dir, n_steps, N, "gate_entropy")
    grad_norm_teacher = load_memmap_series(mm_dir, n_steps, N, "grad_norm_teacher")
    grad_norm_self = load_memmap_series(mm_dir, n_steps, N, "grad_norm_self")
    grad_norm_others = load_memmap_series(mm_dir, n_steps, N, "grad_norm_others")
    dloguv_teacher = load_memmap_series(mm_dir, n_steps, N, "dloguv_teacher")
    dloguv_self = load_memmap_series(mm_dir, n_steps, N, "dloguv_self")
    dloguv_others = load_memmap_series(mm_dir, n_steps, N, "dloguv_others")
    corr_w_grad = load_memmap_series(mm_dir, n_steps, N, "corr_w_grad")
    gate_resid_index = load_memmap_series(mm_dir, n_steps, N, "gate_resid_index")
    gate_switchiness = load_memmap_series(mm_dir, n_steps, N, "gate_switchiness")

    # sort neurons by FINAL log_uv (most support-aligned on top)
    final_loguv = log_uv[-1, :]
    sort_idx = np.argsort(final_loguv)  # increasing
    # reverse to have most aligned at top of heatmap
    sort_idx = sort_idx[::-1]

    def sortH(H):
        return H[:, sort_idx]

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    # weight heatmaps (sorted)
    plot_weight_heatmap(plots_dir / "W_init_heatmap.png", W_init[:, sort_idx], "W init (sorted by final log_uv)", k)
    plot_weight_heatmap(plots_dir / "W_final_heatmap.png", W_final[:, sort_idx], "W final (sorted by final log_uv)", k)

    # neuron-by-time heatmaps (sorted)
    plot_heatmap(plots_dir / "log_uv_heatmap.png", sortH(log_uv), epoch, "log(||w_S||^2 / ||w_off||^2) over time")
    plot_heatmap(plots_dir / "A_ratio_heatmap.png", sortH(A_ratio), epoch, "A_ratio per neuron over time (A_on_abs/A_off_abs)")
    plot_heatmap(plots_dir / "gate_p_heatmap.png", sortH(gate_p), epoch, "gate open fraction p_i=E[g] over time")
    plot_heatmap(plots_dir / "gate_entropy_heatmap.png", sortH(gate_entropy), epoch, "gate entropy over time")

    plot_heatmap(plots_dir / "grad_teacher_heatmap.png", sortH(grad_norm_teacher), epoch, "||grad_teacher|| per neuron")
    plot_heatmap(plots_dir / "grad_self_heatmap.png", sortH(grad_norm_self), epoch, "||grad_self|| per neuron")
    plot_heatmap(plots_dir / "grad_others_heatmap.png", sortH(grad_norm_others), epoch, "||grad_others|| per neuron")

    plot_heatmap(plots_dir / "dloguv_teacher_heatmap.png", sortH(dloguv_teacher), epoch, "Δlog_uv teacher contribution (one-step approx)")
    plot_heatmap(plots_dir / "dloguv_self_heatmap.png", sortH(dloguv_self), epoch, "Δlog_uv self contribution (one-step approx)")
    plot_heatmap(plots_dir / "dloguv_others_heatmap.png", sortH(dloguv_others), epoch, "Δlog_uv others contribution (one-step approx)")

    plot_heatmap(plots_dir / "corr_w_grad_heatmap.png", sortH(corr_w_grad), epoch, "Spearman corr(|w|,|grad|) within neuron (multiplicative feedback)")
    plot_heatmap(plots_dir / "gate_resid_index_heatmap.png", sortH(gate_resid_index), epoch, "gate-residual concentration index E[g r^2]/(E[g]E[r^2])")
    plot_heatmap(plots_dir / "gate_switchiness_heatmap.png", sortH(gate_switchiness), epoch, "gate switchiness proxy E[sig'(z/tau)]/tau")

    # init->final correlations across neurons (the hypothesis you asked for)
    init_loguv = log_uv[0, :]
    final_loguv = log_uv[-1, :]
    init_Ar = A_ratio[0, :]
    final_Ar = A_ratio[-1, :]

    scatter_init_final(plots_dir / "init_vs_final_loguv.png", init_loguv, final_loguv,
                       "Neuronwise: init vs final log_uv", xlabel="init log_uv", ylabel="final log_uv")
    scatter_init_final(plots_dir / "init_vs_final_A_ratio.png", init_Ar, final_Ar,
                       "Neuronwise: init vs final A_ratio", xlabel="init A_ratio", ylabel="final A_ratio")

    # Also: does init log_uv predict final A_ratio?
    scatter_init_final(plots_dir / "init_loguv_vs_final_A_ratio.png", init_loguv, final_Ar,
                       "Neuronwise: init log_uv vs final A_ratio", xlabel="init log_uv", ylabel="final A_ratio")

    # quick summary curves (mean across neurons)
    def mean_curve(H): return np.nanmean(H, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(epoch, mean_curve(log_uv), label="mean log_uv")
    plt.plot(epoch, mean_curve(A_ratio), label="mean A_ratio")
    plt.plot(epoch, mean_curve(gate_entropy), label="mean gate_entropy")
    plt.legend()
    plt.xlabel("epoch (diag snapshots)")
    plt.ylabel("mean over neurons")
    plt.title("Summary: mean neuron metrics vs time")
    plt.tight_layout()
    plt.savefig(plots_dir / "summary_means.png", dpi=200)
    plt.close()

    print(f"[plots] wrote to: {plots_dir}")


# -----------------------------
# Main / args
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="./microscope_run_h100_sig")
    ap.add_argument("--d", type=int, default=35)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--P_train", type=int, default=10_000)     # as requested
    ap.add_argument("--P_test", type=int, default=100_000)
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--g_w", type=float, default=1.0)
    ap.add_argument("--g_a", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--kappa_0", type=float, default=5e-3)

    ap.add_argument("--eta_start", type=float, default=2e-3)
    ap.add_argument("--eta_final", type=float, default=5e-4)
    ap.add_argument("--lr_decay_steps", type=int, default=5_000_000)
    ap.add_argument("--lr_power", type=float, default=2.0)

    # WARNING: "epoch" here is a single full-batch SGLD step.
    # Huge values are allowed; snapshot intervals keep storage sane.
    ap.add_argument("--epochs", type=int, default=2_000_000)
    ap.add_argument("--log_interval", type=int, default=25_000)
    ap.add_argument("--diag_interval", type=int, default=2_000)

    ap.add_argument("--activation", type=str, default="sigmoid", choices=["relu", "sigmoid"])
    ap.add_argument("--homogeneity_penalty_weight", type=float, default=0.0)
    ap.add_argument("--tau_switch", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--no_compile", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    train_with_microscope(
        out_dir=out_dir,
        d=args.d, k=args.k,
        P_train=args.P_train, P_test=args.P_test,
        N=args.N, g_w=args.g_w, g_a=args.g_a,
        gamma=args.gamma,
        kappa_0=args.kappa_0,
        eta_start=args.eta_start, eta_final=args.eta_final,
        lr_decay_steps=args.lr_decay_steps, lr_power=args.lr_power,
        epochs=args.epochs,
        log_interval=args.log_interval,
        diag_interval=args.diag_interval,
        activation=args.activation,
        homogeneity_penalty_weight=args.homogeneity_penalty_weight,
        tau_switch=args.tau_switch,
        seed=args.seed,
        compile_model=(not args.no_compile),
    )


if __name__ == "__main__":
    main()
