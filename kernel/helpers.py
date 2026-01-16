# helper.py
import os, csv
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch import nn


# ===================== basic utils =====================

def _to_cpu_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

def eigendecompose_symmetric(K: torch.Tensor, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    K: [N,N] torch (CPU or GPU), symmetric.
    Returns (evals_desc[:k], evecs[:, :k]) both numpy; evecs Euclidean-orthonormal.
    """
    Kc = K.detach().to(torch.float64).cpu()
    evals, evecs = torch.linalg.eigh(Kc)           # ascending
    evals = evals.flip(0).contiguous().numpy()
    evecs = evecs.flip(1).contiguous().numpy()
    if top_k < len(evals):
        evals = evals[:top_k]
        evecs = evecs[:, :top_k]
    return evals.astype(np.float32), evecs.astype(np.float32)


# ===================== forward state (per layer) =====================

@torch.no_grad()
def collect_forward_state(model: nn.Module, X: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
    """
    Returns dictionaries of per-layer tensors for the whole batch X:
      - h_list: [h^(0)=X, h^(1), ..., h^(L)]  shapes: [N,d], [N,m1], ..., [N,mL]
      - u_list: [u^(1), ..., u^(L)]           shapes: [N,m1], ..., [N,mL]
      - D_list: [D^(1), ..., D^(L)]           ReLU slopes (0/1 masks), same shapes as u^(l)
      - W_list: [W^(1), ..., W^(L)]           weight matrices of hidden layers (torch params, shapes m_l x m_{l-1})
      - a_vec:  final readout vector a ∈ R^{m_L} (from last Linear to scalar)
      - f:      predictions f(X) ∈ R^{N}
    Assumes model is: [Linear,ReLU]×(L-1) → Linear,ReLU → Linear(→1)
    """
    device = X.device
    layers = list(model.net)
    # find hidden linear layers and output linear
    W_list: List[torch.Tensor] = []
    bias_list: List[torch.Tensor] = []
    lin_indices = [i for i,m in enumerate(layers) if isinstance(m, nn.Linear)]
    L = len(lin_indices) - 1  # last linear is output; L hidden layers
    assert L >= 1, "Need at least 1 hidden layer."

    # forward with captures
    h_list: List[torch.Tensor] = [X]
    u_list: List[torch.Tensor] = []
    D_list: List[torch.Tensor] = []

    x = X
    lin_ptr = 0
    act_ptr = 0
    # iterate hidden blocks
    for block in range(L):
        lin = layers[lin_ptr]; assert isinstance(lin, nn.Linear)
        W_list.append(lin.weight)  # [m_l, m_{l-1}]
        bias_list.append(lin.bias)
        u = x @ lin.weight.t() + (lin.bias if lin.bias is not None else 0.0)
        u_list.append(u)
        D = (u > 0).to(x.dtype)    # ReLU'
        D_list.append(D)
        # next should be ReLU
        act = layers[lin_ptr+1]; assert isinstance(act, nn.ReLU)
        x = torch.relu(u)
        h_list.append(x)
        lin_ptr += 2
        act_ptr += 1

    # last hidden is h_list[-1], size m_L
    # output linear
    out_lin = layers[lin_ptr]; assert isinstance(out_lin, nn.Linear) and out_lin.out_features == 1
    a_vec = out_lin.weight.squeeze(0)        # [m_L]
    b_out = out_lin.bias.squeeze(0) if out_lin.bias is not None else torch.tensor(0., device=device, dtype=x.dtype)
    f = (x @ a_vec) + b_out                  # [N]

    return dict(h_list=h_list, u_list=u_list, D_list=D_list,
                W_list=W_list, a_vec=a_vec, f=f)


# ===================== last-hidden kernel & core metrics =====================

def last_hidden_kernel(H_L: torch.Tensor) -> torch.Tensor:
    """
    H_L: [N, mL]
    Returns K^{(L)} = (1/n) H H^T  on CPU float32
    """
    N = H_L.shape[0]
    K = (H_L @ H_L.t()) / float(N)
    return K.detach().cpu().to(torch.float32).contiguous()

def effective_dimension(K: torch.Tensor) -> float:
    """
    d_eff = tr(K^2) / (tr K)^2  (user's definition)
    """
    Kd = K.to(torch.float64)
    trK = torch.trace(Kd).item()
    trK2 = (Kd * Kd).sum().item()
    return float(trK2 / (trK**2 + 1e-12))

def Ak_cumulative(U_top_np: np.ndarray, y_np: np.ndarray, k: int) -> float:
    """
    A_k = sum_{i<=k} ( <y, ψ_i>^2 ) / sum_{i} ( <y, ψ_i>^2 )
    With empirical inner product, ψ_i = √n * U[:,i] so this simplifies to:
    A_k = (sum_{i<=k} (U[:,i]^T y)^2) / (y^T y)
    """
    U = U_top_np[:, :k]
    num = float(np.sum((U.T @ y_np.reshape(-1))**2))
    den = float(np.dot(y_np.reshape(-1), y_np.reshape(-1)) + 1e-12)
    return num / den

def kdot_speed(K_prev: Optional[torch.Tensor], K_curr: torch.Tensor) -> Tuple[Optional[torch.Tensor], float]:
    """
    Returns (Kdot, ΔK) where Kdot = K_curr - K_prev (or None if no prev),
    and ΔK = ||Kdot||_F / ||K_curr||_F (0 if no prev).
    """
    if K_prev is None:
        return None, 0.0
    Kdot = (K_curr.to(torch.float64) - K_prev.to(torch.float64))
    num = torch.linalg.norm(Kdot).item()
    den = torch.linalg.norm(K_curr.to(torch.float64)).item() + 1e-12
    return Kdot.to(torch.float32), float(num / den)


def rotation_speed_rho_k(U_np: np.ndarray, evals_np: np.ndarray, Kdot: torch.Tensor, k: int) -> float:
    """
    ρ_k(t) = ( sum_{i≠j≤k} | <ψ_j, Kdot ψ_i> |^2 / (λ_i - λ_j)^2 )^{1/2}
    Computed entirely in float64 to avoid dtype mismatches and improve stability.
    """
    if Kdot is None:
        return 0.0

    # Ensure CPU, float64 for all operands
    Kd = Kdot.detach().to(torch.float64).cpu()                 # [N,N], float64
    U  = torch.from_numpy(U_np[:, :k]).to(torch.float64).cpu() # [N,k], float64

    # M_ji = <ψ_j, Kdot ψ_i> in the empirical (Euclidean) basis
    M = U.t() @ (Kd @ U)                                       # [k,k], float64

    # Eigengap matrix (avoid division by zero on the diagonal)
    lam = np.asarray(evals_np[:k], dtype=np.float64)
    gaps = np.abs(lam.reshape(-1, 1) - lam.reshape(1, -1)) + 1e-12
    mask = ~np.eye(k, dtype=bool)

    M_np = M.numpy()
    num = (M_np[mask] ** 2) / (gaps[mask] ** 2)
    return float(np.sqrt(np.sum(num)))


def projector_drift(U0_np: np.ndarray, U_np: np.ndarray, k: int) -> float:
    """
    || P_k(t) - P_k(0) ||_F, where P_k = U_k U_k^T in Euclidean metric.
    """
    U0 = torch.from_numpy(U0_np[:, :k]).to(torch.float64)  # [N,k]
    Uc = torch.from_numpy(U_np[:, :k]).to(torch.float64)
    P0 = U0 @ U0.t()
    Pt = Uc @ Uc.t()
    diff = Pt - P0
    return float(torch.linalg.norm(diff).item())


# =======
# ============== per-layer transfers (finite dataset estimator) =====================

@torch.no_grad()
def compute_transfers(
    model: nn.Module,
    X: torch.Tensor,
    f_state: Dict[str, List[torch.Tensor]],
    U_L_np: np.ndarray,               # [N,k] top-k evecs of K^{(L)}
    layer_evects: List[np.ndarray],   # for layers 0..L-1: each [N, P_lower]
    layer_evals:  List[np.ndarray],   # for layers 0..L-1: each [P_lower]
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Implements (finite-dataset version, exactly on empirical measure):
      T^{(ℓ)}_{p→i}(t) = ⟨ δ^{(ℓ)}, r^{(ℓ)}_{i,p} ⟩
    with
      r^{(ℓ)}_{i,p} = (1/n^2) * ( sum_a (U_i[a]*Ψ_{p}^{(ℓ-1)}[a]) A_a ) * ( sum_b U_i[b] h^{(L)}(b) ),
      A_a = D^{(ℓ)}(a) S^{(ℓ→L)}(a)^T  ∈ R^{m_ℓ × m_L},
      S^{(ℓ→L)}(a) = ∏_{r=ℓ+1}^{L} [ D^{(r)}(a) W^{(r)} ].

    NOTE: We return the *raw* T (without the (λ_p^{(ℓ-1)} ⟨e,ψ_p^{(ℓ-1)}⟩) scaling).
          Caller should multiply T[ℓ-1] by that per-p factor before computing shares / coherence.

    Returns:
      T_raw:   [L, k, P_lower]  (signed), to be scaled by caller
      shares:  [L]  (placeholder zeros here; computed by caller after scaling)
      C:       scalar (placeholder 0.0 here; computed by caller after scaling)
    """
    device = X.device
    dtype  = X.dtype

    N = X.shape[0]
    h_list = f_state["h_list"]       # h^(0)=X ... h^(L)
    D_list = f_state["D_list"]       # D^(1..L) masks, shapes: [N, m_ℓ]
    W_list = f_state["W_list"]       # W^(1..L), shapes: [m_ℓ, m_{ℓ-1}]
    a_vec  = f_state["a_vec"]        # [m_L]
    L = len(W_list)

    # ---------------------------
    # Backprop signals δ^(ℓ) over the dataset (vectorized)
    # ---------------------------
    delta_list: List[torch.Tensor] = [None] * (L + 1)  # index 1..L used
    # δ^(L)(x) = a ⊙ σ'(u^(L)(x))  -> elementwise product with D^L
    delta_L = D_list[-1] * a_vec.to(device=device, dtype=dtype).unsqueeze(0)   # [N, m_L]
    delta_list[L] = delta_L
    # δ^(ℓ) = (δ^(ℓ+1) @ W^(ℓ+1)) ⊙ D^(ℓ)
    for ell in range(L - 1, 0, -1):
        delta_next = delta_list[ell + 1]                        # [N, m_{ell+1}]
        W_next = W_list[ell].to(device=device, dtype=dtype)     # [m_{ell+1}, m_{ell}]
        tmp = delta_next @ W_next                               # [N, m_{ell}]
        delta_list[ell] = tmp * D_list[ell - 1]                 # [N, m_{ell}]
    # δ̄^(ℓ) = (1/n) Σ_j δ^(ℓ)(x_j)
    delta_bar = [None] + [delta_list[ell].mean(dim=0) for ell in range(1, L + 1)]  # 1..L

    # ---------------------------
    # Right factor: V_right = Σ_b U_i[b] h^{(L)}(b)  = H_L^T U_L
    # ---------------------------
    H_L = h_list[-1].to(device=device, dtype=dtype)                 # [N, m_L]
    U_L = torch.from_numpy(U_L_np).to(device=device, dtype=dtype)   # [N, k]
    V_right = H_L.t() @ U_L                                         # [m_L, k]

    # ---------------------------
    # Output tensor (assumes uniform P_lower across layers)
    # ---------------------------
    P_lower = layer_evects[0].shape[1] if len(layer_evects) > 0 else 0
    T = torch.zeros((L, U_L.shape[1], P_lower), device=device, dtype=dtype)  # [L, k, P]

    # Heuristic chunk size to control peak memory (you can tune this)
    # Keeps tensors like [chunk, m, k] in memory instead of [N, m, k].
    chunk = min(N, 4096)

    # ---------------------------
    # Main loop over layers ℓ
    # ---------------------------
    for ell in range(1, L + 1):
        m_ell = W_list[ell - 1].shape[0]

        # eigenvectors of layer (ℓ-1) (probe basis)
        U_low_np = layer_evects[ell - 1]                 # [N, P_lower]
        U_low = torch.from_numpy(U_low_np).to(device=device, dtype=dtype)  # [N, P_lower]

        # Accumulator R for r^{(ℓ)}: R[k, P, m_ell] = Σ_a (u_k[a] * ψ_p[a]) * Tmat_T[a,k,m_ell]
        R = torch.zeros((U_L.shape[1], P_lower, m_ell), device=device, dtype=dtype)

        # Process samples in chunks to avoid huge [N, m, k] tensors
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            B = e - s

            # Y starts as V_right replicated across the batch: [B, m_L, k]
            # We'll push it down via Y(a) = (∏_{r=L..ℓ+1} (W_r^T diag(D_r[a]))) @ V_right
            Y = V_right.unsqueeze(0).expand(B, -1, -1).clone()  # [B, m_L, k]

            for r in range(L, ell, -1):
                W_r = W_list[r - 1].to(device=device, dtype=dtype)     # [m_r, m_{r-1}]
                D_r = D_list[r - 1][s:e, :]                             # [B, m_r]
                # Y <- W_r^T @ (diag(D_r[a]) @ Y)  batched:
                # (diag(D_r) @ Y) is just row-wise scale of Y by D_r
                Y = (D_r.unsqueeze(2) * Y).transpose(1, 2) @ W_r        # [B, k, m_{r-1}]
                Y = Y.transpose(1, 2)                                   # [B, m_{r-1}, k]

            # A_a @ V_right = D^{(ℓ)}(a) @ Y
            D_ell = D_list[ell - 1][s:e, :]                              # [B, m_ell]
            Y = D_ell.unsqueeze(2) * Y                                   # [B, m_ell, k]
            Tmat_T = Y.transpose(1, 2)                                   # [B, k, m_ell]

            # Accumulate R over the chunk:
            # R[k,P,m] += Σ_{a in chunk} U_L[a,k] * U_low[a,P] * Tmat_T[a,k,m]
            R = R + torch.einsum('bk,bp,bkm->kpm', U_L[s:e, :], U_low[s:e, :], Tmat_T)

        # Normalize by n^2 (empirical inner-product conventions)
        R = R / float(N * N)

        # dot with δ̄^(ℓ): ⟨δ^(ℓ)⟩ · r  -> [k, P_lower]
        delta_bar_ell = delta_bar[ell]                                   # [m_ell]
        T[ell - 1] = torch.einsum('kpm,m->kp', R, delta_bar_ell)         # [k, P_lower]

    # shares / coherence computed by caller after scaling by (λ_p * ⟨e,ψ_p⟩)
    return T, torch.zeros((L,), device=device, dtype=dtype), 0.0


# ===================== orchestration & logging =====================

def make_kernel_cache(out_dir: str, k: int) -> Dict[str, object]:
    os.makedirs(out_dir, exist_ok=True)
    # summary headers (created lazily on first write)
    return dict(
        prev_K=None,            # K^{(L)} at previous metrics step (torch cpu)
        U0=None,                # initial U_top (numpy) for projector drift
        rho_sum=0.0,            # running sum for rho_k
        rho_count=0,            # #points accumulated
        S_sum=0.0,              # running sum for projector drift
        top_k=k
    )

@torch.no_grad()
def compute_and_log_all_metrics(
    out_dir: str,
    model: nn.Module,
    X: torch.Tensor,          # set to evaluate kernel on (train/probe)
    y: torch.Tensor,          # labels on that set
    top_k: int,
    lower_top_k: int,
    epoch: int,
    betas: Tuple[float, float, float],
    track_U: bool,
    save_transfers: bool,
    kernel_cache: Dict[str, object],
):
    device = X.device
    model.eval()

    # 1) forward state (all hidden layers)
    fstate = collect_forward_state(model, X)
    H_L = fstate["h_list"][-1]               # [N, m_L]
    f = fstate["f"].detach()                 # [N]
    yv = y.view(-1).detach()

    # 2) last-hidden kernel and eigensystem
    K_L = last_hidden_kernel(H_L)            # CPU [N,N]
    evals_L, U_L = eigendecompose_symmetric(K_L, top_k=top_k)   # numpy

    # 3) basic metrics on current K
    A_k = Ak_cumulative(U_L, _to_cpu_np(yv), k=top_k)
    d_eff = effective_dimension(K_L)
    Kdot, dK = kdot_speed(kernel_cache["prev_K"], K_L)

    rho_k = rotation_speed_rho_k(U_L, evals_L, Kdot, k=top_k)
    # projector drift vs. epoch 0
    if kernel_cache["U0"] is None:
        kernel_cache["U0"] = U_L.copy()
        Sk = 0.0
    else:
        Sk = projector_drift(kernel_cache["U0"], U_L, k=top_k)

    # running averages
    kernel_cache["rho_sum"] += rho_k
    kernel_cache["rho_count"] += 1
    rho_bar = kernel_cache["rho_sum"] / max(1, kernel_cache["rho_count"])

    # 4) lower-layer eigens (0..L-1) on their feature kernels (for transfers)
    # layer 0 uses raw input X
    h_list = fstate["h_list"]   # length L+1
    Lh = len(h_list)-1
    layer_evects: List[np.ndarray] = []
    layer_evals:  List[np.ndarray] = []
    for ellm1 in range(0, Lh):    # 0..L-1
        H_ellm1 = h_list[ellm1]   # [N, m_{ell-1}] with m_0=d
        K_low = last_hidden_kernel(H_ellm1)      # treat same helper (1/n) HH^T
        ev, Uv = eigendecompose_symmetric(K_low, top_k=min(lower_top_k, K_low.shape[0]))
        layer_evals.append(ev)     # [P_lower]
        layer_evects.append(Uv)    # [N,P_lower]

    # 5) transfers T^{(ℓ)}_{p→i}: first compute the dot-part; then scale by λ_p * <e, ψ_p>
    T_raw, _, _ = compute_transfers(model, X, fstate, U_L_np=U_L,
                                    layer_evects=layer_evects, layer_evals=layer_evals)
    # projections of residual onto lower-layer eigenfunctions (empirical ⟨·,·⟩ with 1/n)
    e_vec = (f - yv).detach()                        # [N]
    N = X.shape[0]
    coherence_num = 0.0
    # scale T
    T_full = torch.zeros_like(T_raw)
    denom_abs = 0.0
    for ell in range(1, Lh+1):
        U_low = torch.from_numpy(layer_evects[ell-1]).to(device=device, dtype=X.dtype)  # [N, P]
        lam_low = layer_evals[ell-1]                                                      # np [P]
        eproj = (U_low.t() @ e_vec.to(U_low.dtype)) / float(N)                            # [P]
        scale = torch.from_numpy(lam_low).to(device=device, dtype=U_low.dtype) * eproj    # [P]         # [P]
        T_full[ell-1] = T_raw[ell-1] * scale.unsqueeze(0)             # [k,P] * [1,P]
        # accumulate coherence parts with top-k i and all p
        S_signed = torch.sum(T_full[ell-1])
        S_abs    = torch.sum(torch.abs(T_full[ell-1]))
        coherence_num += float(torch.abs(S_signed).item())
        denom_abs     += float(S_abs.item())

    C = (coherence_num / (denom_abs + 1e-12)) if denom_abs>0 else 0.0
    # per-layer shares
    shares = []
    for ell in range(1, Lh+1):
        S_abs = torch.sum(torch.abs(T_full[ell-1])).item()
        shares.append(S_abs)
    S_total = sum(shares) + 1e-12
    shares = [s / S_total for s in shares]

    # 6) composite G
    beta1, beta2, beta3 = betas
    G = A_k - beta1 * (d_eff / float(N)) - beta2 * rho_bar - beta3 * (1.0 - C)

    # 7) save per-epoch blobs
    epoch_dir = os.path.join(out_dir, f"epoch_{epoch:04d}")
    os.makedirs(epoch_dir, exist_ok=True)
    np.save(os.path.join(epoch_dir, "K_L_top_eigvals.npy"), evals_L)
    np.save(os.path.join(epoch_dir, "K_L_top_eigvecs.npy"), U_L)
    if save_transfers:
        np.save(os.path.join(epoch_dir, "transfers.npy"), _to_cpu_np(T_full))  # [L,k,P]

    # 8) append scalars
    summ_csv = os.path.join(out_dir, "summary.csv")
    header_exists = os.path.exists(summ_csv) and os.path.getsize(summ_csv) > 0
    with open(summ_csv, "a", newline="") as f:
        w = csv.writer(f)
        if not header_exists:
            header = ["epoch","A_k","d_eff","DeltaK","rho_k","rho_bar","S_k","C","G"] + \
                     [f"eigval_{i}" for i in range(top_k)]
            w.writerow(header)
        w.writerow([epoch, A_k, d_eff, dK, rho_k, rho_bar, Sk, C, G] + list(evals_L[:top_k]))

    shares_csv = os.path.join(out_dir, "shares.csv")
    header_exists = os.path.exists(shares_csv) and os.path.getsize(shares_csv) > 0
    with open(shares_csv, "a", newline="") as f:
        w = csv.writer(f)
        if not header_exists:
            w.writerow(["epoch"] + [f"share_layer_{ell}" for ell in range(1, Lh+1)])
        w.writerow([epoch] + shares)

    # 9) update cache for next step
    kernel_cache["prev_K"] = K_L
    if kernel_cache["U0"] is None:
        kernel_cache["U0"] = U_L.copy()
    kernel_cache["S_sum"] = kernel_cache.get("S_sum", 0.0) + Sk


# ===================== end =====================
