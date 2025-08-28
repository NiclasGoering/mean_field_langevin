# faithful_compressed_phase_transition_fixed_EM_split.py
# -----------------------------------------------------------
# Mean-field/SGLD with compressed (k, s)-tied classes + higher-cardinality
# sampled buckets split by (k, s), with correct importance weights and
# correct prior diagonal Diag(N_C). E→M leakage reduced via held-out halves,
# alternating each iteration. Back-reaction gated by cross-fit z-score.
# -----------------------------------------------------------

import os, json, time, math, random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

# ----------------------------- Utils -----------------------------

def check_gpu():
    if not torch.cuda.is_available():
        print("CUDA not available. Running on CPU.")
        return []
    n = torch.cuda.device_count()
    info = []
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        info.append({"index": i, "name": name, "capability": cap, "mem_GB": round(total_mem, 2)})
    print("GPUs:", info)
    return list(range(n))

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _get(BASE: Dict, key: str, default=None):
    return BASE.get(key, default)

# ----------------------------- Core math -----------------------------

def make_parity_indices(d: int, k: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    S = torch.randperm(d, generator=g)[:k]
    return S.sort().values

def make_boolean_blocks(K: int, P: int, d: int, device: torch.device, seed: int) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.randint(0, 2, (K, P, d), generator=g, device=device, dtype=torch.int8)
    return X.to(torch.float32) * 2.0 - 1.0

def activation(z: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "relu": return torch.relu(z)
    if kind == "tanh": return torch.tanh(z)
    raise ValueError(f"Unknown activation: {kind}")

# Elementary symmetric sums e_t over a set of coordinates (batched over K,P)
def elem_sym_sums(X_sub: torch.Tensor, t_max: int) -> torch.Tensor:
    # X_sub: (K,P,m)
    K, P, m = X_sub.shape
    t_max = min(t_max, m)
    e = torch.zeros(K, P, t_max + 1, device=X_sub.device, dtype=X_sub.dtype)
    e[:, :, 0] = 1.0
    for j in range(m):
        v = X_sub[:, :, j]  # (K,P)
        for t in range(min(j+1, t_max), 0, -1):
            e[:, :, t] = e[:, :, t] + v * e[:, :, t-1]
    return e  # (K,P,t_max+1)

def sample_subsets_with_overlap(
    teacher_idx: torch.Tensor, d: int, k: int, s: int, M: int, rng: np.random.Generator
) -> List[List[int]]:
    """
    Sample M subsets A (|A|=k) s.t. |A∩S|=s, uniformly at random.
    """
    S = set(teacher_idx.tolist())
    notS = [i for i in range(d) if i not in S]
    S_list = teacher_idx.tolist()
    cols = []
    for _ in range(M):
        pick_s = rng.choice(len(S_list), size=s, replace=False).tolist() if s > 0 else []
        pick_ns = rng.choice(len(notS), size=k - s, replace=False).tolist() if (k - s) > 0 else []
        A = sorted([S_list[i] for i in pick_s] + [notS[j] for j in pick_ns])
        cols.append(A)
    return cols

# ----------------------------- Build tied classes -----------------------------

def build_classes_and_S_values(
    X_blocks: torch.Tensor,          # (K,P,d)
    teacher_idx: torch.Tensor,       # (k_star,)
    k_track_max: int,
    extra_cards: List[int],
    sample_per_ks: int,
    sample_seed: int = 1234
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[str,int,int]], torch.Tensor]:
    """
    Returns:
      S_teacher_vals: (K,P)
      S_values:       (K,P,C)  aggregated sums U_C(x) for each class
      labels:         list of ("ks" or "ks_extra", k, s)
      counts:         (C,) total class sizes |C| for prior diag (teacher handled separately)
    Notes:
      - For k <= k_track_max: exact sums via elementary symmetric polynomials (no sampling).
      - For k in extra_cards: per (k,s) sample 'sample_per_ks' subsets and importance-weight
        by w_{k,s} = total_count / sampled_count; prior diag uses total_count.
    """
    device = X_blocks.device
    K, P, d = X_blocks.shape
    k_star = int(teacher_idx.numel())

    # Teacher S_teacher(x)
    S_teacher_vals = X_blocks[:, :, teacher_idx].prod(dim=2)  # (K,P)

    # Partition into S and S^c for exact (k,s) sums
    S_mask = torch.zeros(d, dtype=torch.bool, device=device)
    S_mask[teacher_idx.to(device)] = True
    X_S   = X_blocks[:, :, S_mask]          # (K,P,k_star)
    X_Sc  = X_blocks[:, :, ~S_mask]         # (K,P,d-k_star)
    e_S   = elem_sym_sums(X_S, t_max=k_star)                      # (K,P,k_star+1)
    e_Sc  = elem_sym_sums(X_Sc, t_max=max(k_track_max, 0))        # (K,P,k_track_max+1)

    S_values_list: List[torch.Tensor] = []
    labels: List[Tuple[str,int,int]] = []
    counts: List[float] = []

    # Exact (k,s) for k <= k_track_max
    if k_track_max > 0:
        for k in range(1, k_track_max + 1):
            s_max = min(k, k_star)
            for s in range(0, s_max + 1):
                # Skip the teacher class itself: (k==k_star and s==k_star)
                if (k == k_star) and (s == k_star):
                    continue
                total_count = math.comb(k_star, s) * math.comb(d - k_star, k - s)
                if total_count == 0:
                    continue
                # U_{k,s}(x) = sum_{|A|=k, |A∩S|=s} chi_A(x) = e_S[s] * e_Sc[k-s]
                Uks = e_S[:, :, s] * e_Sc[:, :, k - s]   # (K,P)
                S_values_list.append(Uks)
                labels.append(("ks", k, s))
                counts.append(float(total_count))

    # Extra higher-cardinality buckets, split by (k,s), importance-weighted
    rng = np.random.default_rng(sample_seed)
    for k in extra_cards:
        s_max = min(k, k_star)
        for s in range(0, s_max + 1):
            # Skip teacher class if k==k_star & s==k_star (already handled as teacher)
            if (k == k_star) and (s == k_star):
                continue
            total_count = math.comb(k_star, s) * math.comb(d - k_star, k - s)
            if total_count == 0:
                continue
            M = min(sample_per_ks, total_count)  # never sample more than exists
            if M == 0:
                continue
            cols = sample_subsets_with_overlap(teacher_idx, d, k, s, M, rng)
            idx = torch.tensor(cols, dtype=torch.long, device=device)  # (M, k)
            # sum_A chi_A(x) over sampled A, then importance-weight by total_count / M
            gathered = torch.gather(
                X_blocks.unsqueeze(2).expand(K, P, M, d),
                3, idx.view(1, 1, M, k).expand(K, P, M, k)
            )
            chi_sum = gathered.prod(dim=3).sum(dim=2)  # (K,P)
            weight = float(total_count) / float(M)
            Uks_hat = chi_sum * weight
            S_values_list.append(Uks_hat)
            labels.append(("ks_extra", k, s))
            counts.append(float(total_count))

    if len(S_values_list) == 0:
        S_values = torch.zeros(K, P, 0, device=device)
        counts_t = torch.zeros(0, device=device, dtype=torch.float32)
    else:
        S_values = torch.stack(S_values_list, dim=2)     # (K,P,C)
        counts_t = torch.tensor(counts, dtype=torch.float32, device=device)  # (C,)

    return S_teacher_vals, S_values, labels, counts_t

# ----------------------------- Params -----------------------------

@dataclass
class ModelParams:
    d: int = 25
    N: int = 1024
    k: int = 4
    sigma_a: float = 1.0
    sigma_w: float = 1.0
    gamma: float = 1.0
    act: str = "relu"

@dataclass
class MCMCParams:
    n_chains_per_device: int = 1024   # B
    n_steps: int = 300
    step_size: float = 5e-3
    step_decay: float = 0.999
    langevin_sqrt2: bool = True
    grad_clip: float = 1e5
    clamp_w: float = 10.0

@dataclass
class SAEMParams:
    max_iters: int = 1200
    a0: float = 1.0
    t0: float = 150.0
    damping_m: float = 3.0
    eps_D: float = 1e-9
    print_every: int = 20
    project_mS: bool = True
    m_noise_clip: float = 3.0
    ridge_lambda: float = 1e-4
    blocks: int = 1
    resample_blocks_every: int = 0
    crossfit_shuffle_each_iter: bool = True

# ----------------------------- E-step (held-out half) -----------------------------

def compute_logp_and_grad_w_blocks_compressed(
    w_blocks: torch.Tensor,          # (K,B,d)
    X_half: torch.Tensor,            # (K,P_half,d)  -- E-step half
    S_teacher_half: torch.Tensor,    # (K,P_half)
    S_values_half: torch.Tensor,     # (K,P_half,C)
    m_full: torch.Tensor,            # (K,R) with R=1+C; first col is m_S, rest alphas
    kappa: float,
    mdl: ModelParams,
    saem: SAEMParams,
    backreact_lambda: float = 0.0
):
    K, P_h, d = X_half.shape
    gw2 = mdl.sigma_w / mdl.d
    ga2 = mdl.sigma_a / (mdl.N ** mdl.gamma)
    Acoef = 1.0 / ga2

    w_blocks = w_blocks.detach().requires_grad_(True)

    z = torch.einsum('kpd,kbd->kpb', X_half, w_blocks)   # (K,P_half,B)
    phi = activation(z, mdl.act)                          # (K,P_half,B)

    # Aggregated projections on the E-step half
    J_S = torch.einsum('kpb,kp->kb', phi, S_teacher_half) / float(P_h)     # (K,B)
    C = S_values_half.shape[2]
    if C > 0:
        U_classes = torch.einsum('kpb,kpc->kbc', phi, S_values_half) / float(P_h)  # (K,B,C)
    else:
        U_classes = torch.zeros(K, w_blocks.shape[1], 0, device=w_blocks.device)

    Sigma = (phi * phi).mean(dim=1)                                      # (K,B)
    D = torch.clamp(Acoef * (kappa**2) + Sigma, min=saem.eps_D)

    m_S = m_full[:, 0]                      # (K,)
    m_C = m_full[:, 1:] if C > 0 else torch.zeros(K, 0, device=w_blocks.device)

    # Back-reaction gated on both teacher & classes:
    # J_beta = J_S - lam * ( m_S*J_S + sum_C alpha_C * U_C )
    term_classes = 0.0 if C == 0 else torch.einsum('kbc,kc->kb', U_classes, m_C)
    J_beta = J_S - backreact_lambda * (m_S.unsqueeze(1) * J_S + term_classes)

    prior_term = -0.5 * (w_blocks * w_blocks).sum(dim=2) / gw2
    data_term  = -0.5 * torch.log(D) + 0.5 * (J_beta * J_beta) / ((kappa**2) * D)
    logp = (prior_term + data_term).sum(dim=1)

    grad = torch.autograd.grad(logp.sum(), w_blocks, create_graph=False, retain_graph=False)[0]
    grad = torch.where(torch.isfinite(grad), grad, torch.zeros_like(grad))
    return logp.detach(), grad.detach()

def mcmc_sgld_w_blocks_compressed(
    w_blocks: torch.Tensor,
    X_half: torch.Tensor,
    S_teacher_half: torch.Tensor,
    S_values_half: torch.Tensor,
    m_full: torch.Tensor,
    kappa: float,
    mdl: ModelParams, saem: SAEMParams, mcmc: MCMCParams,
    backreact_lambda: float = 0.0
):
    step = mcmc.step_size
    for _ in range(mcmc.n_steps):
        _, grad = compute_logp_and_grad_w_blocks_compressed(
            w_blocks, X_half, S_teacher_half, S_values_half, m_full,
            kappa, mdl, saem, backreact_lambda
        )
        if mcmc.grad_clip and mcmc.grad_clip > 0:
            gn = grad.norm(dim=2, keepdim=True).clamp_min(1e-12)
            grad = grad * (mcmc.grad_clip / gn).clamp(max=1.0)

        if mcmc.langevin_sqrt2:
            w_blocks = w_blocks + step * grad + torch.randn_like(w_blocks) * math.sqrt(2.0 * step)
        else:
            w_blocks = w_blocks + 0.5 * step * grad + torch.randn_like(w_blocks) * math.sqrt(step)

        if mcmc.clamp_w:
            w_blocks = torch.clamp(w_blocks, -mcmc.clamp_w, mcmc.clamp_w)
        step *= mcmc.step_decay
    return w_blocks.detach()

# ----------------------------- M-step (other half) + z-score -----------------------------

def batched_spd_solve_chol(R: torch.Tensor, y: torch.Tensor, jitter0: float = 1e-10, max_tries: int = 6) -> torch.Tensor:
    # R: (K,R,R), y: (K,R)
    K, r, _ = R.shape
    dtype = torch.float64
    R = 0.5 * (R + R.transpose(1, 2)).to(dtype)
    y = y.to(dtype)
    eye = torch.eye(r, device=R.device, dtype=dtype).unsqueeze(0).expand(K, r, r)
    jitter = jitter0
    for _ in range(max_tries):
        try:
            L = torch.linalg.cholesky(R + jitter * eye)
            z = torch.cholesky_solve(y.unsqueeze(2), L).squeeze(2)
            return z.to(torch.float32)
        except Exception:
            jitter *= 10.0
    z = torch.linalg.solve(R + jitter * eye, y.unsqueeze(2)).squeeze(2)
    return z.to(torch.float32)

def m_step_small_system_with_stats(
    w_blocks: torch.Tensor,                # (K,B,d)
    X_half_M: torch.Tensor,                # (K,P_m,d)  -- M-step half
    S_teacher_half_M: torch.Tensor,        # (K,P_m)
    S_values_half_M: torch.Tensor,         # (K,P_m,C)
    counts: torch.Tensor,                  # (C,) total counts per class
    mdl: ModelParams, saem: SAEMParams,
    kappa: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      m_star: (K,R) solution on the M-step half
      b0:     (K,) teacher b (pre-multiplied by nothing)
      B00:    (K,) teacher diagonal B
    """
    K, Pm, d = X_half_M.shape
    Bsize = w_blocks.shape[1]
    C = S_values_half_M.shape[2]
    Rdim = 1 + C

    # Forward on M-step half
    z = torch.einsum('kpd,kbd->kpb', X_half_M, w_blocks)  # (K,Pm,B)
    phi = activation(z, mdl.act)

    # Projections
    t = torch.einsum('kpb,kp->kb', phi, S_teacher_half_M) / float(Pm)         # (K,B)
    if C > 0:
        U = torch.einsum('kpb,kpc->kbc', phi, S_values_half_M) / float(Pm)    # (K,B,C)
    else:
        U = torch.zeros(K, B, 0, device=w_blocks.device)

    # D
    ga2 = mdl.sigma_a / (mdl.N ** mdl.gamma)
    Acoef = 1.0 / ga2
    Sigma = (phi * phi).mean(dim=1)         # (K,B)
    D = torch.clamp(Acoef * (kappa**2) + Sigma, min=saem.eps_D)
    invD = 1.0 / D

    m_star = torch.zeros(K, Rdim, device=w_blocks.device)
    b0 = torch.zeros(K, device=w_blocks.device)
    B00 = torch.zeros(K, device=w_blocks.device)

    # Prior diag: teacher count=1, then class total counts
    diag_vec = torch.cat([torch.ones(1, device=w_blocks.device, dtype=torch.float32),
                          counts.to(torch.float32)])  # (Rdim,)

    for k in range(K):
        U_full = torch.cat([t[k].unsqueeze(1), U[k]], dim=1)    # (B, Rdim)

        # Weighted gram B = (U^T diag(invD) U)/Bsize
        Uw = U_full * invD[k].unsqueeze(1).sqrt()
        B_small = (Uw.transpose(0,1) @ Uw) / float(Bsize)       # (R,R)

        # Cross-fitted b = (U^T (t*invD))/Bsize
        w_vec = (t[k] * invD[k]).unsqueeze(1)                   # (B,1)
        b_small = (U_full.transpose(0,1) @ w_vec).squeeze(1) / float(Bsize)  # (R,)

        # Save teacher stats for z-score (index 0)
        b0[k] = b_small[0]
        B00[k] = B_small[0, 0]

        # Solve: (Diag(N_counts) + N B) * alpha = N b   (+ ridge*I)
        I = torch.eye(Rdim, device=w_blocks.device, dtype=B_small.dtype)
        lhs = torch.diag(diag_vec).to(B_small.dtype) + mdl.N * B_small + saem.ridge_lambda * I
        rhs = mdl.N * b_small
        m_star[k] = batched_spd_solve_chol(lhs.unsqueeze(0), rhs.unsqueeze(0)).squeeze(0)

    return m_star, b0, B00

# ----------------------------- Main SAEM loop -----------------------------

def saem_compressed_multi_runs(
    kappa: float, P: int, devices: List[int],
    BASE: Dict, SAVE_DIR: str, run_tag: str = ""
):
    # Params
    mdl = ModelParams(
        d=_get(BASE, "d", 25), N=_get(BASE, "N", 1024), k=_get(BASE, "k", 4),
        sigma_a=_get(BASE, "σa", 1.0), sigma_w=_get(BASE, "σw", 1.0),
        gamma=_get(BASE, "γ", 1.0), act=_get(BASE, "act", "relu")
    )
    mcmc = MCMCParams(
        n_chains_per_device=_get(BASE, "chains_per_device", 1024),
        n_steps=_get(BASE, "mcmc_steps", 400),
        step_size=_get(BASE, "mcmc_step_size", 5e-3),
        step_decay=_get(BASE, "mcmc_step_decay", 0.999),
        langevin_sqrt2=_get(BASE, "langevin_sqrt2", True),
        grad_clip=_get(BASE, "grad_clip", 1e5),
        clamp_w=_get(BASE, "clamp_w", 10.0),
    )
    saem = SAEMParams(
        max_iters=_get(BASE, "opt_steps", 2000),
        a0=_get(BASE, "a0", 1.0),
        t0=_get(BASE, "t0", 150.0),
        damping_m=_get(BASE, "damping_m", 3.0),
        eps_D=_get(BASE, "eps_D", 1e-9),
        print_every=_get(BASE, "print_every", 20),
        project_mS=_get(BASE, "project_mS", True),
        m_noise_clip=_get(BASE, "m_noise_clip", 3.0),
        ridge_lambda=_get(BASE, "ridge_lambda", 1e-4),
        blocks=_get(BASE, "blocks", 1),
        resample_blocks_every=_get(BASE, "resample_blocks_every", 0),
        crossfit_shuffle_each_iter=True,   # we always reshuffle
    )

    # Compression knobs
    k_track_max = int(_get(BASE, "k_track_max", 4))  # exact up to here
    # default: 5 higher k's after k_track_max
    extra_cards = _get(BASE, "extra_cards", None)
    if extra_cards is None:
        extra_cards = [k for k in range(k_track_max + 1, min(mdl.d, k_track_max + 6))][:5]
    sample_per_ks = int(_get(BASE, "sample_per_ks", 200))  # per (k,s)
    # Back-reaction gating
    z_thresh = float(_get(BASE, "z_thresh", 3.0))
    br_ramp_iters = int(_get(BASE, "backreact_ramp_iters", 300))
    br_ramp_pow   = float(_get(BASE, "backreact_ramp_pow", 1.0))

    teacher_S = make_parity_indices(mdl.d, mdl.k, seed=_get(BASE, "teacher_seed", 0))

    # Per device/block state
    per_dev = []
    for di in devices if len(devices) > 0 else [-1]:
        device = torch.device(f"cuda:{di}") if di >= 0 and torch.cuda.is_available() else torch.device("cpu")
        K, B = saem.blocks, mcmc.n_chains_per_device
        gw2 = mdl.sigma_w / mdl.d
        w_blocks = torch.randn(K, B, mdl.d, device=device) * math.sqrt(gw2)

        data_seed = _get(BASE, "data_seed", 0) + 10000 * (di + 1)
        X_blocks = make_boolean_blocks(K, P, mdl.d, device, seed=data_seed)  # fixed datasets (quenched)

        S_teacher_vals, S_values, labels, counts = build_classes_and_S_values(
            X_blocks, teacher_S.to(device),
            k_track_max=k_track_max,
            extra_cards=extra_cards,
            sample_per_ks=sample_per_ks,
            sample_seed=_get(BASE, "noise_seed", 1)
        )
        C = S_values.shape[2]
        Rdim = 1 + C

        m_blocks = torch.zeros(K, Rdim, device=device, dtype=torch.float32)
        m_blocks[:, 0] = float(_get(BASE, "m_init", 0.5))

        per_dev.append({
            "device": device,
            "w_blocks": w_blocks,
            "X_blocks": X_blocks,
            "S_teacher_vals": S_teacher_vals,
            "S_values": S_values,
            "class_counts": counts,     # (C,)
            "labels": labels,
            "m_blocks": m_blocks,       # (K, 1+C)
        })

    t_start = time.time()

    # Trajectories
    traj_mS = []
    traj_noise_rms = []
    traj_sum_all = []
    traj_energy_mass = []
    traj_teacher_frac = []
    traj_time_s = []

    # Back-reaction gating state
    gate_start_iter = None
    current_lambda = 0.0
    # Alternate halves
    use_half_A_for_E = True

    for it in range(1, saem.max_iters + 1):
        # Fresh permutation for held-out halves
        for slot in per_dev:
            K, Ptot, _ = slot["X_blocks"].shape
            perm = torch.randperm(Ptot, device=slot["device"])
            Xp  = slot["X_blocks"][:, perm, :]
            Stp = slot["S_teacher_vals"][:, perm]
            Svp = slot["S_values"][:, perm, :]

            s = Ptot // 2
            if use_half_A_for_E:
                slot["X_E"]  = Xp[:, :s, :]
                slot["St_E"] = Stp[:, :s]
                slot["Sv_E"] = Svp[:, :s, :]
                slot["X_M"]  = Xp[:, s:, :]
                slot["St_M"] = Stp[:, s:]
                slot["Sv_M"] = Svp[:, s:, :]
            else:
                slot["X_E"]  = Xp[:, s:, :]
                slot["St_E"] = Stp[:, s:]
                slot["Sv_E"] = Svp[:, s:, :]
                slot["X_M"]  = Xp[:, :s, :]
                slot["St_M"] = Stp[:, :s]
                slot["Sv_M"] = Svp[:, :s, :]

        # ---- E-step on E-half (uses previous current_lambda) ----
        for slot in per_dev:
            slot["w_blocks"] = mcmc_sgld_w_blocks_compressed(
                slot["w_blocks"],
                slot["X_E"], slot["St_E"], slot["Sv_E"],
                slot["m_blocks"],
                kappa, mdl, saem, mcmc,
                backreact_lambda=current_lambda
            )

        # ---- M-step on M-half ----
        z_scores = []
        for slot in per_dev:
            m_star, b0, B00 = m_step_small_system_with_stats(
                slot["w_blocks"], slot["X_M"], slot["St_M"], slot["Sv_M"],
                slot["class_counts"], mdl, saem, kappa
            )
            # SAEM averaging
            a_t = saem.a0 / (it + saem.t0)
            m_new = (1 - saem.damping_m * a_t) * slot["m_blocks"] + saem.damping_m * a_t * m_star

            if saem.project_mS:
                m_new[:, 0] = m_new[:, 0].clamp(0.0, 1.0)
            if saem.m_noise_clip and saem.m_noise_clip > 0 and m_new.shape[1] > 1:
                m_new[:, 1:] = m_new[:, 1:].clamp(-saem.m_noise_clip, saem.m_noise_clip)

            slot["m_blocks"] = m_new

            # z-score per block: z = (N*b0) / sqrt(eps + N*B00)
            z = (mdl.N * b0) / torch.sqrt(1e-8 + mdl.N * torch.clamp(B00, min=0.0))
            z_scores.append(z.detach().float().cpu())

        # Update back-reaction gate for NEXT E-step
        z_all = torch.cat(z_scores, dim=0)
        z_mean = float(z_all.mean().item())
        if gate_start_iter is None and z_mean > z_thresh:
            gate_start_iter = it
        if gate_start_iter is None:
            current_lambda = 0.0
        else:
            prog = max(0.0, (it - gate_start_iter) / float(max(1, br_ramp_iters)))
            current_lambda = min(1.0, prog ** br_ramp_pow)

        # Alternate halves next iter
        use_half_A_for_E = not use_half_A_for_E

        # ---- Collect stats ----
        with torch.no_grad():
            mS_all = torch.cat([slot["m_blocks"][:, 0].detach().float().cpu() for slot in per_dev], dim=0)
            noise_rms_list, sum_all_list, energy_mass_list, teacher_frac_list = [], [], [], []
            for slot in per_dev:
                counts = slot["class_counts"]  # (C,)
                m_full = slot["m_blocks"]      # (K,R)
                if m_full.shape[1] == 1:
                    rms = torch.zeros(m_full.shape[0])
                    sum_all = m_full[:, 0]
                    mass = m_full[:, 0]**2
                    frac = torch.ones_like(m_full[:, 0])
                else:
                    alphas = m_full[:, 1:]      # (K,C)
                    num = (alphas.pow(2) @ counts).cpu()               # (K,)
                    den = counts.sum().cpu() + 1e-12
                    rms = (num / den).sqrt()
                    sum_all = m_full[:, 0].cpu() + (alphas @ counts).cpu()   # sum_A m_A
                    mass = (m_full[:, 0]**2).cpu() + num                    # L2 energy
                    frac = (m_full[:, 0].cpu()**2) / (mass + 1e-12)
                noise_rms_list.append(rms)
                sum_all_list.append(sum_all)
                energy_mass_list.append(mass)
                teacher_frac_list.append(frac)

            noise_rms_all = torch.cat(noise_rms_list, dim=0)
            sum_all_all = torch.cat(sum_all_list, dim=0)
            energy_mass_all = torch.cat(energy_mass_list, dim=0)
            teacher_frac_all = torch.cat(teacher_frac_list, dim=0)

            traj_mS.append(mS_all.tolist())
            traj_noise_rms.append(noise_rms_all.tolist())
            traj_sum_all.append(sum_all_all.tolist())
            traj_energy_mass.append(energy_mass_all.tolist())
            traj_teacher_frac.append(teacher_frac_all.tolist())
            traj_time_s.append(time.time() - t_start)

        # ---- Logging ----
        if it % saem.print_every == 0 or it == 1:
            ge_all = 0.5 * (1.0 - mS_all.numpy())**2
            msg = {
                "iter": it, "kappa": kappa, "P": P,
                "blocks_total": sum(_get(BASE, "blocks", 1) for _ in per_dev),
                "m_S_mean": float(mS_all.mean().item()),
                "m_S_std": float(mS_all.std(unbiased=False).item()),
                "gen_err_mean": float(ge_all.mean()),
                "gen_err_std": float(ge_all.std()),
                "noise_rms_mean": float(noise_rms_all.mean().item()),
                "sum_all_coeffs_mean": float(sum_all_all.mean().item()),
                "coeff_l2_mass_mean": float(energy_mass_all.mean().item()),
                "teacher_fraction_mean": float(teacher_frac_all.mean().item()),
                "z_mean": z_mean,
                "lambda_br": round(current_lambda, 3),
                "time_s": round(traj_time_s[-1], 2)
            }
            print(json.dumps(msg))

    # ---- Final snapshot & save ----
    with torch.no_grad():
        mS_all = torch.cat([slot["m_blocks"][:, 0].detach().float().cpu() for slot in per_dev], dim=0)
        noise_rms_list, sum_all_list, energy_mass_list, teacher_frac_list = [], [], [], []
        for slot in per_dev:
            counts = slot["class_counts"]
            m_full = slot["m_blocks"]
            if m_full.shape[1] == 1:
                rms = torch.zeros(m_full.shape[0])
                sum_all = m_full[:, 0]
                mass = m_full[:, 0]**2
                frac = torch.ones_like(m_full[:, 0])
            else:
                alphas = m_full[:, 1:]
                num = (alphas.pow(2) @ counts).cpu()
                den = counts.sum().cpu() + 1e-12
                rms = (num / den).sqrt()
                sum_all = m_full[:, 0].cpu() + (alphas @ counts).cpu()
                mass = (m_full[:, 0]**2).cpu() + num
                frac = (m_full[:, 0].cpu()**2) / (mass + 1e-12)
            noise_rms_list.append(rms)
            sum_all_list.append(sum_all)
            energy_mass_list.append(mass)
            teacher_frac_list.append(frac)

        noise_rms_all = torch.cat(noise_rms_list, dim=0)
        sum_all_all = torch.cat(sum_all_list, dim=0)
        energy_mass_all = torch.cat(energy_mass_list, dim=0)
        teacher_frac_all = torch.cat(teacher_frac_list, dim=0)
        ge_all = 0.5 * (1.0 - mS_all.numpy())**2

    blocks_total = int(mS_all.numel())
    summary = {
        "kappa": kappa, "P": P, "mode": "compressed-(k,s)+extra-(k,s) (weighted, held-out E/M)",
        "blocks_total": blocks_total,
        "mS_mean": float(mS_all.mean().item()),
        "mS_std": float(mS_all.std(unbiased=False).item()),
        "gen_err_mean": float(ge_all.mean()),
        "gen_err_std": float(ge_all.std()),
        "noise_rms_mean": float(noise_rms_all.mean().item()),
        "sum_all_coeffs_mean": float(sum_all_all.mean().item()),
        "coeff_l2_mass_mean": float(energy_mass_all.mean().item()),
        "teacher_fraction_mean": float(teacher_frac_all.mean().item()),
        "BASE": BASE
    }

    trajectories = {
        "iters": int(saem.max_iters),
        "time_s": traj_time_s,
        "m_S_per_iter": traj_mS,
        "noise_rms_per_iter": traj_noise_rms,
        "sum_all_coeffs_per_iter": traj_sum_all,
        "coeff_l2_mass_per_iter": traj_energy_mass,
        "teacher_fraction_per_iter": traj_teacher_frac
    }

    result = {"summary": summary, "trajectories": trajectories}

    os.makedirs(SAVE_DIR, exist_ok=True)
    tag = run_tag if run_tag else f"kap{kappa:.3e}_P{P}_K{_get(BASE,'blocks',1)}"
    out_path = os.path.join(SAVE_DIR, f"compressed_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[saved] {out_path}")
    return result

# ----------------------------- Grid runner -----------------------------

def run_grid(
    kappa_list: List[float], P_list: List[int],
    devices: List[int], BASE: Dict, SAVE_DIR: str, run_tag_prefix: str = ""
):
    for kappa in kappa_list:
        for P in P_list:
            tag = f"{run_tag_prefix}_kap{kappa:.3e}_P{P}_K{_get(BASE,'blocks',1)}"
            _ = saem_compressed_multi_runs(
                kappa=kappa, P=P, devices=devices,
                BASE=BASE, SAVE_DIR=SAVE_DIR, run_tag=tag
            )

# ----------------------------- Main -----------------------------

if __name__ == "__main__":
    set_seed(42)
    devices = check_gpu()

    BASE = dict(
        d=25, N=1024, k=4,
        σa=1.0, σw=1.0, γ=1.0,
        act="relu",
        opt_steps=1200,
        # SGLD
        chains_per_device=1024,
        mcmc_steps=400,
        mcmc_step_size=5e-3,
        mcmc_step_decay=0.999,
        langevin_sqrt2=True,
        grad_clip=1e5,
        clamp_w=10.0,
        # SAEM
        a0=1.0, t0=150.0, damping_m=3.0,
        eps_D=1e-9, print_every=50,
        ridge_lambda=1e-4,
        blocks=1,
        resample_blocks_every=0,
        # seeds
        teacher_seed=0, noise_seed=1, data_seed=0,
        m_init=0.5,
        project_mS=True,
        m_noise_clip=3.0,
        # ---- compression controls ----
        k_track_max=5,                  # exact (k,s) up to k=4
        extra_cards=None,               # default: 5 cards after k_track_max
        sample_per_ks=800,              # per (k,s)
        # ---- back-reaction gating ----
        z_thresh=3.0,
        backreact_ramp_iters=100,
        backreact_ramp_pow=1.0,
    )

    save_dir = "/home/goring/mean_field_langevin/MCMC_finiteP/results/2208_grid_testpaul2"
    os.makedirs(save_dir, exist_ok=True)
    run_tag_prefix = time.strftime("%Y%m%d_%H%M%S")

    # Example sweep (adjust κ to move the phase threshold)
    kappa_list = [1e-3]   # try larger κ (e.g., 0.05, 0.1, 0.2) to push transition to larger P
    P_list     = [100,500,1000,2000]

    run_grid(
        kappa_list=kappa_list,
        P_list=P_list,
        devices=devices,
        BASE=BASE,
        SAVE_DIR=save_dir,
        run_tag_prefix=run_tag_prefix
    )
