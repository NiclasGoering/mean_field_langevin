# 0908_der3_nospecialsplit_finiteP_multi_simblocks_fullvar3GF3_fast_fix3.py
import os, json, time, math, random
from typing import Dict, List, Tuple
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch
import torch._dynamo as dynamo

 

# ----------------------------- Global perf knobs (Hopper) -----------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

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

def enumerate_combos(d: int, c: int) -> torch.Tensor:
    rows = list(combinations(range(d), c))
    if len(rows) == 0:
        return torch.empty((0, c), dtype=torch.long)
    return torch.tensor(rows, dtype=torch.long)

def build_modes_index_varcard(
    d: int,
    teacher_S: torch.Tensor,
    card_track_max: int,
    sample_high_per_card: int,
    seed: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    mode_lists: List[List[int]] = []

    # teacher first
    mode_lists.append(teacher_S.tolist())

    for c in range(1, d + 1):
        all_c = enumerate_combos(d, c)
        if c == teacher_S.numel() and all_c.numel() > 0:
            teacher_tuple = tuple(teacher_S.tolist())
            keep = [row for row in all_c.tolist() if tuple(row) != teacher_tuple]
            all_c = torch.tensor(keep, dtype=torch.long) if len(keep) > 0 else torch.empty((0, c), dtype=torch.long)

        Nc = all_c.size(0)
        if Nc == 0:
            continue
        if c <= card_track_max:
            mode_lists.extend([row for row in all_c.tolist()])
        else:
            if sample_high_per_card > 0:
                k_take = min(sample_high_per_card, Nc)
                idx = rng.choice(Nc, size=k_take, replace=False)
                mode_lists.extend([all_c[i].tolist() for i in idx])

    if len(mode_lists) == 0:
        mode_lists = [teacher_S.tolist()]

    k_max = max(len(x) for x in mode_lists)
    M = len(mode_lists)
    idx_padded = torch.zeros((M, k_max), dtype=torch.long)
    mask = torch.zeros((M, k_max), dtype=torch.bool)
    for i, row in enumerate(mode_lists):
        L = len(row)
        if L > 0:
            idx_padded[i, :L] = torch.tensor(row, dtype=torch.long)
            mask[i, :L] = True
    return idx_padded, mask

def make_k_subsets(d: int, k: int, M: int, S: torch.Tensor, seed: int = 1) -> List[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    A_list, seen, S_set = [], set(), set(S.tolist())
    while len(A_list) < M:
        A = torch.randperm(d, generator=g)[:k].sort().values
        key = tuple(A.tolist())
        if key in seen:
            continue
        seen.add(key)
        if set(A.tolist()) != S_set:
            A_list.append(A)
    return A_list

def make_boolean_blocks(K: int, P: int, d: int, device: torch.device, seed: int) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.randint(0, 2, (K, P, d), generator=g, device=device, dtype=torch.int8)
    return X.to(torch.float32) * 2.0 - 1.0

def chi_of_modes_blocks_varcard(
    X_blocks: torch.Tensor,        # (K,P,d) float +/-1
    modes_idx_pad: torch.Tensor,   # (M,k_max) long
    modes_mask: torch.Tensor,      # (M,k_max) bool
    chunk: int = 512
) -> torch.Tensor:
    """
    Return chi as INT8 (±1). Cast to float() inside contractions.
    """
    K, P, d = X_blocks.shape
    M, k_max = modes_idx_pad.shape
    device = X_blocks.device
    chi = torch.empty((K, P, M), device=device, dtype=torch.int8)

    if chunk <= 0:
        chunk = M
    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        idx = modes_idx_pad[s:e].to(device)           # (mchunk,k_max)
        msk = modes_mask[s:e].to(device)              # (mchunk,k_max)

        idx_exp = idx.view(1, 1, e - s, k_max).expand(K, P, e - s, k_max)
        gathered = torch.gather(
            X_blocks.unsqueeze(2).expand(K, P, e - s, d),  # float
            3,
            idx_exp
        )  # (K,P,mchunk,k_max) float

        msk_exp = msk.view(1, 1, e - s, k_max).expand(K, P, e - s, k_max)
        ones = torch.ones_like(gathered)
        gathered = torch.where(msk_exp, gathered, ones)  # float

        out = gathered.prod(dim=3)  # float +/-1
        chi[:, :, s:e] = out.to(torch.int8)  # store as int8
    return chi

def activation(z: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "relu":
        return torch.relu(z)
    if kind == "tanh":
        return torch.tanh(z)
    raise ValueError(f"Unknown activation: {kind}")

# ----------------------------- Params -----------------------------

@dataclass
class ModelParams:
    d: int = 30
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
    max_iters: int = 800
    a0: float = 1.0
    t0: float = 150.0
    damping_m: float = 3.0
    eps_D: float = 1e-9
    print_every: int = 20
    project_mS: bool = True
    m_noise_clip: float = 3.0
    ridge_lambda: float = 1e-4
    blocks: int = 4
    resample_blocks_every: int = 0
    crossfit_shuffle_each_iter: bool = True

# ----------------------------- Moments per block (phi only, GEMM) -----------------------------

def compute_phi_J_Sigma_halves(
    w_blocks: torch.Tensor,   # (K,B,d)
    X1: torch.Tensor,         # (K,P1,d)
    X2: torch.Tensor,         # (K,P2,d)
    act: str
):
    """
    SPEED: Use batched GEMM for z = X @ w^T. Only return phi1, phi2, S1, S2.
    """
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=w_blocks.is_cuda):
        # (K,P,d) @ (K,d,B) -> (K,P,B)
        z1 = torch.matmul(X1, w_blocks.transpose(2, 1).contiguous())
        z2 = torch.matmul(X2, w_blocks.transpose(2, 1).contiguous())
        phi1 = activation(z1, act)
        phi2 = activation(z2, act)
    # Compute S in fp32 for stability
    S1 = (phi1.float() * phi1.float()).mean(dim=1)  # (K,B) fp32
    S2 = (phi2.float() * phi2.float()).mean(dim=1)
    return phi1, phi2, S1, S2

# ----------------------------- Implicit J helpers (GEMM in fp32) -----------------------------

def jv(phi: torch.Tensor, C: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    (K,B) = J v, with J = (1/P) Phi^T C
    phi: (K,P,B)  C: (K,P,M) int8  v: (K,M)
    All matmuls done in fp32 for stability.
    """
    P = C.shape[1]
    s = torch.matmul(C.float(), v.float().unsqueeze(2)).squeeze(2)                # (K,P) fp32
    return torch.matmul(phi.float().transpose(1, 2).contiguous(), s.unsqueeze(2)).squeeze(2) / float(P)  # (K,B) fp32

def jtv(phi: torch.Tensor, C: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    (K,M) = J^T u  (fp32 path)
    """
    P = C.shape[1]
    r = torch.matmul(phi.float(), u.float().unsqueeze(2)).squeeze(2)              # (K,P) fp32
    return torch.matmul(C.float().transpose(1, 2).contiguous(), r.unsqueeze(2)).squeeze(2) / float(P)    # (K,M) fp32

# ----------------------------- logπ(w|D_k) and grad (implicit J, GEMM) -----------------------------

@dynamo.disable
def compute_logp_and_grad_w_blocks_indep(
    w_blocks: torch.Tensor, X_blocks: torch.Tensor, chi_blocks: torch.Tensor,
    m_blocks: torch.Tensor, kappa: float, mdl: ModelParams, saem: SAEMParams,
    backreact_lambda: float = 1.0
):
    # g_w^2 = σ_w^2 / d  (σ_w is a std)
    gw2 = (mdl.sigma_w ** 2) / mdl.d
    # per-neuron variance: σ_a^2 / N^γ
    ga2 = (mdl.sigma_a ** 2) / (mdl.N ** mdl.gamma)
    # precision on 'a': 1/ga2 = N^γ / σ_a^2
    A   = 1.0 / ga2

    w_blocks = w_blocks.detach().requires_grad_(True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=w_blocks.is_cuda):
        # GEMM: (K,P,d) @ (K,d,B) -> (K,P,B)
        z   = torch.matmul(X_blocks, w_blocks.transpose(2, 1).contiguous())
        phi = activation(z, mdl.act)

        Ptot = X_blocks.shape[1]
        # JY = (1/P) Phi^T C_S
        JY = torch.matmul(
            phi.transpose(1, 2).contiguous(),
            chi_blocks[:, :, 0].float().unsqueeze(2)
        ).squeeze(2) / float(Ptot)  # (K,B), bf16 under autocast

        # Jm = (1/P) Phi^T (C m)
        Cm = torch.matmul(chi_blocks.float(), m_blocks.unsqueeze(2)).squeeze(2)  # (K,P), fp32
        Cm = Cm.to(phi.dtype)
        Jm = torch.matmul(phi.transpose(1, 2).contiguous(), Cm.unsqueeze(2)).squeeze(2) / float(Ptot)  # (K,B)

        # use fp32 for Sigma/D to avoid bf16 precision issues
        Sigma = (phi.float() * phi.float()).mean(dim=1)  # (K,B) fp32

        D = (A * (kappa ** 2) + Sigma).clamp_min(saem.eps_D)  # fp32

        Jdiff = (JY.float() - Jm.float())                     # fp32
        prior_term = -0.5 * (w_blocks * w_blocks).sum(dim=2) / gw2           # fp32
        data_quad  = 0.5 * (Jdiff * Jdiff) / ((kappa ** 2) * D)              # fp32
        logp       = prior_term + data_quad - 0.5 * torch.log(D)             # fp32

    grad = torch.autograd.grad(logp.sum(), w_blocks, create_graph=False, retain_graph=False)[0]
    grad = torch.where(torch.isfinite(grad), grad, torch.zeros_like(grad))
    return logp.detach(), grad.detach()



# ----------------------------- SGLD -----------------------------

def mcmc_sgld_w_blocks_indep(
    w_blocks: torch.Tensor, X_blocks: torch.Tensor, chi_blocks: torch.Tensor,
    m_blocks: torch.Tensor, kappa: float, mdl: ModelParams, saem: SAEMParams, mcmc: MCMCParams,
    backreact_lambda: float = 1.0
):
    step = mcmc.step_size
    for _ in range(mcmc.n_steps):
        # grad in implicit-J path (internally autocast bf16, critical parts in fp32)
        _, grad = compute_logp_and_grad_w_blocks_indep(
            w_blocks, X_blocks, chi_blocks, m_blocks.to(w_blocks.device),
            kappa, mdl, saem, backreact_lambda=backreact_lambda
        )
        if mcmc.grad_clip and mcmc.grad_clip > 0:
            gn = grad.norm(dim=2, keepdim=True).clamp_min(1e-12)
            grad = grad * (mcmc.grad_clip / gn).clamp(max=1.0)

        noise = torch.randn_like(w_blocks)
        if mcmc.langevin_sqrt2:
            w_blocks = w_blocks + step * grad + noise * math.sqrt(2.0 * step)
        else:
            w_blocks = w_blocks + 0.5 * step * grad + noise * math.sqrt(step)

        if mcmc.clamp_w:
            w_blocks = torch.clamp(w_blocks, -mcmc.clamp_w, mcmc.clamp_w)
        step *= mcmc.step_decay
    return w_blocks.detach()

# ----------------------------- Preconditioned CG (Jacobi) -----------------------------

def pcg_solve_operator(apply_A, rhs, diag_A, tol=1e-6, max_iter=200):
    """
    Solve A x = rhs with PCG (Jacobi preconditioner M=diag(A)), batched over K rows.
    Shapes: rhs, diag_A are (K, M).
    Returns (x, relres, iters).
    """
    device = rhs.device
    dtype  = rhs.dtype

    x  = torch.zeros_like(rhs, dtype=dtype, device=device)
    r  = rhs - apply_A(x)
    Minv = 1.0 / diag_A.clamp_min(1e-30)
    z  = Minv * r
    p  = z.clone()

    rz = (r*z).sum(dim=1, keepdim=True)
    rhs_norm = rhs.norm(dim=1, keepdim=True).clamp_min(1e-30)

    it = 0
    while it < max_iter:
        Ap    = apply_A(p)
        denom = (p*Ap).sum(dim=1, keepdim=True).clamp_min(1e-30)
        alpha = rz / denom
        x     = x + alpha * p
        r     = r - alpha * Ap
        relres = (r.norm(dim=1, keepdim=True) / rhs_norm)
        if (relres <= tol).all():
            break
        z_new = Minv * r
        rz_new = (r*z_new).sum(dim=1, keepdim=True)
        beta = rz_new / rz
        p = z_new + beta * p
        rz = rz_new
        it += 1

    relres_scalar = float(relres.mean().item())
    return x, relres_scalar, it+1

# ----------------------------- SAEM (E: SGLD, M: PCG) -----------------------------

@dataclass
class Sanity:
    bS: float = 0.0
    BSS: float = 0.0
    JS_abs: float = 0.0
    backreact_abs: float = 0.0
    mnoise_l2: float = 0.0

def saem_multimode_quenched_fixedD_multi_runs(
    kappa: float, P: int, M_noise: int, devices: List[int],
    BASE: Dict, SAVE_DIR: str, run_tag: str = ""
):
    mdl = ModelParams(
        d=_get(BASE, "d", 25), N=_get(BASE, "N", 1024), k=_get(BASE, "k", 4),
        sigma_a=_get(BASE, "σa", 1.0), sigma_w=_get(BASE, "σw", 1.0),
        gamma=_get(BASE, "γ", 1.0), act=_get(BASE, "act", "relu")
    )
    mcmc = MCMCParams(
        n_chains_per_device=_get(BASE, "chains_per_device", 1024),
        n_steps=_get(BASE, "mcmc_steps", 300),
        step_size=_get(BASE, "mcmc_step_size", 5e-3),
        step_decay=_get(BASE, "mcmc_step_decay", 0.999),
        langevin_sqrt2=_get(BASE, "langevin_sqrt2", True),
        grad_clip=_get(BASE, "grad_clip", 1e5),
        clamp_w=_get(BASE, "clamp_w", 10.0),
    )
    saem = SAEMParams(
        max_iters=_get(BASE, "opt_steps", 800),
        a0=_get(BASE, "a0", 1.0),
        t0=_get(BASE, "t0", 150.0),
        damping_m=_get(BASE, "damping_m", 3.0),
        eps_D=_get(BASE, "eps_D", 1e-9),
        print_every=_get(BASE, "print_every", 20),
        project_mS=_get(BASE, "project_mS", True),
        m_noise_clip=_get(BASE, "m_noise_clip", 3.0),
        ridge_lambda=_get(BASE, "ridge_lambda", 1e-4),
        blocks=_get(BASE, "blocks", 4),
        resample_blocks_every=_get(BASE, "resample_blocks_every", 0),
        crossfit_shuffle_each_iter=_get(BASE, "crossfit_shuffle_each_iter", True),
    )

    # debug & solver knobs
    DEBUG_SCALE_CHECK   = bool(_get(BASE, "debug_scale_check", True))
    DEBUG_MS_CHECK      = bool(_get(BASE, "debug_ms_check", True))
    DEBUG_CG_RESID      = bool(_get(BASE, "debug_cg_resid", True))
    FORCE_DIAG_IN_CG    = bool(_get(BASE, "force_diag_in_cg", False))
    OFFDIAG_SHRINK      = float(_get(BASE, "offdiag_shrink", 1.0))  # 1=full, 0=diag-only
    PCG_MAX_ITERS       = int(_get(BASE, "pcg_max_iters", 2000))
    PCG_TOL             = float(_get(BASE, "pcg_tol", 1e-5))
    FALLBACK_TO_DIAG    = bool(_get(BASE, "fallback_to_diag_if_bad", True))
    FALLBACK_RELRES_MAX = float(_get(BASE, "fallback_relres_max", 1e-3))

    # back-reaction ramp (for E-step only)
    br_ramp_iters = _get(BASE, "backreact_ramp_iters", 0)
    br_ramp_pow   = _get(BASE, "backreact_ramp_pow", 1.0)

    # modes (variable cardinality supported)
    teacher_S = make_parity_indices(mdl.d, mdl.k, seed=_get(BASE, "teacher_seed", 0))
    card_track_max = int(_get(BASE, "card_track_max", 4))
    sample_high_per_card = int(_get(BASE, "sample_high_per_card", 2000))

    if card_track_max > 0:
        modes_idx_pad, modes_mask = build_modes_index_varcard(
            d=mdl.d,
            teacher_S=teacher_S,
            card_track_max=card_track_max,
            sample_high_per_card=sample_high_per_card,
            seed=_get(BASE, "noise_seed", 1),
        )
    else:
        A_noise = make_k_subsets(mdl.d, mdl.k, M_noise, teacher_S, seed=_get(BASE, "noise_seed", 1))
        modes = [teacher_S] + A_noise
        k_max = teacher_S.numel()
        modes_idx_pad = torch.stack(modes).long()
        modes_mask = torch.ones((len(modes), k_max), dtype=torch.bool)

    Mtot, k_max = modes_idx_pad.shape

    per_dev = []
    for di in devices if len(devices) > 0 else [-1]:
        device = torch.device(f"cuda:{di}") if di >= 0 and torch.cuda.is_available() else torch.device("cpu")
        K, B = saem.blocks, mcmc.n_chains_per_device
        # NOTE: keep original init logic (unchanged as requested)
        gw2 = mdl.sigma_w / mdl.d
        w_blocks = torch.randn(K, B, mdl.d, device=device) * math.sqrt(gw2)

        data_seed = _get(BASE, "data_seed", 0) + 10000 * (di + 1)
        X_blocks = make_boolean_blocks(K, P, mdl.d, device, seed=data_seed)
        chi_blocks = chi_of_modes_blocks_varcard(
            X_blocks, modes_idx_pad.to(device), modes_mask.to(device), chunk=min(512, Mtot)
        )  # (K,P,M) int8

        m_blocks = torch.zeros(K, Mtot, device=device, dtype=torch.float32)
        m_init = _get(BASE, "m_init", 0.5)
        m_blocks[:, 0] = m_init

        per_dev.append(dict(
            device=device, w_blocks=w_blocks, X_blocks=X_blocks,
            chi_blocks=chi_blocks, m_blocks=m_blocks
        ))

    Acoef = (mdl.N**mdl.gamma) / mdl.sigma_a
    t_start = time.time()

    traj_mS, traj_mnoise_rms, traj_time_s = [], [], []

    for it in range(1, saem.max_iters + 1):
        lam_br = min(1.0, (it / float(br_ramp_iters)) ** br_ramp_pow) if br_ramp_iters else 1.0

        # E-step (SGLD)
        for slot in per_dev:
            slot["w_blocks"] = mcmc_sgld_w_blocks_indep(
                slot["w_blocks"], slot["X_blocks"], slot["chi_blocks"],
                slot["m_blocks"], kappa, mdl, saem, mcmc,
                backreact_lambda=lam_br
            )

        sane_list: List[Sanity] = []

        # M-step
        for slot in per_dev:
            device = slot["device"]
            Xb, Cb, w_blocks, m_blocks = slot["X_blocks"], slot["chi_blocks"], slot["w_blocks"], slot["m_blocks"]
            Kb, Ptot, _ = Xb.shape
            Bsize = w_blocks.shape[1]

            if saem.crossfit_shuffle_each_iter:
                perm = torch.randperm(Ptot, device=device)
                Xb = Xb[:, perm, :]
                Cb = Cb[:, perm, :]

            s = Ptot // 2
            X1, X2 = Xb[:, :s, :], Xb[:, s:, :]
            C1, C2 = Cb[:, :s, :], Cb[:, s:, :]

            # ---- Heavy precomputations (bf16 autocast for phi computation only) ----
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=w_blocks.is_cuda):
                phi1, phi2, S1, S2 = compute_phi_J_Sigma_halves(w_blocks, X1, X2, mdl.act)

                # D in fp32
                D1 = torch.clamp(Acoef * (kappa**2) + S1, min=saem.eps_D)  # (K,B) fp32
                D2 = torch.clamp(Acoef * (kappa**2) + S2, min=saem.eps_D)
                invD1 = 1.0 / D1
                invD2 = 1.0 / D2

                # teacher columns via GEMM (cast to fp32)
                P1, P2 = X1.shape[1], X2.shape[1]
                C1_S = C1[:, :, 0].float()
                C2_S = C2[:, :, 0].float()
                J1_S = torch.matmul(phi1.transpose(1, 2).contiguous(), C1_S.unsqueeze(2)).squeeze(2) / float(P1)
                J2_S = torch.matmul(phi2.transpose(1, 2).contiguous(), C2_S.unsqueeze(2)).squeeze(2) / float(P2)
                J1_S = J1_S.float()
                J2_S = J2_S.float()

                # b_vec (implicit, fp32)
                b12 = jtv(phi1, C1, (J2_S * invD2)) / float(Bsize)  # (K,M) fp32
                b21 = jtv(phi2, C2, (J1_S * invD1)) / float(Bsize)
                b_vec = 0.5 * (b12 + b21)                           # (K,M) fp32

                # diag_B (exact, big chunks) -> fp32
                Mtot_local = C1.shape[2]
                diag_B = torch.empty((Kb, Mtot_local), device=device, dtype=torch.float32)
                Mc = 2048
                for m0 in range(0, Mtot_local, Mc):
                    m1 = min(m0 + Mc, Mtot_local)
                    C1c = C1[:, :, m0:m1].float()
                    C2c = C2[:, :, m0:m1].float()
                    T1 = torch.matmul(phi1.transpose(1, 2).contiguous(), C1c).float() / float(P1)  # (K,B,Mc) fp32
                    T2 = torch.matmul(phi2.transpose(1, 2).contiguous(), C2c).float() / float(P2)
                    d1c = torch.einsum('kbm,kbm,kb->km', T1, T1, invD1) / float(Bsize)
                    d2c = torch.einsum('kbm,kbm,kb->km', T2, T2, invD2) / float(Bsize)
                    diag_B[:, m0:m1] = 0.5 * (d1c + d2c)

                # teacher-only diagonal scalar (for logs)
                BSS_vec = 0.5 * ((J1_S * J1_S * invD1).mean(dim=1) + (J2_S * J2_S * invD2).mean(dim=1))  # (K,)

            # Cast for linear solve (already fp32)
            diag_B_f32 = diag_B
            b_vec_f32  = b_vec

            lam = saem.ridge_lambda
            Nf  = float(mdl.N)
            rhs = (Nf * b_vec_f32)

            if bool(_get(BASE, "force_diag_in_cg", False)):
                def apply_A(v):
                    return (1.0 + lam) * v + Nf * (diag_B_f32 * v)
                diag_A = (1.0 + lam) + Nf * diag_B_f32
            else:
                # correct matvec_B with invD factors, all fp32
                def matvec_B(v):
                    if OFFDIAG_SHRINK <= 0.0:
                        return diag_B_f32 * v
                    t1 = jv(phi1, C1, v)                       # (K,B) fp32
                    u1 = jtv(phi1, C1, t1 * invD1)             # (K,M) fp32
                    t2 = jv(phi2, C2, v)
                    u2 = jtv(phi2, C2, t2 * invD2)
                    full = 0.5 * (u1 + u2) / float(Bsize)
                    return OFFDIAG_SHRINK * full + (1.0 - OFFDIAG_SHRINK) * (diag_B_f32 * v)

                def apply_A(v):
                    return (1.0 + lam) * v + Nf * matvec_B(v)

                diag_A = (1.0 + lam) + Nf * diag_B_f32

            # scale sanity
            if DEBUG_SCALE_CHECK:
                with torch.no_grad():
                    v_e0 = torch.zeros(Kb, diag_B_f32.shape[1], device=device, dtype=torch.float32); v_e0[:, 0] = 1.0
                    lhs = apply_A(v_e0) - (1.0 + lam) * v_e0
                    Be0 = (diag_B_f32 * v_e0) if OFFDIAG_SHRINK <= 0.0 else matvec_B(v_e0)
                    ratio = (lhs[:, 0].abs().mean() / (Nf * Be0[:, 0].abs().mean() + 1e-30)).item()
                    print(f"[scale-check] mean |A(eS)-(1+lam)eS| / (N*B eS) ~ {ratio:.3f}  (want ~1.0)")

            # --- Solve with PCG
            m_star, relres, iters = pcg_solve_operator(
                apply_A, rhs, diag_A,
                tol=PCG_TOL,
                max_iter=PCG_MAX_ITERS
            )
            if DEBUG_CG_RESID:
                print(f"pcg_relres: {relres:.2e} (iters={iters})")

            # optional fallback if PCG didn't converge well
            if FALLBACK_TO_DIAG and relres > FALLBACK_RELRES_MAX and not bool(_get(BASE, "force_diag_in_cg", False)):
                m_diag = rhs / diag_A
                print(f"[pcg fallback] relres={relres:.2e} > {FALLBACK_RELRES_MAX:.1e} -> using diag solution")
                m_star = m_diag

            if DEBUG_MS_CHECK:
                with torch.no_grad():
                    ms_1d = (Nf * b_vec_f32[:, :1]) / (1.0 + lam + Nf * diag_B_f32[:, :1])
                    print("check mS:", float(ms_1d.mean().item()), float(m_star[:, :1].mean().item()))

            # SAEM averaging per block
            a_t = saem.a0 / (it + saem.t0)
            m_new = (1 - saem.damping_m * a_t) * m_blocks + saem.damping_m * a_t * m_star

            if saem.project_mS:
                m_new[:, 0] = m_new[:, 0].clamp(0.0, 1.0)
            if saem.m_noise_clip and saem.m_noise_clip > 0:
                m_new[:, 1:] = m_new[:, 1:].clamp(-saem.m_noise_clip, saem.m_noise_clip)

            slot["m_blocks"] = m_new

            # --------- Sanity ----------
            with torch.no_grad():
                JS_abs = 0.5 * (J1_S.abs().mean() + J2_S.abs().mean())
                back1 = jv(phi1, C1, m_blocks).abs().mean()
                back2 = jv(phi2, C2, m_blocks).abs().mean()
                sane = Sanity(
                    bS=float(b_vec_f32[:, 0].mean().detach().cpu()),
                    BSS=float(BSS_vec.mean().detach().cpu()),
                    JS_abs=float(JS_abs.detach().cpu()),
                    backreact_abs=float(0.5*(back1+back2).detach().cpu()),
                    mnoise_l2=float(m_blocks[:, 1:].pow(2).sum(dim=1).sqrt().mean().detach().cpu()) if m_blocks.shape[1] > 1 else 0.0
                )
                sane_list.append(sane)

        # --- Collect trajectories ---
        with torch.no_grad():
            mS_all = torch.cat([slot["m_blocks"][:, 0].detach().float().cpu() for slot in per_dev], dim=0)
            if Mtot > 1:
                noise_rms_list = [slot["m_blocks"][:, 1:].pow(2).mean(dim=1).sqrt().detach().float().cpu()
                                  for slot in per_dev]
                noise_rms = torch.cat(noise_rms_list, dim=0)
            else:
                noise_rms = torch.zeros_like(mS_all)

            traj_mS.append(mS_all.tolist())
            traj_mnoise_rms.append(noise_rms.tolist())
            traj_time_s.append(time.time() - t_start)

        if it % saem.print_every == 0 or it == 1:
            ge_all = 0.5 * (1.0 - mS_all.numpy())**2

            if sane_list:
                bS_mean = float(np.mean([s.bS for s in sane_list]))
                BSS_mean = float(np.mean([s.BSS for s in sane_list]))
                JS_abs_mean = float(np.mean([s.JS_abs for s in sane_list]))
                backreact_abs_mean = float(np.mean([s.backreact_abs for s in sane_list]))
                m_noise_l2_mean = float(np.mean([s.mnoise_l2 for s in sane_list]))
            else:
                bS_mean = BSS_mean = JS_abs_mean = backreact_abs_mean = m_noise_l2_mean = 0.0

            msg = {
                "iter": it, "kappa": kappa, "P": P,
                "blocks": sum(_get(BASE, "blocks", 32) for _ in per_dev),
                "m_S_mean": float(mS_all.mean().item()),
                "m_S_std": float(mS_all.std(unbiased=False).item()),
                "gen_err_mean": float(ge_all.mean()),
                "gen_err_std": float(ge_all.std()),
                "m_noise_rms_mean": float(noise_rms.mean().item()) if noise_rms.numel() > 0 else 0.0,
                # sanity
                "bS_mean": bS_mean,
                "BSS_mean": BSS_mean,
                "J_S_abs_mean": JS_abs_mean,
                "backreact_abs_mean": backreact_abs_mean,
                "m_noise_l2_mean": m_noise_l2_mean,
                "time_s": round(traj_time_s[-1], 2)
            }
            print(json.dumps(msg))

    # Snapshot & save
    with torch.no_grad():
        mS_all = torch.cat([slot["m_blocks"][:, 0].detach().float().cpu() for slot in per_dev], dim=0)
        if Mtot > 1:
            noise_rms_list = [slot["m_blocks"][:, 1:].pow(2).mean(dim=1).sqrt().detach().float().cpu()
                              for slot in per_dev]
            noise_rms = torch.cat(noise_rms_list, dim=0)
        else:
            noise_rms = torch.zeros_like(mS_all)
        ge_all = 0.5 * (1.0 - mS_all.numpy())**2

    blocks_total = int(mS_all.numel())
    summary = {
        "kappa": kappa, "P": P, "mode": "multi-quenched-fixedD-parallel-xfit-matfreePCG-varcard",
        "blocks_total": blocks_total,
        "mS_mean": float(mS_all.mean().item()),
        "mS_std": float(mS_all.std(unbiased=False).item()),
        "gen_err_mean": float(ge_all.mean()),
        "gen_err_std": float(ge_all.std()),
        "m_noise_rms_mean": float(noise_rms.mean().item()) if noise_rms.numel() > 0 else 0.0,
        "BASE": BASE
    }

    trajectories = {
        "iters": int(saem.max_iters),
        "time_s": traj_time_s,
        "m_S_per_iter": traj_mS,
        "m_noise_rms_per_iter": traj_mnoise_rms
    }

    result = {"summary": summary, "trajectories": trajectories}

    os.makedirs(SAVE_DIR, exist_ok=True)
    tag = run_tag if run_tag else f"kap{kappa:.3e}_P{P}_K{_get(BASE,'blocks',32)}_it{_get(BASE,'opt_steps',800)}_B{_get(BASE,'chains_per_device',1024)}"
    out_path = os.path.join(SAVE_DIR, f"quenched_parallel_xfit_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[saved] {out_path}")
    return result

# ----------------------------- Grid runner -----------------------------

def run_grid(
    kappa_list: List[float], P_list: List[int], M_noise: int,
    devices: List[int], BASE: Dict, SAVE_DIR: str, run_tag_prefix: str = ""
):
    for kappa in kappa_list:
        for P in P_list:
            tag = f"{run_tag_prefix}_kap{kappa:.3e}_P{P}_K{_get(BASE,'blocks',32)}"
            _ = saem_multimode_quenched_fixedD_multi_runs(
                kappa=kappa, P=P, M_noise=M_noise, devices=devices,
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
        opt_steps=2000,
        # SGLD
        chains_per_device=8192*2,
        mcmc_steps=400,
        mcmc_step_size=5e-3,
        mcmc_step_decay=0.999,
        langevin_sqrt2=True,
        grad_clip=1e5,
        clamp_w=10.0,
        # SAEM / quenched runs
        a0=1.0, t0=150.0, damping_m=3.0,
        eps_D=1e-9, print_every=50,
        ridge_lambda=1e-3,
        blocks=1,
        resample_blocks_every=0,
        crossfit_shuffle_each_iter=False,
        # seeds
        teacher_seed=0, noise_seed=1, data_seed=0,
        m_init=0.5,
        project_mS=True,
        m_noise_clip=3.0,
        # PCG controls
        pcg_max_iters=4000,
        pcg_tol=1e-5,
        debug_cg_resid=True,
        fallback_to_diag_if_bad=True,
        fallback_relres_max=1e-3,
        # variable-cardinality tracking
        card_track_max=2,
        sample_high_per_card=2000,
        # backreaction ramp for E-step (optional)
        backreact_ramp_iters=100,
        backreact_ramp_pow=1.0,
        # diagnostics
        debug_scale_check=True,
        debug_ms_check=True,
        # modeling toggle
        force_diag_in_cg=False,
        offdiag_shrink=1.0,  # try 0.0 (diag-only), 0.5, 1.0
    )

    save_dir = "/home/goring/mean_field_langevin/MCMC_finiteP/results2/test_card2_h100_8k_2000_2_test"
    os.makedirs(save_dir, exist_ok=True)
    run_tag_prefix = time.strftime("%Y%m%d_%H%M%S")

    kappa_list = [1e-3]
    P_list     = [2000] #[10, 100, 250, 500, 600, 700, 800, 900, 1000, 2000,5000]
    M_noise    = 0  # ignored when card_track_max>0

    run_grid(
        kappa_list=kappa_list,
        P_list=P_list,
        M_noise=M_noise,
        devices=devices,
        BASE=BASE,
        SAVE_DIR=save_dir,
        run_tag_prefix=run_tag_prefix
    )
