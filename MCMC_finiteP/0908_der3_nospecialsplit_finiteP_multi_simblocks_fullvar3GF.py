# 0908_der3_nospecialsplit_finiteP_multi_simblocks_matfreecg.py
import os, json, time, math, random
from typing import Dict, List, Tuple
from dataclasses import dataclass
from itertools import combinations

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
    """
    Return (idx_padded, mask) where:
      idx_padded: (M, k_max) LongTensor with zero-filled padding
      mask:       (M, k_max) BoolTensor, True where a position is active
    Teacher first; include all modes up to 'card_track_max'; for cardinalities > card_track_max,
    sample 'sample_high_per_card' modes per degree.
    """
    rng = np.random.default_rng(seed)
    mode_lists: List[List[int]] = []

    mode_lists.append(teacher_S.tolist())

    for c in range(1, d + 1):
        all_c = enumerate_combos(d, c)
        if c == teacher_S.numel() and all_c.numel() > 0:
            teacher_tuple = tuple(teacher_S.tolist())
            keep = [row for row in all_c.tolist() if tuple(row) != teacher_tuple]
            all_c = torch.tensor(keep, dtype=torch.long) if len(keep) > 0 else torch.empty((0, c), dtype=torch.long)

        Nc = all_c.size(0)
        if Nc == 0: continue

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
        if key in seen: continue
        seen.add(key)
        if set(A.tolist()) != S_set:
            A_list.append(A)
    return A_list

def make_boolean_blocks(K: int, P: int, d: int, device: torch.device, seed: int) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.randint(0, 2, (K, P, d), generator=g, device=device, dtype=torch.int8)
    return X.to(torch.float32) * 2.0 - 1.0

def chi_of_modes_blocks_varcard(
    X_blocks: torch.Tensor,        # (K,P,d)
    modes_idx_pad: torch.Tensor,   # (M,k_max) long
    modes_mask: torch.Tensor,      # (M,k_max) bool
    chunk: int = 512
) -> torch.Tensor:
    """
    Compute chi_A(x) = prod_{i in A} x_i for variable-cardinality modes.
    Padded positions contribute multiplicative 1 (via mask).
    Returns (K,P,M) matching dtype of X_blocks.
    """
    K, P, d = X_blocks.shape
    M, k_max = modes_idx_pad.shape
    device = X_blocks.device
    chi = torch.empty((K, P, M), device=device, dtype=X_blocks.dtype)

    if chunk <= 0: chunk = M
    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        idx = modes_idx_pad[s:e].to(device)           # (mchunk,k_max)
        msk = modes_mask[s:e].to(device)              # (mchunk,k_max)

        idx_exp = idx.view(1, 1, e - s, k_max).expand(K, P, e - s, k_max)
        gathered = torch.gather(
            X_blocks.unsqueeze(2).expand(K, P, e - s, d),
            3,
            idx_exp
        )  # (K,P,mchunk,k_max)

        msk_exp = msk.view(1, 1, e - s, k_max).expand(K, P, e - s, k_max)
        ones = torch.ones_like(gathered)
        gathered = torch.where(msk_exp, gathered, ones)

        chi[:, :, s:e] = gathered.prod(dim=3)
    return chi

def activation(z: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "relu": return torch.relu(z)
    if kind == "tanh": return torch.tanh(z)
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

# ----------------------------- Moments per block -----------------------------

def compute_phi_J_Sigma_halves(
    w_blocks: torch.Tensor,   # (K,B,d)
    X1: torch.Tensor,         # (K,P1,d)
    X2: torch.Tensor,         # (K,P2,d)
    C1: torch.Tensor,         # (K,P1,M) chi
    C2: torch.Tensor,         # (K,P2,M)
    act: str
):
    # phi
    z1 = torch.einsum('kpd,kbd->kpb', X1, w_blocks)  # (K,P1,B)
    z2 = torch.einsum('kpd,kbd->kpb', X2, w_blocks)  # (K,P2,B)
    phi1 = activation(z1, act)                       # (K,P1,B)
    phi2 = activation(z2, act)                       # (K,P2,B)

    # J (same-half), Sigma
    P1, P2 = X1.shape[1], X2.shape[1]
    J1 = torch.einsum('kpb,kpm->kbm', phi1, C1) / float(P1)  # (K,B,M)
    J2 = torch.einsum('kpb,kpm->kbm', phi2, C2) / float(P2)  # (K,B,M)
    S1 = (phi1 * phi1).mean(dim=1)                           # (K,B)
    S2 = (phi2 * phi2).mean(dim=1)                           # (K,B)
    return phi1, phi2, J1, J2, S1, S2

# ----------------------------- logπ(w|D_k) and grad -----------------------------

def compute_logp_and_grad_w_blocks_indep(
    w_blocks: torch.Tensor, X_blocks: torch.Tensor, chi_blocks: torch.Tensor,
    m_blocks: torch.Tensor, kappa: float, mdl: ModelParams, saem: SAEMParams,
    backreact_lambda: float = 1.0   # keep the arg for API compatibility; we won't use it
):
    # g_w^2 = σ_w^2 / d  (σ_w is a std)
    gw2 = (mdl.sigma_w ** 2) / mdl.d
    # per-neuron variance: σ_a^2 / N^γ
    ga2 = (mdl.sigma_a ** 2) / (mdl.N ** mdl.gamma)
    # precision on 'a': 1/ga2 = N^γ / σ_a^2
    A   = 1.0 / ga2

    w_blocks = w_blocks.detach().requires_grad_(True)

    z   = torch.einsum('kpd,kbd->kpb', X_blocks, w_blocks)
    phi = activation(z, mdl.act)

    # J_A(w) for all tracked modes, including S at index 0
    J   = torch.einsum('kpb,kpm->kbm', phi, chi_blocks) / float(X_blocks.shape[1])

    # Σ(w) = E_x[φ(w·x)^2] — keep gradient!
    Sigma = (phi * phi).mean(dim=1)  # (K,B)

    # D(w) = κ^2 * (N^γ / σ_a^2) + Σ(w)
    D = (A * (kappa ** 2) + Sigma).clamp_min(saem.eps_D)

    # β(w;m) = (J_Y - Σ_B m_B J_B)/κ^2 ; for parity teacher: J_Y ≡ J_S (mode 0)
    JY   = J[:, :, 0]
    Jm   = torch.einsum('kbm,km->kb', J, m_blocks)
    Jdiff = JY - Jm
    beta = Jdiff / (kappa ** 2)

    # log π(w|m) up to additive const:
    #   - ||w||^2/(2 g_w^2) - 0.5 log D + 0.5 * (beta^2 / (1/D)) = prior - 0.5 log D + 0.5 * (Jdiff^2)/(κ^2 D)
    prior_term = -0.5 * (w_blocks * w_blocks).sum(dim=2) / gw2
    data_quad  = 0.5 * (Jdiff * Jdiff) / ( (kappa ** 2) * D )
    logp       = prior_term + data_quad - 0.5 * torch.log(D)

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

# ----------------------------- Matrix-free CG for (I + λI + N B) m = N b -----------------------------
def cg_solve_matfree(matvec_B, rhs, diag_B, N, lam, tol=1e-6, max_iter=200, verbose=False):
    device = rhs.device
    dtype  = rhs.dtype
    N_t   = torch.as_tensor(N,   dtype=dtype, device=device)
    lam_t = torch.as_tensor(lam, dtype=dtype, device=device)

    def apply_A(v):
        v = v.to(dtype)
        return (1.0 + lam_t) * v + N_t * matvec_B(v)  # <-- N_t here


    x  = torch.zeros_like(rhs, dtype=dtype, device=device)
    r  = rhs - apply_A(x)
    p  = r.clone()
    rs = (r*r).sum(dim=1, keepdim=True)

    rhs_norm = rhs.norm(dim=1, keepdim=True).clamp_min(1e-30)

    for _ in range(max_iter):
        Ap    = apply_A(p)
        denom = (p*Ap).sum(dim=1, keepdim=True).clamp_min(1e-30)
        alpha = rs / denom
        x     = x + alpha * p
        r     = r - alpha * Ap
        rs_new = (r*r).sum(dim=1, keepdim=True)
        if (rs_new.sqrt() <= tol * rhs_norm).all():
            break
        beta = rs_new / rs
        p    = r + beta * p
        rs   = rs_new
    return x


# ----------------------------- SAEM (E: SGLD, M: exact matrix-free CG) -----------------------------

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
        gw2 = mdl.sigma_w / mdl.d
        w_blocks = torch.randn(K, B, mdl.d, device=device) * math.sqrt(gw2)

        data_seed = _get(BASE, "data_seed", 0) + 10000 * (di + 1)
        X_blocks = make_boolean_blocks(K, P, mdl.d, device, seed=data_seed)
        chi_blocks = chi_of_modes_blocks_varcard(
            X_blocks, modes_idx_pad.to(device), modes_mask.to(device), chunk=min(512, Mtot)
        )  # (K,P,M)

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

        # E-step
        for slot in per_dev:
            slot["w_blocks"] = mcmc_sgld_w_blocks_indep(
                slot["w_blocks"], slot["X_blocks"], slot["chi_blocks"],
                slot["m_blocks"], kappa, mdl, saem, mcmc,
                backreact_lambda=lam_br
            )

        # per-iter sanity accumulators
        sane_list: List[Sanity] = []

        # M-step: exact matrix-free CG
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

            # moments per half
            phi1, phi2, J1, J2, S1, S2 = compute_phi_J_Sigma_halves(w_blocks, X1, X2, C1, C2, mdl.act)
            D1 = torch.clamp(Acoef * (kappa**2) + S1, min=saem.eps_D)  # (K,B)
            D2 = torch.clamp(Acoef * (kappa**2) + S2, min=saem.eps_D)  # (K,B)
            invD1, invD2 = 1.0 / D1, 1.0 / D2

            # b (cross-fit)
            J1_S = J1[:, :, 0]  # (K,B)
            J2_S = J2[:, :, 0]
            b12  = torch.einsum('kbm,kb->km', J1, (J2_S * invD2)) / float(Bsize)
            b21  = torch.einsum('kbm,kb->km', J2, (J1_S * invD1)) / float(Bsize)
            b_vec = 0.5 * (b12 + b21)  # (K,M)

            # --- build diag(B) correctly ---
            d1 = torch.einsum('kbm,kbm,kb->km', J1, J1, invD1) / float(Bsize)
            d2 = torch.einsum('kbm,kbm,kb->km', J2, J2, invD2) / float(Bsize)
            diag_B = 0.5 * (d1 + d2)  # (K, M)
            

            # --- B matvec (matrix-free); keep the 1/B scaling ---
            def matvec_B(v: torch.Tensor) -> torch.Tensor:
                v = v.to(J1.dtype)
                t1 = torch.einsum('kbm,km->kb', J1, v)             # (K,B)
                u1 = torch.einsum('kbm,kb->km', J1, t1 * invD1)    # (K,M)
                t2 = torch.einsum('kbm,km->kb', J2, v)             # (K,B)
                u2 = torch.einsum('kbm,kb->km', J2, t2 * invD2)    # (K,M)
                return 0.5 * (u1 + u2) / float(Bsize)

            # --- CG RHS + dtype alignment ---
            lam = saem.ridge_lambda
            rhs = (float(mdl.N) * b_vec).to(J1.dtype)  # <-- multiply by N
       # <-- make sure this line is NOT commented out
            diag_B_t = diag_B.to(J1.dtype)            # <-- use diag_B (not d_diag)

            m_star = cg_solve_matfree(
                matvec_B, rhs, diag_B_t, N=float(mdl.N), lam=lam,
                tol=BASE.get("cg_tol", 1e-5), max_iter=BASE.get("cg_max_iter", 200)
            )
            ms_1d = (mdl.N * b_vec[:, :1]) / (1.0 + saem.ridge_lambda + mdl.N * diag_B[:, :1])
            mS_from_CG = m_star[:, :1]
            print("check mS:", float(ms_1d.mean()), float(mS_from_CG.mean()))
            # (optional) one-time diagonal consistency check
            # with torch.no_grad():
            #     v_e0 = torch.zeros(Kb, diag_B.shape[1], device=device, dtype=J1.dtype); v_e0[:, 0] = 1.0
            #     col0 = matvec_B(v_e0)
            #     err  = (col0[:, 0] - diag_B[:, 0]).abs().mean()
            #     print(f"[debug] diag(B) consistency err: {err.item():.3e}")


            
            # SAEM averaging per block
            a_t = saem.a0 / (it + saem.t0)
            m_new = (1 - saem.damping_m * a_t) * m_blocks + saem.damping_m * a_t * m_star

            if saem.project_mS:
                m_new[:, 0] = m_new[:, 0].clamp(0.0, 1.0)
            if saem.m_noise_clip and saem.m_noise_clip > 0:
                m_new[:, 1:] = m_new[:, 1:].clamp(-saem.m_noise_clip, saem.m_noise_clip)

            slot["m_blocks"] = m_new

            # --------- Sanity on this device ----------
            JS_abs = 0.5 * (J1_S.abs().mean() + J2_S.abs().mean())
            back1 = torch.einsum('kbm,km->kb', J1, m_blocks).abs().mean()
            back2 = torch.einsum('kbm,km->kb', J2, m_blocks).abs().mean()
            sane = Sanity(
                bS=float(b_vec[:, 0].mean().detach().cpu()),
                BSS=float(diag_B[:, 0].mean().detach().cpu()),
                JS_abs=float(JS_abs.detach().cpu()),
                backreact_abs=float(0.5*(back1+back2).detach().cpu()),
                mnoise_l2=float(m_blocks[:, 1:].pow(2).sum(dim=1).sqrt().mean().detach().cpu())
            )
            sane_list.append(sane)

        # --- Collect trajectories every iteration ---
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

            # aggregate sanity
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
        "kappa": kappa, "P": P, "mode": "multi-quenched-fixedD-parallel-xfit-matfreeCG-varcard",
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
        chains_per_device=1024,
        mcmc_steps=400,
        mcmc_step_size=5e-3,
        mcmc_step_decay=0.999,
        langevin_sqrt2=True,
        grad_clip=1e5,
        clamp_w=10.0,
        # SAEM / quenched runs
        a0=1.0, t0=150.0, damping_m=3.0,
        eps_D=1e-9, print_every=5,
        ridge_lambda=1e-3,
        blocks=1,
        resample_blocks_every=0,
        crossfit_shuffle_each_iter=False,
        # seeds
        teacher_seed=0, noise_seed=1, data_seed=0,
        m_init=0.5,
        project_mS=True,
        m_noise_clip=3.0,
        # CG controls
        cg_max_iter=200,
        cg_tol=1e-5,
        # variable-cardinality tracking
        card_track_max=4,
        sample_high_per_card=2000,
        # backreaction ramp for E-step (optional)
        backreact_ramp_iters=100,
        backreact_ramp_pow=1.0,
    )

    save_dir = "/home/goring/mean_field_langevin/MCMC_finiteP/results/matfree_cg_varcard"
    os.makedirs(save_dir, exist_ok=True)
    run_tag_prefix = time.strftime("%Y%m%d_%H%M%S")

    kappa_list = [1e-3]
    P_list     = [2000]
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
