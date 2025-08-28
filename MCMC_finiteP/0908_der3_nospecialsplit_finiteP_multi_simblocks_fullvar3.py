# 0908_der3_nospecialsplit_finiteP_multi_simblocks_fullvar4.py
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
    Includes teacher first, then all modes up to 'card_track_max',
    then for cardinalities > card_track_max, a random sample per degree.
    """
    rng = np.random.default_rng(seed)
    mode_lists: List[List[int]] = []

    # teacher first
    mode_lists.append(teacher_S.tolist())

    # per-cardinality
    for c in range(1, d + 1):
        all_c = enumerate_combos(d, c)  # (Nc, c)
        # drop teacher if same cardinality and exactly equal
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

    # fallback: ensure at least the teacher
    if len(mode_lists) == 0:
        mode_lists = [teacher_S.tolist()]

    # pad to k_max with mask
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

        idx_exp = idx.view(1, 1, e - s, k_max).expand(K, P, e - s, k_max)  # (K,P,mchunk,k_max)
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

def compute_J_Sigma_blocks_indep(
    w_blocks: torch.Tensor,   # (K,B,d)
    X_blocks: torch.Tensor,   # (K,P,d)
    chi_blocks: torch.Tensor, # (K,P,M)
    act: str
):
    z = torch.einsum('kpd,kbd->kpb', X_blocks, w_blocks)  # (K,P,B)
    phi = activation(z, act)                              # (K,P,B)
    J = torch.einsum('kpb,kpm->kbm', phi, chi_blocks) / float(X_blocks.shape[1])  # (K,B,M)
    Sigma = (phi * phi).mean(dim=1)                       # (K,B)
    return J, Sigma

# ----------------------------- logπ(w|D_k) and grad -----------------------------

def compute_logp_and_grad_w_blocks_indep(
    w_blocks: torch.Tensor, X_blocks: torch.Tensor, chi_blocks: torch.Tensor,
    m_blocks: torch.Tensor, kappa: float, mdl: ModelParams, saem: SAEMParams,
    backreact_lambda: float = 1.0
):
    gw2 = mdl.sigma_w / mdl.d
    ga2 = mdl.sigma_a / (mdl.N ** mdl.gamma)
    A   = 1.0 / ga2

    w_blocks = w_blocks.detach().requires_grad_(True)
    J, Sigma = compute_J_Sigma_blocks_indep(w_blocks, X_blocks, chi_blocks, mdl.act)  # (K,B,M),(K,B)
    J_beta = J[:, :, 0] - backreact_lambda * torch.einsum('kbm,km->kb', J, m_blocks)  # (K,B)
    D = A * (kappa**2) + Sigma
    D = torch.clamp(D, min=saem.eps_D)

    prior_term = -0.5 * (w_blocks * w_blocks).sum(dim=2) / gw2
    data_term  = -0.5 * torch.log(D) + 0.5 * (J_beta * J_beta) / ((kappa**2) * D)
    logp = prior_term + data_term

    grad = torch.autograd.grad(logp.sum(), w_blocks, create_graph=False, retain_graph=False)[0]
    grad = torch.where(torch.isfinite(grad), grad, torch.zeros_like(grad))
    return logp.detach(), grad.detach()

# ----------------------------- SGLD (independent per block) -----------------------------

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

        if mcmc.langevin_sqrt2:
            noise = torch.randn_like(w_blocks)
            w_blocks = w_blocks + step * grad + noise * math.sqrt(2.0 * step)
        else:
            noise = torch.randn_like(w_blocks)
            w_blocks = w_blocks + 0.5 * step * grad + noise * math.sqrt(step)

        if mcmc.clamp_w:
            w_blocks = torch.clamp(w_blocks, -mcmc.clamp_w, mcmc.clamp_w)
        step *= mcmc.step_decay
    return w_blocks.detach()

# ----------------------------- Robust batched SPD solve -----------------------------

def batched_spd_solve_chol(R: torch.Tensor, y: torch.Tensor, jitter0: float = 1e-10, max_tries: int = 6) -> torch.Tensor:
    """
    Solve (per K) R z = y with R SPD (or nearly), using adaptive jitter.
    R: (K,r,r), y: (K,r)  -> z: (K,r)
    """
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
            return z
        except Exception:
            jitter *= 10.0
    z = torch.linalg.solve(R + jitter * eye, y.unsqueeze(2)).squeeze(2)
    return z

# ----------------------------- SAEM (diag + low-rank residual) -----------------------------

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

    # sketch + ramp knobs
    r = int(_get(BASE, "sketch_rank_r", 128))
    sketch_seed = int(_get(BASE, "sketch_seed", 123))
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
        # fallback: teacher + random same-k noise
        A_noise = make_k_subsets(mdl.d, mdl.k, M_noise, teacher_S, seed=_get(BASE, "noise_seed", 1))
        modes = [teacher_S] + A_noise
        k_max = teacher_S.numel()
        M = len(modes)
        modes_idx_pad = torch.stack(modes).long()
        modes_mask = torch.ones((M, k_max), dtype=torch.bool)

    Mtot, k_max = modes_idx_pad.shape

    # per-device state
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

        # one m-vector PER BLOCK (quenched): (K,M)
        m_blocks = torch.zeros(K, Mtot, device=device, dtype=torch.float32)
        m_init = _get(BASE, "m_init", 0.5)
        m_blocks[:, 0] = m_init  # teacher first

        # independent Ω for the two halves
        g1 = torch.Generator(device=device).manual_seed(sketch_seed + 17)
        g2 = torch.Generator(device=device).manual_seed(sketch_seed + 23)
        Omega1 = torch.randn(B, r, generator=g1, device=device) / math.sqrt(r)
        Omega2 = torch.randn(B, r, generator=g2, device=device) / math.sqrt(r)

        per_dev.append({
            "device": device,
            "w_blocks": w_blocks,
            "X_blocks": X_blocks,
            "chi_blocks": chi_blocks,
            "m_blocks": m_blocks,
            "Omega1": Omega1,
            "Omega2": Omega2,
        })

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

        # Sanity accumulators across devices for this iter
        acc_bS, acc_BSS, acc_JS_abs, acc_backreact_abs, acc_mnoise_l2 = [], [], [], [], []

        # M-step: diag + low-rank residual (Woodbury)
        for slot in per_dev:
            device = slot["device"]
            Xb, chib, w_blocks, m_blocks = slot["X_blocks"], slot["chi_blocks"], slot["w_blocks"], slot["m_blocks"]
            Omega1, Omega2 = slot["Omega1"], slot["Omega2"]
            Kb, Ptot, _ = Xb.shape
            Bsize = w_blocks.shape[1]

            if saem.crossfit_shuffle_each_iter:
                perm = torch.randperm(Ptot, device=device)
                Xb = Xb[:, perm, :]
                chib = chib[:, perm, :]

            s = Ptot // 2
            X1, X2 = Xb[:, :s, :], Xb[:, s:, :]
            C1, C2 = chib[:, :s, :], chib[:, s:, :]

            J1, S1 = compute_J_Sigma_blocks_indep(w_blocks, X1, C1, mdl.act)  # (K,B,M),(K,B)
            J2, S2 = compute_J_Sigma_blocks_indep(w_blocks, X2, C2, mdl.act)
            D1 = torch.clamp(Acoef * (kappa**2) + S1, min=saem.eps_D)
            D2 = torch.clamp(Acoef * (kappa**2) + S2, min=saem.eps_D)
            invD1, invD2 = 1.0 / D1, 1.0 / D2

            # ----- CROSS-FIT b (BUG FIX) -----
            J1_S = J1[:, :, 0]
            J2_S = J2[:, :, 0]
            # b12: J1_A with (J2_S * invD2)
            b12  = torch.einsum('kbm,kb->km', J1, (J2_S * invD2)) / float(Bsize)
            # b21: J2_A with (J1_S * invD1)
            b21  = torch.einsum('kbm,kb->km', J2, (J1_S * invD1)) / float(Bsize)
            b_vec = 0.5 * (b12 + b21)                                   # (K,M)

            # ----- SAME-HALF diagonal B -----
            d1 = torch.einsum('kbm,kbm,kb->km', J1, J1, invD1) / float(Bsize)
            d2 = torch.einsum('kbm,kbm,kb->km', J2, J2, invD2) / float(Bsize)
            d_diag = 0.5 * (d1 + d2)   # (K,M)

            # ----- Low-rank sketches (independent Ω) -----
            sqrt_invD1 = invD1.sqrt().unsqueeze(2)          # (K,B,1)
            sqrt_invD2 = invD2.sqrt().unsqueeze(2)          # (K,B,1)
            Y1 = J1 * sqrt_invD1                            # (K,B,M)
            Y2 = J2 * sqrt_invD2                            # (K,B,M)

            S1 = torch.einsum('kbm,br->kmr', Y1, Omega1) / math.sqrt(Bsize)  # (K,M,r)
            S2 = torch.einsum('kbm,br->kmr', Y2, Omega2) / math.sqrt(Bsize)  # (K,M,r)
            S = torch.cat([S1, S2], dim=2) * (1.0 / math.sqrt(2.0))          # (K,M,2r)

            # diagonal residual after removing sketched variance
            diag_S = (S.double() ** 2).sum(dim=2).to(torch.float32)          # (K,M)
            d_res  = torch.clamp(d_diag - diag_S, min=0.0)                   # (K,M)

            lam = saem.ridge_lambda
            A0_diag = (1.0 + lam) + mdl.N * d_res                            # (K,M)
            A0_inv  = 1.0 / A0_diag                                          # (K,M)

            # SAinv: (K,2r,M) = (S^T) * A0^{-1}  (broadcast over M)
            SAinv = S.transpose(1, 2) * A0_inv.unsqueeze(1)                  # (K,2r,M)

            # R = N * (S^T A0^{-1} S)  (K,2r,2r)
            R = torch.bmm(SAinv.to(torch.float64), S.to(torch.float64)) * mdl.N
            R = 0.5 * (R + R.transpose(1, 2))                                # symmetrize

            # rhs: S^T A0^{-1} (N b)
            ST_Ainv_b = torch.einsum('krm,km->kr', SAinv, (mdl.N * b_vec))   # (K,2r)

            # solve R z = S^T A0^{-1} (N b)
            z = batched_spd_solve_chol(R.to(torch.float64), ST_Ainv_b.to(torch.float64))  # (K,2r)

            # m* = A0^{-1} [N b - N * S z]
            Sz = torch.einsum('kmr,kr->km', S.to(torch.float64), z)          # (K,M)
            m_star = (A0_inv.to(torch.float64) * (mdl.N * b_vec.to(torch.float64) - mdl.N * Sz)).to(torch.float32)

            # SAEM averaging per block
            a_t = saem.a0 / (it + saem.t0)
            m_new = (1 - saem.damping_m * a_t) * m_blocks + saem.damping_m * a_t * m_star

            # constraints
            if saem.project_mS:
                m_new[:, 0] = m_new[:, 0].clamp(0.0, 1.0)
            if saem.m_noise_clip and saem.m_noise_clip > 0:
                m_new[:, 1:] = m_new[:, 1:].clamp(-saem.m_noise_clip, saem.m_noise_clip)

            # -------- Sanity metrics on this device --------
            acc_bS.append(b_vec[:, 0].mean().detach().float().cpu())
            acc_BSS.append(d_diag[:, 0].mean().detach().float().cpu())

            # |J_S| vs |sum_A m_A J_A| (use same-half, averaged halves)
            back1 = torch.einsum('kbm,km->kb', J1, m_blocks).abs().mean()
            js1   = J1[:, :, 0].abs().mean()
            back2 = torch.einsum('kbm,km->kb', J2, m_blocks).abs().mean()
            js2   = J2[:, :, 0].abs().mean()
            acc_backreact_abs.append(0.5 * (back1 + back2).detach().float().cpu())
            acc_JS_abs.append(0.5 * (js1 + js2).detach().float().cpu())

            mnoise_l2 = m_blocks[:, 1:].pow(2).sum(dim=1).sqrt().mean()
            acc_mnoise_l2.append(mnoise_l2.detach().float().cpu())

            slot["m_blocks"] = m_new

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

        # Logging (means & stds across blocks) + sanity checks
        if it % saem.print_every == 0 or it == 1:
            ge_all = 0.5 * (1.0 - mS_all.numpy())**2

            bS_mean = float(torch.stack(acc_bS).mean().item()) if len(acc_bS) else 0.0
            BSS_mean = float(torch.stack(acc_BSS).mean().item()) if len(acc_BSS) else 0.0
            JS_abs_mean = float(torch.stack(acc_JS_abs).mean().item()) if len(acc_JS_abs) else 0.0
            backreact_abs_mean = float(torch.stack(acc_backreact_abs).mean().item()) if len(acc_backreact_abs) else 0.0
            m_noise_l2_mean = float(torch.stack(acc_mnoise_l2).mean().item()) if len(acc_mnoise_l2) else 0.0

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
        "kappa": kappa, "P": P, "mode": "multi-quenched-fixedD-parallel-xfit-diag+lowrank-varcard",
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
        opt_steps=2200,
        # SGLD
        chains_per_device=1024,
        mcmc_steps=400,
        mcmc_step_size=5e-3,
        mcmc_step_decay=0.999,
        langevin_sqrt2=True,
        grad_clip=1e5,
        clamp_w=10.0,
        # SAEM / quenched runs
        a0=1.0, t0=200.0, damping_m=3.0,
        eps_D=1e-9, print_every=100,
        ridge_lambda=1e-4,
        blocks=1,
        resample_blocks_every=0,
        crossfit_shuffle_each_iter=True,
        # seeds
        teacher_seed=0, noise_seed=1, data_seed=0,
        m_init=0.5,
        project_mS=True,
        m_noise_clip=3.0,
        # ---- low-rank sketch ----
        sketch_rank_r=128,
        sketch_seed=123,
        # ---- back-reaction ramp ----
        backreact_ramp_iters=400,
        backreact_ramp_pow=1.0,
        # ---- variable-cardinality tracking ----
        card_track_max=4,             # include ALL modes up to degree 4
        sample_high_per_card=2000,    # and sample per degree > 4
    )

    save_dir = "/home/goring/mean_field_langevin/MCMC_finiteP/results/2208_grid_testpaul"
    os.makedirs(save_dir, exist_ok=True)
    run_tag_prefix = time.strftime("%Y%m%d_%H%M%S")

    kappa_list = [1e-3]
    P_list     =  [10,400,800,1500,2500]
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
