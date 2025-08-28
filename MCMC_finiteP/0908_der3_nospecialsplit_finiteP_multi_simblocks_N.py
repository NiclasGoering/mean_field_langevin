# 0908_der3_nospecialsplit_finiteP_multi_simblocks_N.py
import os, json, time, math, random
from typing import Dict, List, Tuple
from dataclasses import dataclass

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

def make_k_subsets(d: int, k: int, M: int, S: torch.Tensor, seed: int = 1) -> List[torch.Tensor]:
    """Sample M random k-subsets (skip exact equality with S)."""
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
    """Return X of shape (K,P,d) with entries in {-1,+1}."""
    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.randint(0, 2, (K, P, d), generator=g, device=device, dtype=torch.int8)
    return X.to(torch.float32) * 2.0 - 1.0

def chi_of_modes_blocks(X_blocks: torch.Tensor, modes_idx: torch.Tensor, chunk: int = 512) -> torch.Tensor:
    """Compute chi_A for all blocks/modes. Returns (K,P,M)."""
    K, P, d = X_blocks.shape
    M, k = modes_idx.shape
    device = X_blocks.device
    chi = torch.empty((K, P, M), device=device, dtype=X_blocks.dtype)
    if chunk <= 0: chunk = M
    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        idx = modes_idx[s:e].to(device)                  # (mchunk,k)
        idx_exp = idx.view(1,1,e-s,k).expand(K,P,e-s,k)  # (K,P,mchunk,k)
        gathered = torch.gather(X_blocks.unsqueeze(2).expand(K,P,e-s,d), 3, idx_exp)
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
    a0: float = 0.5
    t0: float = 100.0
    damping_m: float = 1.0
    eps_D: float = 1e-8           # robust floor for Deff
    print_every: int = 20
    project_mS: bool = True
    m_noise_clip: float = 3.0
    ridge_lambda: float = 1e-5
    blocks: int = 32
    resample_blocks_every: int = 0
    crossfit_shuffle_each_iter: bool = True
    # χ update controls
    chi_a0: float = 0.1
    chi_t0: float = 200.0
    eig_min_chi: float = 1e-8
    # χ guard: keep Deff away from the floor by margin tau
    chi_guard_margin: float = 0.10  # tau in (0,1); we enforce Deff >= tau*(A k^2 + Sigma_min)

# ----------------------------- Moments per block -----------------------------

def compute_J_Sigma_blocks_indep(
    w_blocks: torch.Tensor,   # (K,B,d)
    X_blocks: torch.Tensor,   # (K,P,d)
    chi_blocks_inputs: torch.Tensor, # (K,P,M)
    act: str
):
    z = torch.einsum('kpd,kbd->kpb', X_blocks, w_blocks)  # (K,P,B)
    phi = activation(z, act)                              # (K,P,B)
    J = torch.einsum('kpb,kpm->kbm', phi, chi_blocks_inputs) / float(X_blocks.shape[1])  # (K,B,M)
    Sigma = (phi * phi).mean(dim=1)                       # (K,B)
    return J, Sigma

# ----------------------------- PSD projection & safe solver -----------------------------

def project_psd(mat: torch.Tensor, eig_floor: float) -> torch.Tensor:
    mat = 0.5 * (mat + mat.transpose(1, 2))
    evals, evecs = torch.linalg.eigh(mat)
    evals = torch.clamp(evals, min=eig_floor)
    return evecs @ torch.diag_embed(evals) @ evecs.transpose(1, 2)

def safe_spd_solve(lhs: torch.Tensor,
                   rhs: torch.Tensor,
                   I: torch.Tensor,
                   base_ridge: float = 1e-5,
                   max_tries: int = 3,
                   eig_min_clip: float = 1e-8) -> torch.Tensor:
    K, M, _ = lhs.shape
    lhs = 0.5 * (lhs + lhs.transpose(1, 2))
    ridge = base_ridge
    for _ in range(max_tries):
        lhs_try = lhs + ridge * I
        try:
            L = torch.linalg.cholesky(lhs_try)
            return torch.cholesky_solve(rhs, L)
        except Exception:
            ridge *= 10.0
    evals, evecs = torch.linalg.eigh(lhs)
    evals = torch.clamp(evals, min=eig_min_clip)
    lhs_psd = evecs @ torch.diag_embed(evals) @ evecs.transpose(1,2)
    try:
        return torch.linalg.solve(lhs_psd, rhs)
    except Exception:
        lhs_pinv = torch.linalg.pinv(lhs_psd)
        return lhs_pinv @ rhs

# ----------------------------- logπ(w|D_k) + grad with 1/N (uses Deff) -----------------------------

def compute_logp_and_grad_w_blocks_indep(
    w_blocks: torch.Tensor, X_blocks: torch.Tensor, chi_inputs: torch.Tensor,
    m_blocks: torch.Tensor, chi_susc: torch.Tensor,
    kappa: float, mdl: ModelParams, saem: SAEMParams
):
    gw2 = mdl.sigma_w / mdl.d
    A   = (mdl.N ** mdl.gamma) / mdl.sigma_a

    w_blocks = w_blocks.detach().requires_grad_(True)
    J, Sigma = compute_J_Sigma_blocks_indep(w_blocks, X_blocks, chi_inputs, mdl.act)  # (K,B,M),(K,B)

    J_beta = J[:, :, 0] - torch.einsum('kbm,km->kb', J, m_blocks)                     # (K,B)

    zeta = torch.einsum('kbm,kmn,kbn->kb', J, chi_susc, J)
    Deff = A * (kappa**2) + Sigma - (1.0 / mdl.N) * zeta
    Deff = torch.clamp(Deff, min=saem.eps_D)

    prior_term = -0.5 * (w_blocks * w_blocks).sum(dim=2) / gw2                        # (K,B)
    data_term  = -0.5 * torch.log(Deff) + 0.5 * (J_beta * J_beta) / ((kappa**2) * Deff)  # (K,B)
    logp = prior_term + data_term

    grad = torch.autograd.grad(logp.sum(), w_blocks, create_graph=False, retain_graph=False)[0]
    grad = torch.where(torch.isfinite(grad), grad, torch.zeros_like(grad))
    return logp.detach(), grad.detach()

# ----------------------------- SGLD over w (independent per block) -----------------------------

def mcmc_sgld_w_blocks_indep(
    w_blocks: torch.Tensor, X_blocks: torch.Tensor, chi_inputs: torch.Tensor,
    m_blocks: torch.Tensor, chi_susc: torch.Tensor,
    kappa: float, mdl: ModelParams, saem: SAEMParams, mcmc: MCMCParams
):
    step = mcmc.step_size
    for _ in range(mcmc.n_steps):
        _, grad = compute_logp_and_grad_w_blocks_indep(
            w_blocks, X_blocks, chi_inputs, m_blocks.to(w_blocks.device),
            chi_susc, kappa, mdl, saem
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

# ----------------------------- SAEM (K independent quenched runs, with 1/N + χ-guard) -----------------------------

def saem_multimode_quenched_fixedD_multi_runs_1N(
    kappa: float, P: int, M_noise: int, devices: List[int],
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
        n_steps=_get(BASE, "mcmc_steps", 300),
        step_size=_get(BASE, "mcmc_step_size", 5e-3),
        step_decay=_get(BASE, "mcmc_step_decay", 0.999),
        langevin_sqrt2=_get(BASE, "langevin_sqrt2", True),
        grad_clip=_get(BASE, "grad_clip", 1e5),
        clamp_w=_get(BASE, "clamp_w", 10.0),
    )
    saem = SAEMParams(
        max_iters=_get(BASE, "opt_steps", 800),
        a0=_get(BASE, "a0", 0.5),
        t0=_get(BASE, "t0", 100.0),
        damping_m=_get(BASE, "damping_m", 1.0),
        eps_D=_get(BASE, "eps_D", 1e-8),
        print_every=_get(BASE, "print_every", 20),
        project_mS=_get(BASE, "project_mS", True),
        m_noise_clip=_get(BASE, "m_noise_clip", 3.0),
        ridge_lambda=_get(BASE, "ridge_lambda", 1e-5),
        blocks=_get(BASE, "blocks", 32),
        resample_blocks_every=_get(BASE, "resample_blocks_every", 0),
        crossfit_shuffle_each_iter=_get(BASE, "crossfit_shuffle_each_iter", True),
        chi_a0=_get(BASE, "chi_a0", 0.1),
        chi_t0=_get(BASE, "chi_t0", 200.0),
        eig_min_chi=_get(BASE, "eig_min_chi", 1e-8),
        chi_guard_margin=_get(BASE, "chi_guard_margin", 0.10),
    )

    # teacher + noise modes
    S = make_parity_indices(mdl.d, mdl.k, seed=_get(BASE, "teacher_seed", 0))
    A_noise = make_k_subsets(mdl.d, mdl.k, M_noise, S, seed=_get(BASE, "noise_seed", 1))
    modes = [S] + A_noise
    modes_idx = torch.stack(modes).long()   # (Mtot,k)
    Mtot = modes_idx.shape[0]

    # per-device state
    per_dev = []
    for di in devices if len(devices) > 0 else [-1]:
        device = torch.device(f"cuda:{di}") if di >= 0 and torch.cuda.is_available() else torch.device("cpu")
        K, B = saem.blocks, mcmc.n_chains_per_device
        gw2 = mdl.sigma_w / mdl.d
        w_blocks = torch.randn(K, B, mdl.d, device=device) * math.sqrt(gw2)

        data_seed = _get(BASE, "data_seed", 0) + 10000 * (di + 1)
        X_blocks = make_boolean_blocks(K, P, mdl.d, device, seed=data_seed)     # FIXED datasets
        chi_inputs = chi_of_modes_blocks(X_blocks, modes_idx, chunk=min(512, Mtot))

        # one m-vector PER BLOCK (quenched): (K,M)
        m_blocks = torch.zeros(K, Mtot, device=device, dtype=torch.float32)
        m_init = _get(BASE, "m_init", 0.5)
        m_blocks[:, 0] = m_init

        # susceptibility χ per block (K,M,M), start PSD at zero
        chi_susc = torch.zeros(K, Mtot, Mtot, device=device, dtype=torch.float32)

        per_dev.append({
            "device": device,
            "w_blocks": w_blocks,
            "X_blocks": X_blocks,
            "chi_inputs": chi_inputs,
            "m_blocks": m_blocks,
            "chi_susc": chi_susc,
        })

    Acoef = (mdl.N**mdl.gamma) / mdl.sigma_a
    t_start = time.time()
    history = []

    for it in range(1, saem.max_iters + 1):
        # E-step: SGLD independently per block with its own m_k, χ_k, and fixed D_k
        for slot in per_dev:
            slot["w_blocks"] = mcmc_sgld_w_blocks_indep(
                slot["w_blocks"], slot["X_blocks"], slot["chi_inputs"],
                slot["m_blocks"], slot["chi_susc"],
                kappa, mdl, saem, mcmc
            )

        # M-step: per-block cross-fitted linear solve (using Deff)
        for slot in per_dev:
            device = slot["device"]
            Xb, chib_in, w_blocks = slot["X_blocks"], slot["chi_inputs"], slot["w_blocks"]
            m_blocks, chi_susc = slot["m_blocks"], slot["chi_susc"]
            Kb, Ptot, _ = Xb.shape

            # Optional shuffle along P for crossfit (still within each fixed dataset)
            if saem.crossfit_shuffle_each_iter:
                perm = torch.randperm(Ptot, device=device)
                Xb = Xb[:, perm, :]
                chib_in = chib_in[:, perm, :]

            s = Ptot // 2
            X1, X2 = Xb[:, :s, :], Xb[:, s:, :]
            C1, C2 = chib_in[:, :s, :], chib_in[:, s:, :]

            # Moments on halves
            J1, S1 = compute_J_Sigma_blocks_indep(w_blocks, X1, C1, mdl.act)  # (K,B,M),(K,B)
            J2, S2 = compute_J_Sigma_blocks_indep(w_blocks, X2, C2, mdl.act)

            # Raw Deff (for diagnostics before guard)
            z1_raw = torch.einsum('kbm,kmn,kbn->kb', J1, chi_susc, J1)
            z2_raw = torch.einsum('kbm,kmn,kbn->kb', J2, chi_susc, J2)
            D1_raw = Acoef * (kappa**2) + S1 - (1.0 / mdl.N) * z1_raw
            D2_raw = Acoef * (kappa**2) + S2 - (1.0 / mdl.N) * z2_raw

            # ----------- χ update (SAEM + PSD + GUARD) -----------
            # Posterior moments of a|w on half 1
            Jbeta1 = J1[:,:,0] - torch.einsum('kbm,km->kb', J1, m_blocks)     # (K,B)
            alpha1 = Acoef + (torch.clamp(D1_raw, min=saem.eps_D) / (kappa**2))
            var_a1 = 1.0 / alpha1
            mu_a1  = (Jbeta1 / (kappa**2)) / alpha1
            fac = (mdl.N / (kappa**2))
            chi_new = fac * torch.einsum('kbm,kbn,kb->kmn', J1, J1, (var_a1 + mu_a1**2)) / float(J1.shape[1])  # (K,M,M)

            a_t_chi = saem.chi_a0 / (it + saem.chi_t0)
            chi_prop = (1 - a_t_chi) * chi_susc + a_t_chi * chi_new
            chi_prop = project_psd(chi_prop, saem.eig_min_chi)

            # ---- GUARD: scale χ to ensure Deff >= tau*(A*k^2 + Sigma_min) on BOTH halves ----
            tau = saem.chi_guard_margin  # 0.10 by default
            # Evaluate zeta with proposed χ on both halves
            z1_eval = torch.einsum('kbm,kmn,kbn->kb', J1, chi_prop, J1)  # (K,B)
            z2_eval = torch.einsum('kbm,kmn,kbn->kb', J2, chi_prop, J2)  # (K,B)
            cap1 = mdl.N * (Acoef * (kappa**2) + S1) * (1.0 - tau)       # (K,B)
            cap2 = mdl.N * (Acoef * (kappa**2) + S2) * (1.0 - tau)       # (K,B)

            # worst-case over chains b
            zmax_k = torch.maximum(z1_eval.max(dim=1).values, z2_eval.max(dim=1).values)  # (K,)
            cap_k  = torch.minimum(cap1.min(dim=1).values, cap2.min(dim=1).values)        # (K,)

            need_scale = zmax_k > cap_k
            # default s_k = 1; where needed, s_k = cap/zmax
            eps = 1e-12
            s_k = torch.ones_like(zmax_k)
            s_k = torch.where(need_scale, cap_k / (zmax_k + eps), s_k)
            s_k = s_k.clamp(min=0.0, max=1.0).view(Kb,1,1)

            chi_guarded = chi_prop * s_k
            chi_guarded = project_psd(chi_guarded, saem.eig_min_chi)  # keep PSD after scaling
            slot["chi_susc"] = chi_guarded

            # Now build guarded Deff for M-step linear stats
            z1 = torch.einsum('kbm,kmn,kbn->kb', J1, chi_guarded, J1)
            z2 = torch.einsum('kbm,kmn,kbn->kb', J2, chi_guarded, J2)
            D1 = torch.clamp(Acoef * (kappa**2) + S1 - (1.0 / mdl.N) * z1, min=saem.eps_D)
            D2 = torch.clamp(Acoef * (kappa**2) + S2 - (1.0 / mdl.N) * z2, min=saem.eps_D)
            invD1, invD2 = 1.0 / D1, 1.0 / D2

            # B_k: average of in-sample quadratics per half
            B1 = torch.einsum('kbm,kbn,kb->kmn', J1, J1, invD1) / float(J1.shape[1])  # (K,M,M)
            B2 = torch.einsum('kbm,kbn,kb->kmn', J2, J2, invD2) / float(J2.shape[1])  # (K,M,M)
            Bk = 0.5 * (B1 + B2)   # (K,M,M)

            # b_k: cross-fitted (disjoint halves)
            w12 = (J2[:, :, 0] * invD1)                                           # (K,B)
            w21 = (J1[:, :, 0] * invD2)                                           # (K,B)
            b12 = torch.einsum('kbm,kb->km', J1, w12) / float(J1.shape[1])        # (K,M)
            b21 = torch.einsum('kbm,kb->km', J2, w21) / float(J2.shape[1])        # (K,M)
            bk = 0.5 * (b12 + b21)                                                # (K,M)

            # Solve per block: (I + N Bk) m_k* = N bk
            I = torch.eye(Mtot, device=device, dtype=torch.float32).expand(Kb, Mtot, Mtot)
            lhs = I + mdl.N * Bk
            rhs = (mdl.N * bk).unsqueeze(2)

            m_star = safe_spd_solve(lhs, rhs, I,
                                    base_ridge=saem.ridge_lambda,
                                    max_tries=3,
                                    eig_min_clip=1e-8).squeeze(2)

            # SAEM averaging per block
            a_t = saem.a0 / (it + saem.t0)
            m_new = (1 - saem.damping_m * a_t) * m_blocks + saem.damping_m * a_t * m_star

            # constraints
            if saem.project_mS:
                m_new[:, 0] = m_new[:, 0].clamp(0.0, 1.0)
            if saem.m_noise_clip and saem.m_noise_clip > 0:
                m_new[:, 1:] = m_new[:, 1:].clamp(-saem.m_noise_clip, saem.m_noise_clip)

            slot["m_blocks"] = m_new

        # Logging (means & stds across blocks)
        if it % saem.print_every == 0 or it == 1:
            with torch.no_grad():
                mS_all = torch.cat([slot["m_blocks"][:, 0].detach().float().cpu() for slot in per_dev], dim=0)
                noise_rms = []
                for slot in per_dev:
                    mm = slot["m_blocks"]
                    if mm.shape[1] > 1:
                        noise_rms.append(mm[:, 1:].pow(2).mean(dim=1).sqrt().detach().float().cpu())
                noise_rms = torch.cat(noise_rms, dim=0) if len(noise_rms) > 0 else torch.tensor([0.0])
                ge_all = 0.5 * (1.0 - mS_all.numpy())**2

                # Diagnostics: Deff min before/after guard on first device/half
                slot0 = per_dev[0]
                Kb, Ptot, _ = slot0["X_blocks"].shape
                s = max(1, Ptot//2)
                Jd, Sd = compute_J_Sigma_blocks_indep(slot0["w_blocks"], slot0["X_blocks"][:, :s, :], slot0["chi_inputs"][:, :s, :], mdl.act)
                z_d = torch.einsum('kbm,kmn,kbn->kb', Jd, slot0["chi_susc"], Jd)
                Deff_diag = (Acoef * (kappa**2) + Sd - (1.0 / mdl.N) * z_d).min().item()
                Deff_diag = float(max(Deff_diag, saem.eps_D))

                dt = time.time() - t_start
                msg = {
                    "iter": it, "kappa": kappa, "P": P,
                    "blocks": sum(_get(BASE, "blocks", 32) for _ in per_dev),
                    "m_S_mean": float(mS_all.mean().item()),
                    "m_S_std": float(mS_all.std(unbiased=False).item()),
                    "gen_err_mean": float(ge_all.mean()),
                    "gen_err_std": float(ge_all.std()),
                    "m_noise_rms_mean": float(noise_rms.mean().item()) if noise_rms.numel() > 1 else 0.0,
                    "Deff_min_guarded": Deff_diag,
                    "time_s": round(dt, 2)
                }
                print(json.dumps(msg))

    # Snapshot & save
    with torch.no_grad():
        mS_all = torch.cat([slot["m_blocks"][:, 0].detach().float().cpu() for slot in per_dev], dim=0)
        noise_rms = []
        for slot in per_dev:
            mm = slot["m_blocks"]
            if mm.shape[1] > 1:
                noise_rms.append(mm[:, 1:].pow(2).mean(dim=1).sqrt().detach().float().cpu())
        noise_rms = torch.cat(noise_rms, dim=0) if len(noise_rms) > 0 else torch.tensor([0.0])
        ge_all = 0.5 * (1.0 - mS_all.numpy())**2

    result = {
        "kappa": kappa, "P": P, "mode": "multi-quenched-fixedD-parallel-xfit-1N-guarded",
        "blocks_total": int(mS_all.numel()),
        "mS_mean": float(mS_all.mean().item()),
        "mS_std": float(mS_all.std(unbiased=False).item()),
        "gen_err_mean": float(ge_all.mean()),
        "gen_err_std": float(ge_all.std()),
        "m_noise_rms_mean": float(noise_rms.mean().item()) if noise_rms.numel() > 1 else 0.0,
        "BASE": BASE
    }

    os.makedirs(SAVE_DIR, exist_ok=True)
    tag = run_tag if run_tag else f"kappa_{kappa:.6f}_P_{P}"
    out_path = os.path.join(SAVE_DIR, f"quenched_parallel_xfit_1N_guarded_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[saved] {out_path}")
    return result

# ----------------------------- Grid runner (κ, P) -----------------------------

def run_grid_1N(
    kappa_list: List[float], P_list: List[int], M_noise: int,
    BASE: Dict, devices: List[int], SAVE_DIR: str, run_tag_prefix: str
):
    all_paths = []
    for kap in kappa_list:
        for P in P_list:
            tag = f"{run_tag_prefix}_kap{kap:.3e}_P{P}_M{M_noise}"
            _ = saem_multimode_quenched_fixedD_multi_runs_1N(
                kappa=kap, P=P, M_noise=M_noise, devices=devices,
                BASE=BASE, SAVE_DIR=SAVE_DIR, run_tag=tag
            )
            all_paths.append(tag)
    summary_path = os.path.join(SAVE_DIR, f"grid_summary_{run_tag_prefix}.json")
    with open(summary_path, "w") as f:
        json.dump({"tags": all_paths, "kappa": kappa_list, "P": P_list, "M_noise": M_noise}, f, indent=2)
    print(f"[saved grid] {summary_path}")
    return all_paths

# ----------------------------- Main -----------------------------

if __name__ == "__main__":
    set_seed(42)
    devices = check_gpu()

    BASE = dict(
        d=25, N=1024, k=4,
        σa=1.0, σw=1.0, γ=1.0,
        act="relu",
        opt_steps=4000,
        # SGLD
        chains_per_device=1024,
        mcmc_steps=300,
        mcmc_step_size=5e-3,
        mcmc_step_decay=0.999,
        langevin_sqrt2=True,
        grad_clip=1e5,
        clamp_w=10.0,
        # SAEM / quenched runs
        a0=1.0, t0=20, damping_m=3.0,
        eps_D=1e-8, print_every=20,
        ridge_lambda=1e-5,
        blocks=4,
        resample_blocks_every=0,
        crossfit_shuffle_each_iter=True,
        # χ controls
        chi_a0=0.5, chi_t0=50.0, eig_min_chi=1e-8,
        chi_guard_margin=0.02,  # keep Deff comfortably positive
        # seeds
        teacher_seed=0, noise_seed=1, data_seed=0,
        m_init=0.5,
        project_mS=True,
        m_noise_clip=3.0,
    )

    save_dir = "/home/goring/mean_field_langevin/MCMC_finiteP/results/2108_grid_infN_1Ncorr"
    os.makedirs(save_dir, exist_ok=True)
    run_tag_prefix = time.strftime("%Y%m%d_%H%M%S")

    # Example grid
    kappa_list = [1e-3]         # iterate over kappas
    P_list     = [10,100, 250, 500, 600,700,800,900,1000,1500,2000,5000]  # iterate over dataset sizes
    M_noise    = 2048

    _ = run_grid_1N(
        kappa_list=kappa_list, P_list=P_list, M_noise=M_noise,
        BASE=BASE, devices=devices,
        SAVE_DIR=save_dir, run_tag_prefix=run_tag_prefix
    )
