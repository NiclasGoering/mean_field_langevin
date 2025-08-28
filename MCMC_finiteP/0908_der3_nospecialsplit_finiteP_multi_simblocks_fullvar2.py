# multimode_quenched_fixedD_multi_runs_diag_inflated.py
import os, json, time, math, random
from typing import Dict, List
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
        if key in seen:
            continue
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
    """Compute chi_A for all modes. Returns (K,P,M)."""
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
    eps_D: float = 1e-9
    print_every: int = 20
    project_mS: bool = True
    m_noise_clip: float = 3.0
    ridge_lambda: float = 1e-6
    blocks: int = 32                 # K parallel quenched runs
    resample_blocks_every: int = 0   # 0 => FIXED datasets (true quenched)
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

# ----------------------------- logπ(w|D_k) and grad (per block) -----------------------------

def compute_logp_and_grad_w_blocks_indep(
    w_blocks: torch.Tensor, X_blocks: torch.Tensor, chi_blocks: torch.Tensor,
    m_blocks: torch.Tensor, kappa: float, mdl: ModelParams, saem: SAEMParams,
    backreact_lambda: float = 1.0
):
    """
    w_blocks: (K,B,d)
    m_blocks: (K,M)
    backreact_lambda in [0,1]: tamed back-reaction multiplier.
    """
    gw2 = mdl.sigma_w / mdl.d
    ga2 = mdl.sigma_a / (mdl.N ** mdl.gamma)
    A   = 1.0 / ga2

    w_blocks = w_blocks.detach().requires_grad_(True)
    J, Sigma = compute_J_Sigma_blocks_indep(w_blocks, X_blocks, chi_blocks, mdl.act)  # (K,B,M),(K,B)

    # J_beta[k,b] = J_S - λ_br * sum_A m_k[A] * J_A
    J_beta = J[:, :, 0] - backreact_lambda * torch.einsum('kbm,km->kb', J, m_blocks)  # (K,B)
    D = A * (kappa**2) + Sigma
    D = torch.clamp(D, min=saem.eps_D)

    prior_term = -0.5 * (w_blocks * w_blocks).sum(dim=2) / gw2                        # (K,B)
    data_term  = -0.5 * torch.log(D) + 0.5 * (J_beta * J_beta) / ((kappa**2) * D)     # (K,B)
    logp = prior_term + data_term

    grad = torch.autograd.grad(logp.sum(), w_blocks, create_graph=False, retain_graph=False)[0]
    grad = torch.where(torch.isfinite(grad), grad, torch.zeros_like(grad))
    return logp.detach(), grad.detach()

# ----------------------------- SGLD over w (independent per block) -----------------------------

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
            w_blocks = w_blocks + step * grad + torch.randn_like(w_blocks) * math.sqrt(2.0 * step)
        else:
            w_blocks = w_blocks + 0.5 * step * grad + torch.randn_like(w_blocks) * math.sqrt(step)

        if mcmc.clamp_w:
            w_blocks = torch.clamp(w_blocks, -mcmc.clamp_w, mcmc.clamp_w)
        step *= mcmc.step_decay
    return w_blocks.detach()

# ----------------------------- SAEM (K independent quenched runs) -----------------------------

def saem_multimode_quenched_fixedD_multi_runs(
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
        eps_D=_get(BASE, "eps_D", 1e-9),
        print_every=_get(BASE, "print_every", 20),
        project_mS=_get(BASE, "project_mS", True),
        m_noise_clip=_get(BASE, "m_noise_clip", 3.0),
        ridge_lambda=_get(BASE, "ridge_lambda", 1e-6),
        blocks=_get(BASE, "blocks", 32),
        resample_blocks_every=_get(BASE, "resample_blocks_every", 0),
        crossfit_shuffle_each_iter=_get(BASE, "crossfit_shuffle_each_iter", True),
    )

    # Extra knobs (with reasonable defaults)
    center_noise_J      = _get(BASE, "center_noise_J", True)
    alpha_offdiag_noise = _get(BASE, "alpha_offdiag_noise", 1.0)  # fraction of estimated noise off-mass to add
    alpha_offdiag_sig   = _get(BASE, "alpha_offdiag_signal", 1.0) # fraction for the signal row
    # back-reaction ramp:
    br_ramp_iters       = _get(BASE, "backreact_ramp_iters", 0)   # 0 => no ramp (λ=1)
    br_ramp_pow         = _get(BASE, "backreact_ramp_pow", 1.0)

    # teacher + noise modes
    S = make_parity_indices(mdl.d, mdl.k, seed=_get(BASE, "teacher_seed", 0))
    A_noise = make_k_subsets(mdl.d, mdl.k, M_noise, S, seed=_get(BASE, "noise_seed", 1))
    modes = [S] + A_noise
    modes_idx = torch.stack(modes).long()   # (Mtot,k)
    Mtot = modes_idx.shape[0]
    Mn = Mtot - 1  # number of noise modes

    # per-device state
    per_dev = []
    for di in devices if len(devices) > 0 else [-1]:
        device = torch.device(f"cuda:{di}") if di >= 0 and torch.cuda.is_available() else torch.device("cpu")
        K, B = saem.blocks, mcmc.n_chains_per_device
        gw2 = mdl.sigma_w / mdl.d
        w_blocks = torch.randn(K, B, mdl.d, device=device) * math.sqrt(gw2)

        data_seed = _get(BASE, "data_seed", 0) + 10000 * (di + 1)
        X_blocks = make_boolean_blocks(K, P, mdl.d, device, seed=data_seed)          # FIXED datasets
        chi_blocks = chi_of_modes_blocks(X_blocks, modes_idx, chunk=min(512, Mtot))

        # one m-vector PER BLOCK (quenched): (K,M)
        m_blocks = torch.zeros(K, Mtot, device=device, dtype=torch.float32)
        m_init = _get(BASE, "m_init", 0.5)
        m_blocks[:, 0] = m_init

        per_dev.append({
            "device": device,
            "w_blocks": w_blocks,
            "X_blocks": X_blocks,
            "chi_blocks": chi_blocks,
            "m_blocks": m_blocks,
        })

    Acoef = (mdl.N**mdl.gamma) / mdl.sigma_a
    t_start = time.time()

    # --- Trajectory collectors ---
    traj_mS = []
    traj_mnoise_rms = []
    traj_time_s = []

    for it in range(1, saem.max_iters + 1):
        # back-reaction ramp λ_br(it)
        if br_ramp_iters and br_ramp_iters > 0:
            lam_br = min(1.0, (it / float(br_ramp_iters)) ** br_ramp_pow)
        else:
            lam_br = 1.0

        # E-step: SGLD with tamed back-reaction
        for slot in per_dev:
            slot["w_blocks"] = mcmc_sgld_w_blocks_indep(
                slot["w_blocks"], slot["X_blocks"], slot["chi_blocks"],
                slot["m_blocks"], kappa, mdl, saem, mcmc,
                backreact_lambda=lam_br
            )

        # M-step (diagonal + centering + offdiag inflation)
        for slot in per_dev:
            device = slot["device"]
            Xb, chib, w_blocks, m_blocks = slot["X_blocks"], slot["chi_blocks"], slot["w_blocks"], slot["m_blocks"]
            Kb, Ptot, _ = Xb.shape

            # Optional shuffle along P for crossfit (still within each fixed dataset)
            if saem.crossfit_shuffle_each_iter:
                perm = torch.randperm(Ptot, device=device)
                Xb = Xb[:, perm, :]
                chib = chib[:, perm, :]

            s = Ptot // 2
            X1, X2 = Xb[:, :s, :], Xb[:, s:, :]
            C1, C2 = chib[:, :s, :], chib[:, s:, :]

            # Moments on halves
            J1, S1 = compute_J_Sigma_blocks_indep(w_blocks, X1, C1, mdl.act)  # (K,B,M),(K,B)
            J2, S2 = compute_J_Sigma_blocks_indep(w_blocks, X2, C2, mdl.act)
            D1 = torch.clamp(Acoef * (kappa**2) + S1, min=saem.eps_D)        # (K,B)
            D2 = torch.clamp(Acoef * (kappa**2) + S2, min=saem.eps_D)
            invD1, invD2 = 1.0 / D1, 1.0 / D2

            # --- Center noise J's to remove common "all-ones" leakage ---
            if center_noise_J and Mtot > 1:
                mu1 = J1[:, :, 1:].mean(dim=2, keepdim=True)     # (K,B,1)
                mu2 = J2[:, :, 1:].mean(dim=2, keepdim=True)
                J1c_noise = J1[:, :, 1:] - mu1
                J2c_noise = J2[:, :, 1:] - mu2
                # keep signal unchanged, concat back
                J1c = torch.cat([J1[:, :, :1], J1c_noise], dim=2)
                J2c = torch.cat([J2[:, :, :1], J2c_noise], dim=2)
            else:
                J1c, J2c = J1, J2

            # Diagonal B: average per mode of J^2 / D over chains (cross-fit halves averaged)
            B1_diag = torch.einsum('kbm,kb->km', J1c * J1c, invD1) / float(J1c.shape[1])  # (K,M)
            B2_diag = torch.einsum('kbm,kb->km', J2c * J2c, invD2) / float(J2c.shape[1])  # (K,M)
            B_diag  = 0.5 * (B1_diag + B2_diag)                                           # (K,M)

            # Cross-fitted b as before
            J1_S = J1c[:, :, 0]    # (K,B)
            J2_S = J2c[:, :, 0]    # (K,B)
            w12  = J2_S * invD1    # (K,B)
            w21  = J1_S * invD2    # (K,B)
            b12  = torch.einsum('kbm,kb->km', J1c, w12) / float(J1c.shape[1])  # (K,M)
            b21  = torch.einsum('kbm,kb->km', J2c, w21) / float(J2c.shape[1])  # (K,M)
            b_vec = 0.5 * (b12 + b21)                                           # (K,M)

            # ----- Data-driven inflation for missing off-diagonals -----
            # noise block estimate
            if Mtot > 1:
                # half-1
                J1n = J1c[:, :, 1:]                               # (K,B,Mn)
                sumJ1n = J1n.sum(dim=2)                           # (K,B)
                off_mass1 = torch.einsum('kb,kb->k', sumJ1n*sumJ1n, invD1) - \
                            torch.einsum('kbm,kb->k', J1n*J1n, invD1)       # (K,)
                diag_mass1 = torch.einsum('kbm,kb->k', J1n*J1n, invD1)      # (K,)

                # half-2
                J2n = J2c[:, :, 1:]
                sumJ2n = J2n.sum(dim=2)
                off_mass2 = torch.einsum('kb,kb->k', sumJ2n*sumJ2n, invD2) - \
                            torch.einsum('kbm,kb->k', J2n*J2n, invD2)
                diag_mass2 = torch.einsum('kbm,kb->k', J2n*J2n, invD2)

                # average and normalize by B
                Bsize = float(J1c.shape[1])
                rho_noise = 0.5 * ((off_mass1 + off_mass2) / Bsize) / \
                            (0.5 * ((diag_mass1 + diag_mass2) / Bsize) + 1e-12)  # (K,)

                # signal row off mass vs diag (cross with noise)
                cross1 = torch.einsum('kb,kb->k', J1_S, sumJ1n) * (1.0 / Bsize)
                cross2 = torch.einsum('kb,kb->k', J2_S, sumJ2n) * (1.0 / Bsize)
                diagS1 = torch.einsum('kb,kb->k', J1_S, J1_S) * (1.0 / Bsize)
                diagS2 = torch.einsum('kb,kb->k', J2_S, J2_S) * (1.0 / Bsize)
                rho_sig = 0.5 * ((cross1 + cross2) / ( (diagS1 + diagS2) * 0.5 + 1e-12))  # (K,)
            else:
                rho_noise = torch.zeros(Kb, device=device)
                rho_sig   = torch.zeros(Kb, device=device)

            # Build per-mode inflation multipliers τ
            tau = torch.ones_like(B_diag)                               # (K,M)
            if Mtot > 1 and alpha_offdiag_noise != 0.0:
                tau[:, 1:] = 1.0 + alpha_offdiag_noise * rho_noise.unsqueeze(1)
            if alpha_offdiag_sig != 0.0:
                tau[:, 0]  = 1.0 + alpha_offdiag_sig * rho_sig

            B_diag_infl = tau * B_diag

            # Scalar (per-mode) update: m* = N b / (1 + N B_infl + ridge)
            denom = 1.0 + mdl.N * B_diag_infl + saem.ridge_lambda
            m_star = (mdl.N * b_vec) / denom                                   # (K,M)

            # SAEM averaging per block
            a_t = saem.a0 / (it + saem.t0)
            m_new = (1 - saem.damping_m * a_t) * m_blocks + saem.damping_m * a_t * m_star

            # constraints
            if saem.project_mS:
                m_new[:, 0] = m_new[:, 0].clamp(0.0, 1.0)
            if saem.m_noise_clip and saem.m_noise_clip > 0:
                m_new[:, 1:] = m_new[:, 1:].clamp(-saem.m_noise_clip, saem.m_noise_clip)

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

        # Logging
        if it % saem.print_every == 0 or it == 1:
            ge_all = 0.5 * (1.0 - mS_all.numpy())**2
            msg = {
                "iter": it, "kappa": kappa, "P": P, "blocks": sum(_get(BASE, "blocks", 32) for _ in per_dev),
                "m_S_mean": float(mS_all.mean().item()),
                "m_S_std": float(mS_all.std(unbiased=False).item()),
                "gen_err_mean": float(ge_all.mean()),
                "gen_err_std": float(ge_all.std()),
                "m_noise_rms_mean": float(noise_rms.mean().item()) if noise_rms.numel() > 0 else 0.0,
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
        "kappa": kappa, "P": P, "mode": "multi-quenched-fixedD-parallel-xfit-diag-centered-inflated",
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
    tag = run_tag if run_tag else f"kap{kappa:.3e}_P{P}_K{_get(BASE,'blocks',32)}_M{M_noise}_it{_get(BASE,'opt_steps',800)}_B{_get(BASE,'chains_per_device',1024)}"
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
            tag = f"{run_tag_prefix}_kap{kappa:.3e}_P{P}_K{_get(BASE,'blocks',32)}_M{M_noise}"
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
        eps_D=1e-9, print_every=20,
        ridge_lambda=1e-4,
        blocks=2,                       # try 2–8 for quick tests
        resample_blocks_every=0,
        crossfit_shuffle_each_iter=True,
        # seeds
        teacher_seed=0, noise_seed=1, data_seed=0,
        m_init=0.25,
        project_mS=True,
        m_noise_clip=3.0,
        # ---- diagonal M-step stabilizers ----
        center_noise_J=True,
        alpha_offdiag_noise=1.0,        # 0.5–1.5 are reasonable
        alpha_offdiag_signal=2.2,       # tune up/down if S still pops early
        # ---- back-reaction ramp ----
        backreact_ramp_iters=300,
        backreact_ramp_pow=1.0,
    )

    save_dir = "/home/goring/mean_field_langevin/MCMC_finiteP/results/diag_center_inflate"
    os.makedirs(save_dir, exist_ok=True)
    run_tag_prefix = time.strftime("%Y%m%d_%H%M%S")

    kappa_list = [1e-3]         # start with one κ to validate the transition
    P_list     = [100, 250, 500, 600, 700, 800, 900, 1000]
    M_noise    = 2048

    run_grid(
        kappa_list=kappa_list,
        P_list=P_list,
        M_noise=M_noise,
        devices=devices,
        BASE=BASE,
        SAVE_DIR=save_dir,
        run_tag_prefix=run_tag_prefix
    )
