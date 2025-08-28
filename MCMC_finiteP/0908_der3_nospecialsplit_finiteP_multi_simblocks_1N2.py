# multimode_quenched_fixedD_multi_runs_backreacted_ramped_nobatchsolve.py
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
    eps_D: float = 1e-9
    print_every: int = 20
    project_mS: bool = True
    m_noise_clip: float = 3.0
    ridge_lambda: float = 1e-6
    blocks: int = 32
    resample_blocks_every: int = 0
    crossfit_shuffle_each_iter: bool = True
    use_backreaction: bool = True
    # ---- Back-reaction taming (A) ----
    backreact_ramp_iters: int = 300     # t_ramp
    backreact_ramp_pow: float = 1.0     # p
    invD_cap: float = 1e6               # cap on 1 / D_eff (<=0 disables)

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

# ----------------------------- Back-reaction helper -----------------------------
def compute_D_eff_from_J(
    J: torch.Tensor,          # (K,B,M)
    Sigma: torch.Tensor,      # (K,B)
    m_blocks: torch.Tensor,   # (K,M)
    kappa: float,
    mdl: ModelParams,
    eps: float,
    lambda_scale: float       # λ_t in [0,1]
) -> torch.Tensor:
    """
    D_eff = A*kappa^2 + Sigma - λ*(1/N) * (J χ J^T),
    χ_AB ≈ (N/kappa^2) E_chains[ a^2 J_A J_B ],
    E[a^2 | w] = 1/alpha + (beta/alpha)^2, alpha = A*kappa^2 + Sigma, beta = J_beta/kappa^2.
    """
    ga2 = mdl.sigma_a / (mdl.N ** mdl.gamma)
    A = 1.0 / ga2

    J_beta = J[:, :, 0] - torch.einsum('kbm,km->kb', J, m_blocks)  # (K,B)
    alpha = A * (kappa**2) + Sigma                                  # (K,B)
    beta  = J_beta / (kappa**2)                                     # (K,B)

    a2 = (1.0 / alpha) + (beta * beta) / (alpha * alpha)            # (K,B)

    chi_AB = (mdl.N / (kappa**2)) * torch.einsum('kbm,kbn,kb->kmn', J, J, a2) / float(J.shape[1])
    J_chi_J = torch.einsum('kbm,kmn,kbn->kb', J, chi_AB, J)         # (K,B)

    D_eff = alpha - lambda_scale * (1.0 / mdl.N) * J_chi_J          # (K,B)
    D_eff = torch.clamp(D_eff, min=eps)
    return D_eff

# ----------------------------- logπ(w|D_k) and grad (per block) -----------------------------

def compute_logp_and_grad_w_blocks_indep(
    w_blocks: torch.Tensor, X_blocks: torch.Tensor, chi_blocks: torch.Tensor,
    m_blocks: torch.Tensor, kappa: float, mdl: ModelParams, saem: SAEMParams,
    lambda_scale: float
):
    """
    E-step: log p(w | m, D_eff) gradient with back-reaction λ and invD cap.
    """
    gw2 = mdl.sigma_w / mdl.d
    ga2 = mdl.sigma_a / (mdl.N ** mdl.gamma)
    A   = 1.0 / ga2

    w_blocks = w_blocks.detach().requires_grad_(True)
    J, Sigma = compute_J_Sigma_blocks_indep(w_blocks, X_blocks, chi_blocks, mdl.act)  # (K,B,M),(K,B)
    J_beta = J[:, :, 0] - torch.einsum('kbm,km->kb', J, m_blocks)                     # (K,B)

    if saem.use_backreaction:
        D_eff = compute_D_eff_from_J(J, Sigma, m_blocks, kappa, mdl, saem.eps_D, lambda_scale)  # (K,B)
    else:
        D_eff = torch.clamp(A * (kappa**2) + Sigma, min=saem.eps_D)                             # (K,B)

    # invD with cap
    invD = 1.0 / D_eff
    if saem.invD_cap and saem.invD_cap > 0:
        invD = torch.clamp_max(invD, saem.invD_cap)

    # Use clamped D for log to avoid -inf
    D_for_log = torch.clamp(D_eff, min=saem.eps_D)

    prior_term = -0.5 * (w_blocks * w_blocks).sum(dim=2) / gw2
    data_term  = -0.5 * torch.log(D_for_log) + 0.5 * (J_beta * J_beta) * invD / (kappa**2)
    logp = prior_term + data_term

    grad = torch.autograd.grad(logp.sum(), w_blocks, create_graph=False, retain_graph=False)[0]
    grad = torch.where(torch.isfinite(grad), grad, torch.zeros_like(grad))
    return logp.detach(), grad.detach()

# ----------------------------- SGLD over w (independent per block) -----------------------------

def mcmc_sgld_w_blocks_indep(
    w_blocks: torch.Tensor, X_blocks: torch.Tensor, chi_blocks: torch.Tensor,
    m_blocks: torch.Tensor, kappa: float, mdl: ModelParams, saem: SAEMParams, mcmc: MCMCParams,
    lambda_scale: float
):
    step = mcmc.step_size
    for _ in range(mcmc.n_steps):
        _, grad = compute_logp_and_grad_w_blocks_indep(
            w_blocks, X_blocks, chi_blocks, m_blocks.to(w_blocks.device), kappa, mdl, saem, lambda_scale
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
        use_backreaction=_get(BASE, "use_backreaction", True),
        backreact_ramp_iters=_get(BASE, "backreact_ramp_iters", 300),
        backreact_ramp_pow=_get(BASE, "backreact_ramp_pow", 1.0),
        invD_cap=_get(BASE, "invD_cap", 1e6),
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
    traj_mS, traj_mnoise_rms, traj_time_s = [], [], []

    for it in range(1, saem.max_iters + 1):
        # λ-ramp
        if saem.use_backreaction and saem.backreact_ramp_iters > 0:
            lam = min(1.0, (it / float(saem.backreact_ramp_iters)) ** saem.backreact_ramp_pow)
        else:
            lam = 1.0 if saem.use_backreaction else 0.0

        # E-step
        for slot in per_dev:
            slot["w_blocks"] = mcmc_sgld_w_blocks_indep(
                slot["w_blocks"], slot["X_blocks"], slot["chi_blocks"],
                slot["m_blocks"], kappa, mdl, saem, mcmc, lam
            )

        # M-step
        for slot in per_dev:
            device = slot["device"]
            Xb, chib, w_blocks, m_blocks = slot["X_blocks"], slot["chi_blocks"], slot["w_blocks"], slot["m_blocks"]
            Kb, Ptot, _ = Xb.shape
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

            if saem.use_backreaction:
                D1 = compute_D_eff_from_J(J1, S1, m_blocks, kappa, mdl, saem.eps_D, lam)
                D2 = compute_D_eff_from_J(J2, S2, m_blocks, kappa, mdl, saem.eps_D, lam)
            else:
                D1 = torch.clamp(Acoef * (kappa**2) + S1, min=saem.eps_D)
                D2 = torch.clamp(Acoef * (kappa**2) + S2, min=saem.eps_D)

            # invD with cap
            invD1 = 1.0 / D1
            invD2 = 1.0 / D2
            if saem.invD_cap and saem.invD_cap > 0:
                invD1 = torch.clamp_max(invD1, saem.invD_cap)
                invD2 = torch.clamp_max(invD2, saem.invD_cap)

            # B_k: average of in-sample quadratics per half
            B1 = torch.einsum('kbm,kbn,kb->kmn', J1, J1, invD1) / float(J1.shape[1])  # (K,M,M)
            B2 = torch.einsum('kbm,kbn,kb->kmn', J2, J2, invD2) / float(J2.shape[1])  # (K,M,M)
            Bk = 0.5 * (B1 + B2)                                                     # (K,M,M)
            Bk = 0.5 * (Bk + Bk.transpose(-1, -2))                                   # symmetrize

            # b_k: cross-fitted (disjoint halves)
            w12 = (J2[:, :, 0] * invD1)                                            # (K,B)
            w21 = (J1[:, :, 0] * invD2)                                            # (K,B)
            b12 = torch.einsum('kbm,kb->km', J1, w12) / float(J1.shape[1])         # (K,M)
            b21 = torch.einsum('kbm,kb->km', J2, w21) / float(J2.shape[1])         # (K,M)
            bk = 0.5 * (b12 + b21)                                                 # (K,M)

            # Solve per block (NO BATCH) to avoid batched-kernel warning
            Kb_, Mtot_local = bk.shape
            m_star = torch.empty_like(bk)

            for k_idx in range(Kb_):
                I = torch.eye(Mtot_local, device=device, dtype=torch.float64)
                lhs64 = (torch.eye(Mtot_local, device=device, dtype=torch.float32)
                         + mdl.N * Bk[k_idx]
                         + saem.ridge_lambda * torch.eye(Mtot_local, device=device, dtype=torch.float32)).to(torch.float64)
                rhs64 = (mdl.N * bk[k_idx]).to(torch.float64).unsqueeze(1)

                # Symmetrize lhs64 numerically
                lhs64 = 0.5 * (lhs64 + lhs64.T)

                jitter_schedule = [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
                solved = False
                for jit in jitter_schedule:
                    try:
                        L = torch.linalg.cholesky(lhs64 + jit * I)
                        x64 = torch.cholesky_solve(rhs64, L)
                        m_star[k_idx] = x64.squeeze(1).to(torch.float32)
                        solved = True
                        break
                    except Exception:
                        try:
                            x64 = torch.linalg.solve(lhs64 + jit * I, rhs64)
                            m_star[k_idx] = x64.squeeze(1).to(torch.float32)
                            solved = True
                            break
                        except Exception:
                            continue
                if not solved:
                    # Least-squares fallback
                    x64, *_ = torch.linalg.lstsq(lhs64, rhs64)
                    m_star[k_idx] = x64.squeeze(1).to(torch.float32)

            # SAEM averaging per block
            a_t = saem.a0 / (it + saem.t0)
            m_new = (1 - saem.damping_m * a_t) * m_blocks + saem.damping_m * a_t * m_star

            # constraints
            if saem.project_mS:
                m_new[:, 0] = m_new[:, 0].clamp(0.0, 1.0)
            if saem.m_noise_clip and saem.m_noise_clip > 0:
                m_new[:, 1:] = m_new[:, 1:].clamp(-saem.m_noise_clip, saem.m_noise_clip)

            slot["m_blocks"] = m_new

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

        # Logging
        if it % saem.print_every == 0 or it == 1:
            ge_all = 0.5 * (1.0 - mS_all.numpy())**2
            msg = {
                "iter": it, "kappa": kappa, "P": P,
                "blocks": sum(_get(BASE, "blocks", 32) for _ in per_dev),
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
        "kappa": kappa, "P": P, "mode": "multi-quenched-fixedD-parallel-xfit",
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
        opt_steps=4000,
        # SGLD
        chains_per_device=1024,
        mcmc_steps=400,
        mcmc_step_size=5e-3,
        mcmc_step_decay=0.999,
        langevin_sqrt2=True,
        grad_clip=1e5,
        clamp_w=10.0,
        # SAEM / quenched runs
        a0=1.0, t0=200.0, damping_m=1.5,
        eps_D=1e-6,                 # you can raise/lower this floor
        print_every=5,
        ridge_lambda=1e-6,
        blocks=4,
        resample_blocks_every=0,
        crossfit_shuffle_each_iter=True,
        use_backreaction=True,
        # ---- A) Back-reaction taming ----
        backreact_ramp_iters=800,   # ramp λ to 1 over 300 SAEM iterations
        backreact_ramp_pow=2.0,     # linear ramp; try 2.0 for slower start
        invD_cap=1e8,               # cap on 1/D_eff; adjust if still spiky
        # seeds
        teacher_seed=0, noise_seed=1, data_seed=0,
        m_init=0.5,
        project_mS=True,
        m_noise_clip=3.0,
    )

    save_dir = "/home/goring/mean_field_langevin/MCMC_finiteP/results/2208_grid_infN_1Ncorr"
    os.makedirs(save_dir, exist_ok=True)
    run_tag_prefix = time.strftime("%Y%m%d_%H%M%S")

    # >>> Set your sweeps here <<<
    kappa_list = [1e-3,1e-2,1e-1]
    P_list     = [10, 100, 250, 500,600,700,800,900, 1000, 2000]
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
