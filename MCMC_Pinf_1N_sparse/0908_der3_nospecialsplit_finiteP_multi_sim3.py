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
    return X.to(torch.float32) * 2.0 - 1.0  # {-1,+1}

def chi_of_modes_blocks(X_blocks: torch.Tensor, modes_idx: torch.Tensor, chunk: int = 512) -> torch.Tensor:
    K, P, d = X_blocks.shape
    M, k = modes_idx.shape
    device = X_blocks.device
    chi = torch.empty((K, P, M), device=device, dtype=X_blocks.dtype)
    if chunk <= 0: chunk = M
    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        idx = modes_idx[s:e].to(device)                 # (mchunk,k)
        idx_exp = idx.view(1,1,e-s,k).expand(K,P,e-s,k) # (K,P,mchunk,k)
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
    blocks: int = 16                    # K datasets per device
    resample_blocks_every: int = 1
    crossfit_shuffle_each_iter: bool = True  # shuffle samples before splitting

# ----------------------------- Moments per block -----------------------------

def compute_J_Sigma_blocks_indep(
    w_blocks: torch.Tensor,   # (K,B,d)
    X_blocks: torch.Tensor,   # (K,P,d)
    chi_blocks: torch.Tensor, # (K,P,M)
    act: str
):
    K, P, d = X_blocks.shape
    _, B, d2 = w_blocks.shape
    assert d == d2
    z = torch.einsum('kpd,kbd->kpb', X_blocks, w_blocks)  # (K,P,B)
    phi = activation(z, act)                              # (K,P,B)
    J = torch.einsum('kpb,kpm->kbm', phi, chi_blocks) / float(P)  # (K,B,M)
    Sigma = (phi * phi).mean(dim=1)                       # (K,B)
    return J, Sigma, phi

# ----------------------------- logπ(w|D_k) and grad (per block) -----------------------------

def compute_logp_and_grad_w_blocks_indep(
    w_blocks: torch.Tensor, X_blocks: torch.Tensor, chi_blocks: torch.Tensor,
    m_vec: torch.Tensor, kappa: float, mdl: ModelParams, saem: SAEMParams
):
    gw2 = mdl.sigma_w / mdl.d
    ga2 = mdl.sigma_a / (mdl.N ** mdl.gamma)
    A   = 1.0 / ga2

    w_blocks = w_blocks.detach().requires_grad_(True)
    J_blocks, Sigma_blocks, _ = compute_J_Sigma_blocks_indep(w_blocks, X_blocks, chi_blocks, mdl.act)

    J_beta = J_blocks[:, :, 0] - (J_blocks * m_vec[None, None, :]).sum(dim=2)  # (K,B)
    D = A * (kappa**2) + Sigma_blocks
    D = torch.clamp(D, min=saem.eps_D)

    prior_term = -0.5 * (w_blocks * w_blocks).sum(dim=2) / gw2                # (K,B)
    data_term  = -0.5 * torch.log(D) + 0.5 * (J_beta * J_beta) / ((kappa**2) * D)
    logp = prior_term + data_term

    grad = torch.autograd.grad(logp.sum(), w_blocks, create_graph=False, retain_graph=False)[0]
    grad = torch.where(torch.isfinite(grad), grad, torch.zeros_like(grad))
    return logp.detach(), grad.detach()

# ----------------------------- SGLD over w (independent per block) -----------------------------

def mcmc_sgld_w_blocks_indep(
    w_blocks: torch.Tensor, X_blocks: torch.Tensor, chi_blocks: torch.Tensor,
    m_vec: torch.Tensor, kappa: float, mdl: ModelParams, saem: SAEMParams, mcmc: MCMCParams
):
    step = mcmc.step_size
    for _ in range(mcmc.n_steps):
        _, grad = compute_logp_and_grad_w_blocks_indep(
            w_blocks, X_blocks, chi_blocks, m_vec.to(w_blocks.device), kappa, mdl, saem
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

# ----------------------------- SAEM with cross-fitted M-step -----------------------------

def saem_multimode_quenched_exact_blocks_faithful_xfit(
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
        a0=_get(BASE, "a0", 0.5),
        t0=_get(BASE, "t0", 100.0),
        damping_m=_get(BASE, "damping_m", 1.0),
        eps_D=_get(BASE, "eps_D", 1e-9),
        print_every=_get(BASE, "print_every", 20),
        project_mS=_get(BASE, "project_mS", True),
        m_noise_clip=_get(BASE, "m_noise_clip", 3.0),
        ridge_lambda=_get(BASE, "ridge_lambda", 1e-6),
        blocks=_get(BASE, "blocks", 16),
        resample_blocks_every=_get(BASE, "resample_blocks_every", 1),
        crossfit_shuffle_each_iter=_get(BASE, "crossfit_shuffle_each_iter", True),
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
        K, B = _get(BASE, "blocks", 16), mcmc.n_chains_per_device
        gw2 = mdl.sigma_w / mdl.d
        w_blocks = torch.randn(K, B, mdl.d, device=device) * math.sqrt(gw2)

        data_seed = _get(BASE, "data_seed", 0) + 10000 * (di + 1)
        X_blocks = make_boolean_blocks(K, P, mdl.d, device, seed=data_seed)
        chi_blocks = chi_of_modes_blocks(X_blocks, modes_idx, chunk=min(512, Mtot))

        per_dev.append({"device": device, "w_blocks": w_blocks,
                        "X_blocks": X_blocks, "chi_blocks": chi_blocks,
                        "data_seed": data_seed})

    # init m
    m_vec = torch.zeros(Mtot, dtype=torch.float32); m_vec[0] = _get(BASE, "m_init", 0.5)
    m_bar = m_vec.clone()

    Acoef = (mdl.N**mdl.gamma) / mdl.sigma_a
    t_start, history = time.time(), []

    for it in range(1, saem.max_iters + 1):
        # (re)sample fresh datasets if requested
        if saem.resample_blocks_every and (it % saem.resample_blocks_every == 0):
            for slot in per_dev:
                device = slot["device"]; K = saem.blocks
                slot["data_seed"] += 1
                Xb = make_boolean_blocks(K, P, mdl.d, device, seed=slot["data_seed"])
                chib = chi_of_modes_blocks(Xb, modes_idx, chunk=min(512, Mtot))
                slot["X_blocks"], slot["chi_blocks"] = Xb, chib

        # SGLD: independent posterior per block (w.r.t. the full P samples)
        for slot in per_dev:
            slot["w_blocks"] = mcmc_sgld_w_blocks_indep(
                slot["w_blocks"], slot["X_blocks"], slot["chi_blocks"],
                m_vec.to(slot["device"]), kappa, mdl, saem, mcmc
            )

        # CPU master buffers for cross-fitted M-step
        B_master = torch.zeros((Mtot, Mtot), dtype=torch.float64, device="cpu")
        b_master = torch.zeros(Mtot, dtype=torch.float64, device="cpu")
        n_slots = 0

        for slot in per_dev:
            device = slot["device"]
            Xb, chib, w_blocks = slot["X_blocks"], slot["chi_blocks"], slot["w_blocks"]  # (K,P,d),(K,P,M),(K,B,d)
            K, Ptot, _ = Xb.shape
            # optional shuffle along P for crossfit
            if saem.crossfit_shuffle_each_iter:
                # one common permutation per device is enough (iid)
                perm = torch.randperm(Ptot, device=device)
                Xb = Xb[:, perm, :]
                chib = chib[:, perm, :]

            # split into two halves
            s = Ptot // 2
            X1, X2 = Xb[:, :s, :], Xb[:, s:, :]
            C1, C2 = chib[:, :s, :], chib[:, s:, :]

            # moments on halves (per block, per chain)
            J1, S1, _ = compute_J_Sigma_blocks_indep(w_blocks, X1, C1, mdl.act)   # (K,B,M),(K,B)
            J2, S2, _ = compute_J_Sigma_blocks_indep(w_blocks, X2, C2, mdl.act)
            D1 = torch.clamp(Acoef * (kappa**2) + S1, min=saem.eps_D)
            D2 = torch.clamp(Acoef * (kappa**2) + S2, min=saem.eps_D)
            invD1, invD2 = 1.0 / D1, 1.0 / D2     # (K,B)

            # B_k: average of in-sample quadratic terms
            B1 = torch.einsum('kbm,kbn,kb->kmn', J1, J1, invD1) / float(J1.shape[1])
            B2 = torch.einsum('kbm,kbn,kb->kmn', J2, J2, invD2) / float(J2.shape[1])
            Bk = 0.5 * (B1 + B2)                 # (K,M,M)

            # b_k: CROSS-FIT (no same-sample product)
            # weights12 = (J_S^{(2)} / D^{(1)}), weights21 = (J_S^{(1)} / D^{(2)})
            w12 = (J2[:, :, 0] * invD1)          # (K,B)
            w21 = (J1[:, :, 0] * invD2)          # (K,B)
            b12 = torch.einsum('kbm,kb->km', J1, w12) / float(J1.shape[1])  # (K,M)
            b21 = torch.einsum('kbm,kb->km', J2, w21) / float(J2.shape[1])  # (K,M)
            bk = 0.5 * (b12 + b21)               # (K,M)

            # average across blocks (K), move to CPU/double
            B_master += Bk.mean(dim=0).double().cpu()
            b_master += bk.mean(dim=0).double().cpu()
            n_slots += 1

        if n_slots > 1:
            B_master /= n_slots
            b_master /= n_slots

        # Linear solve: (I + N * E[B]) m* = N * E[b]
        I = torch.eye(Mtot, dtype=torch.float64, device="cpu")
        lhs = I + (mdl.N * B_master) + saem.ridge_lambda * I
        rhs = (mdl.N * b_master)

        try:
            L = torch.linalg.cholesky(lhs)
            m_star = torch.cholesky_solve(rhs.unsqueeze(1), L).squeeze(1)
        except Exception:
            m_star = torch.linalg.solve(lhs, rhs)

        m_star = m_star.to(m_vec.dtype)

        # SAEM averaging
        a_t = saem.a0 / (it + saem.t0)
        m_vec = (1 - saem.damping_m * a_t) * m_vec + saem.damping_m * a_t * m_star

        # constraints
        if saem.project_mS:
            m_vec[0] = float(min(max(m_vec[0].item(), 0.0), 1.0))
        if saem.m_noise_clip and saem.m_noise_clip > 0:
            m_vec[1:] = m_vec[1:].clamp(-saem.m_noise_clip, saem.m_noise_clip)

        m_bar = (m_bar * (it - 1) + m_vec) / it

        if it % saem.print_every == 0 or it == 1:
            ge = 0.5 * (1.0 - float(m_vec[0].item()))**2
            dt = time.time() - t_start
            print(json.dumps({
                "iter": it, "kappa": kappa, "P": P, "blocks": saem.blocks,
                "m_S": round(float(m_vec[0].item()), 6),
                "gen_err": round(ge, 6),
                "m_noise_rms": round(float(m_vec[1:].pow(2).mean().sqrt().item()), 6) if Mtot > 1 else 0.0,
                "time_s": round(dt, 2)
            }))

    result = {
        "kappa": kappa, "P": P, "blocks": saem.blocks, "mode": "multi-quenched-exact-blocks-faithful-xfit",
        "m_final": float(m_vec[0].item()),
        "gen_err": 0.5 * (1.0 - float(m_vec[0].item()))**2,
        "m_noise_rms": float(m_vec[1:].pow(2).mean().sqrt().item()) if Mtot > 1 else 0.0,
        "BASE": BASE
    }
    os.makedirs(SAVE_DIR, exist_ok=True)
    tag = run_tag if run_tag else f"kappa_{kappa:.6f}_P_{P}_K{saem.blocks}"
    out_path = os.path.join(SAVE_DIR, f"saem_multimode_quenched_exact_blocks_faithful_xfit_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[saved] {out_path}")
    return result

# ----------------------------- Main -----------------------------

if __name__ == "__main__":
    set_seed(42)
    devices = check_gpu()

    BASE = dict(
        d=25, N=1024, k=4,
        σa=1.0, σw=1.0, γ=1.0,
        act="relu",
        opt_steps=400,
        # SGLD
        chains_per_device=1024,
        mcmc_steps=600,
        mcmc_step_size=5e-3,
        mcmc_step_decay=0.999,
        langevin_sqrt2=True,
        grad_clip=1e5,
        clamp_w=10.0,
        # SAEM / blocks
        a0=0.5, t0=100.0, damping_m=1.0,
        eps_D=1e-9, print_every=20,
        ridge_lambda=1e-6,
        blocks=32,
        resample_blocks_every=1,
        crossfit_shuffle_each_iter=True,
        # seeds
        teacher_seed=0, noise_seed=1, data_seed=0,
        m_init=0.5,
        project_mS=True,
        m_noise_clip=3.0,
    )

    save_dir = "./results_multimode_quenched_exact_blocks_faithful_xfit"
    os.makedirs(save_dir, exist_ok=True)
    run_tag_prefix = time.strftime("%Y%m%d_%H%M%S")

    # Example single run
    kappa = 1e-3
    P = 50
    M_noise = 0

    _ = saem_multimode_quenched_exact_blocks_faithful_xfit(
        kappa=kappa, P=P, M_noise=M_noise, devices=devices,
        BASE=BASE, SAVE_DIR=save_dir, run_tag=run_tag_prefix + "_single"
    )
