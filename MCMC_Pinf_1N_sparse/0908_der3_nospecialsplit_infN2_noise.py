import os, json, time, math, random
from typing import Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
import torch
# torch.set_default_dtype(torch.float64)

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

def _get(BASE: Dict, *keys, default=None):
    for k in keys:
        if k in BASE:
            return BASE[k]
    return default

# ----------------------------- Core math -----------------------------

def make_parity_indices(d: int, k: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    S = torch.randperm(d, generator=g)[:k]
    return S.sort().values

def make_k_subsets(d: int, k: int, M: int, S: torch.Tensor, seed: int = 1) -> List[torch.Tensor]:
    """Sample M random k-subsets (skip exact equality with S)."""
    g = torch.Generator().manual_seed(seed)
    A_list, tried = [], set()
    S_set = set(S.tolist())
    while len(A_list) < M:
        A = torch.randperm(d, generator=g)[:k].sort().values
        key = tuple(A.tolist())
        if key in tried:
            continue
        tried.add(key)
        if set(A.tolist()) != S_set:
            A_list.append(A)
    return A_list

def make_boolean_batch(R: int, d: int, device: torch.device, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.randint(low=0, high=2, size=(R, d), generator=g, device=device, dtype=torch.int8)
    X = X.to(torch.float32) * 2.0 - 1.0
    return X

def chi_of_modes(X: torch.Tensor, modes: List[torch.Tensor]) -> torch.Tensor:
    """Return (R, M) matrix with columns chi_A(x) for each A in modes; modes[0] should be S."""
    cols = [X[:, A.long()].prod(dim=1) for A in modes]
    return torch.stack(cols, dim=1)  # (R, M)

def activation(z: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "relu":
        return torch.relu(z)
    elif kind == "tanh":
        return torch.tanh(z)
    else:
        raise ValueError(f"Unknown activation: {kind}")

# ----------------------------- Params -----------------------------

@dataclass
class ModelParams:
    d: int = 30
    N: int = 1024
    k: int = 4
    sigma_a: float = 1.0     # σ_a in the paper
    sigma_w: float = 1.0     # σ_w in the paper
    gamma: float = 1.0
    act: str = "relu"

@dataclass
class MCMCParams:
    R_inputs: int = 8192
    n_chains_per_device: int = 8192
    n_steps: int = 30
    step_size: float = 5e-3
    step_decay: float = 0.999
    langevin_sqrt2: bool = True
    grad_clip: float = 10.0
    clamp_w: float = 10.0

@dataclass
class SAEMParams:
    max_iters: int = 4000
    a0: float = 0.5
    t0: float = 100.0
    damping_m: float = 1.0
    eps_D: float = 1e-6
    print_every: int = 50

# ----------------------------- J, Σ (multi-mode) -----------------------------

def compute_J_Sigma_multimode(
    w: torch.Tensor, X: torch.Tensor, chi_mat: torch.Tensor, act: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    J_mat: (B, M) with J_A = E_x[phi * chi_A], Sigma: (B,)
    """
    z = X @ w.T                # (R,B)
    phi = activation(z, act)   # (R,B)
    R = X.shape[0]
    J_mat = (phi.T @ chi_mat) / float(R)  # (B,M)
    Sigma = (phi * phi).mean(dim=0)       # (B,)
    return J_mat, Sigma

# ----------------------------- logπ and grad (multi-mode; no 1/P) -----------------------------

def compute_logp_and_grad_multimode(
    w: torch.Tensor, X: torch.Tensor, chi_mat: torch.Tensor,
    m_vec: torch.Tensor, kappa: float,
    mdl: ModelParams, saem: SAEMParams, mcmc: MCMCParams
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Integrate out 'a' in the infinite-data limit with multiple modes:
      A = N^γ / σ_a
      D = A κ^2 + Σ
      B = J_S - sum_A m_A J_A
      logp(w) = -||w||^2/(2 g_w^2) - 0.5*log D + (1/(2 κ^2)) * (B^2 / D)
    """
    gw2 = mdl.sigma_w / mdl.d
    ga2 = mdl.sigma_a / (mdl.N ** mdl.gamma)  # g_a^2
    Acoef = 1.0 / ga2                         # = N^γ / σ_a

    w = w.detach().requires_grad_(True)

    J_mat, Sigma = compute_J_Sigma_multimode(w, X, chi_mat, mdl.act)  # (B,M), (B,)
    J_S = J_mat[:, 0]                                                 # (B,)
    B = J_S - (J_mat @ m_vec)                                         # (B,)
    D = Acoef * (kappa**2) + Sigma                                    # (B,)
    D_safe = torch.clamp(D, min=saem.eps_D)

    prior_term   = -0.5 * (w * w).sum(dim=1) / gw2
    log_det_term = -0.5 * torch.log(D_safe)
    data_term    = 0.5 * (B * B) / ((kappa**2) * D_safe)

    logp = prior_term + log_det_term + data_term

    grad = torch.autograd.grad(logp.sum(), w, create_graph=False, retain_graph=False)[0]
    if mcmc.grad_clip and mcmc.grad_clip > 0:
        gn = grad.norm(dim=1, keepdim=True).clamp_min(1e-12)
        grad = grad * (mcmc.grad_clip / gn).clamp(max=1.0)
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    return logp.detach(), grad.detach(), J_mat.detach(), Sigma.detach()

# ----------------------------- Sampler (multi-mode) -----------------------------

def mcmc_sgld_multimode(
    w: torch.Tensor, X: torch.Tensor, chi_mat: torch.Tensor,
    m_vec: torch.Tensor, kappa: float,
    mdl: ModelParams, saem: SAEMParams, mcmc: MCMCParams
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    step = mcmc.step_size
    for _ in range(mcmc.n_steps):
        _, grad, J_mat, Sigma = compute_logp_and_grad_multimode(
            w, X, chi_mat, m_vec, kappa, mdl, saem, mcmc
        )
        if mcmc.langevin_sqrt2:
            noise = torch.randn_like(w) * math.sqrt(2.0 * step)
            w = w + step * grad + noise
        else:
            noise = torch.randn_like(w) * math.sqrt(step)
            w = w + 0.5 * step * grad + noise

        if mcmc.clamp_w:
            w = torch.clamp(w, min=-mcmc.clamp_w, max=mcmc.clamp_w)
        step *= mcmc.step_decay

    with torch.no_grad():
        J_mat, Sigma = compute_J_Sigma_multimode(w, X, chi_mat, mdl.act)
    return w.detach(), J_mat.detach(), Sigma.detach()

# ----------------------------- SAEM Loop (vector m with noise modes) -----------------------------

def saem_solve_for_kappa_multimode(
    kappa: float, devices: List[int],
    BASE: Dict, SAVE_DIR: str, run_tag: str = ""
):
    # Unpack params
    mdl = ModelParams(
        d=BASE["d"], N=BASE["N"], k=BASE["k"],
        sigma_a=_get(BASE, "σa", "sigma_a", default=1.0),
        sigma_w=_get(BASE, "σw", "sigma_w", default=1.0),
        gamma=_get(BASE, "γ", "gamma", default=1.0),
        act=BASE.get("act", "relu")
    )
    mcmc = MCMCParams(
        R_inputs=BASE.get("R_inputs", 8192),
        n_chains_per_device=BASE.get("chains_per_device", 8192),
        n_steps=BASE.get("mcmc_steps", 30),
        step_size=BASE.get("mcmc_step_size", 5e-3),
        step_decay=BASE.get("mcmc_step_decay", 0.999),
        langevin_sqrt2=BASE.get("langevin_sqrt2", True),
        grad_clip=BASE.get("grad_clip", 10.0),
        clamp_w=BASE.get("clamp_w", 10.0),
    )
    saem = SAEMParams(
        max_iters=BASE.get("opt_steps", 4000),
        a0=BASE.get("a0", 0.5),
        t0=BASE.get("t0", 100.0),
        damping_m=BASE.get("damping_m", 1.0),
        eps_D=BASE.get("eps_D", 1e-6),
        print_every=BASE.get("print_every", 50),
    )

    # Teacher + noise modes
    M_noise = BASE.get("M_noise", 0)  # number of extra k-parity modes
    S = make_parity_indices(mdl.d, mdl.k, seed=BASE.get("teacher_seed", 0))
    A_noise = make_k_subsets(mdl.d, mdl.k, M_noise, S, seed=BASE.get("noise_seed", 1))
    modes = [S] + A_noise                      # modes[0] == S
    Mtot = 1 + M_noise

    # Per-device state
    per_dev = []
    total_chains = 0
    for di in devices if len(devices) > 0 else [-1]:
        device = torch.device(f"cuda:{di}") if di >= 0 and torch.cuda.is_available() else torch.device("cpu")
        X = make_boolean_batch(mcmc.R_inputs, mdl.d, device, seed=BASE.get("data_seed", 0))
        chi_mat = chi_of_modes(X, modes).to(device)  # (R, Mtot)
        chains = mcmc.n_chains_per_device

        # init w with variance g_w^2 = σ_w/d
        gw2 = mdl.sigma_w / mdl.d
        w = torch.randn(chains, mdl.d, device=device) * math.sqrt(gw2)

        per_dev.append({"device": device, "X": X, "chi_mat": chi_mat, "w": w})
        total_chains += chains

    # Init m-vector: m_S from m_init, others 0
    m_vec = torch.zeros(Mtot, dtype=torch.float32)
    m_vec[0] = float(BASE.get("m_init", 0.0))
    m_bar = m_vec.clone()

    history = []
    t_start = time.time()

    # Constants
    ga2 = mdl.sigma_a / (mdl.N ** mdl.gamma)
    Acoef = 1.0 / ga2  # = N^γ / σ_a

    for it in range(1, saem.max_iters + 1):
        # CPU accumulators (device-agnostic aggregation)
        A_sum = torch.zeros(Mtot, Mtot, dtype=torch.float64, device="cpu")  # E[(J J^T)/D]
        b_sum = torch.zeros(Mtot,        dtype=torch.float64, device="cpu")  # E[(J_S J)/D]
        n_sum = 0

        # E-step: refresh chains & collect pooled stats
        for slot in per_dev:
            X, chi_mat, w = slot["X"], slot["chi_mat"], slot["w"]
            device = slot["device"]
            w, J_mat, Sigma = mcmc_sgld_multimode(
                w, X, chi_mat, m_vec.to(device), kappa, mdl, saem, mcmc
            )
            slot["w"] = w

            # D on device
            D = (Acoef * (kappa**2) + Sigma).clamp_min(saem.eps_D)  # (B,)
            invD = (1.0 / D).to(torch.float64)                      # (B,) device

            # stats on device
            J64 = J_mat.to(torch.float64)                           # (B,M) device
            local_A = J64.T @ (J64 * invD[:, None])                 # (M,M) device
            local_b = ((J64[:, 0] * invD)[:, None] * J64).sum(dim=0)  # (M,) device

            # move to CPU to accumulate
            A_sum += local_A.detach().cpu()
            b_sum += local_b.detach().cpu()
            n_sum += J64.shape[0]

        # finalize pooled averages (CPU)
        A_hat = (A_sum / max(1, n_sum)).to(torch.float32)  # (M,M)
        b_hat = (b_sum / max(1, n_sum)).to(torch.float32)  # (M,)

        # Robbins–Monro step size
        a_t = saem.a0 / (it + saem.t0)

        # Vector fixed-point residual:
        # F(m) = (I + N A_hat) m - N b_hat
        I = torch.eye(Mtot, dtype=torch.float32)
        F = (I + mdl.N * A_hat) @ m_vec - mdl.N * b_hat

        # Parameter proposal
        m_new = m_vec - saem.damping_m * a_t * F

        # Optional: clip m_S for interpretability
        m_new[0] = float(min(max(m_new[0].item(), 0.0), 1.0))
        m_vec = m_new

        # Polyak–Ruppert average (optional)
        m_bar = ((it - 1) * m_bar + m_vec) / it

        if it % saem.print_every == 0 or it == 1:
            # Diagnostic fixed point: (I + N A_hat) m* = N b_hat
            try:
                m_fixed = torch.linalg.solve(I + mdl.N * A_hat, mdl.N * b_hat)
            except RuntimeError:
                m_fixed = torch.nan * torch.ones_like(m_vec)

            dt = time.time() - t_start
            msg = {
                "iter": it, "kappa": kappa,
                "m_S": float(m_vec[0].item()),
                "m_S_fixed": float(m_fixed[0].item()) if torch.isfinite(m_fixed[0]) else None,
                "m_noise_rms": float(m_vec[1:].pow(2).mean().sqrt().item()) if Mtot > 1 else 0.0,
                "time_s": round(dt, 2)
            }
            print(json.dumps(msg))
            history.append(msg)

    # Final snapshot
    result = {
        "kappa": kappa,
        "m_final": [float(x) for x in m_vec.tolist()],
        "m_bar":   [float(x) for x in m_bar.tolist()],
        "history": history[-10:],  # last few for quick glance
        "BASE": BASE
    }

    os.makedirs(SAVE_DIR, exist_ok=True)
    tag = run_tag if run_tag else f"kappa_{kappa:.6f}"
    out_path = os.path.join(SAVE_DIR, f"saem_multimode_result_{tag}.json")
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
        opt_steps=2000,
        # MCMC controls
        R_inputs=16384,
        chains_per_device=4096,
        mcmc_steps=600,
        mcmc_step_size=5e-3,
        mcmc_step_decay=0.999,
        langevin_sqrt2=True,
        grad_clip=1e5,
        clamp_w=100.0,
        # SAEM controls
        a0=0.5, t0=100.0, damping_m=1.0,
        eps_D=1e-6, print_every=5,
        # Seeds
        teacher_seed=0, data_seed=0, noise_seed=1,
        # init
        m_init=0.5,
        # number of tracked noise k-modes
        M_noise=50,
    )

    # Warm start from large -> small kappa
    kappa_list = sorted(np.logspace(np.log10(1e-4), np.log10(5e1), 15), reverse=True)

    # Choose a save directory you have write access to
    save_dir = "/home/goring/mean_field_langevin/MCMC_Pinf_1N_sparse/results/2008_d25k4_infm2_nn"
    os.makedirs(save_dir, exist_ok=True)

    run_tag_prefix = time.strftime("%Y%m%d_%H%M%S")

    # Warm start across kappas (vector m)
    m_ws = None

    for idx, kappa in enumerate(kappa_list):
        print(f"\n=== SAEM (multi-mode) for kappa={kappa} ===")
        if m_ws is not None:
            BASE["m_init"] = float(m_ws[0])  # warm-start teacher amplitude only
        tag = f"{run_tag_prefix}_k{idx}_{kappa:.6f}"
        result = saem_solve_for_kappa_multimode(
            kappa=kappa,
            devices=devices,
            BASE=BASE,
            SAVE_DIR=save_dir,
            run_tag=tag
        )
        m_ws = result["m_final"]
