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

# ----------------------------- Core math -----------------------------

def make_parity_indices(d: int, k: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    S = torch.randperm(d, generator=g)[:k]
    return S.sort().values

def make_boolean_batch(R: int, d: int, device: torch.device, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.randint(low=0, high=2, size=(R, d), generator=g, device=device, dtype=torch.int8)
    X = X.to(torch.float32) * 2.0 - 1.0
    return X

def chi_S_of_X(X: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    return X[:, S.long()].prod(dim=1)

def activation(z: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "relu":
        return torch.relu(z)
    elif kind == "tanh":
        return torch.tanh(z)
    else:
        raise ValueError(f"Unknown activation: {kind}")

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
    damping: float = 1.0
    eps_D: float = 1e-6
    eps_proj: float = 1e-3
    print_every: int = 50

# ----------------------------- J, Σ -----------------------------

def compute_J_Sigma(
    w: torch.Tensor, X: torch.Tensor, chi_S_vec: torch.Tensor, act: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    z = X @ w.T                 # (R,B)
    phi = activation(z, act)    # (R,B)
    J = (phi * chi_S_vec[:, None]).mean(dim=0)  # (B,)  -> J_S(w)
    Sigma = (phi * phi).mean(dim=0)             # (B,)  -> Σ(w)
    return J, Sigma

# ----------------------------- logπ and grad (∞-width / no TAP) -----------------------------

def compute_logp_and_grad(
    w: torch.Tensor, X: torch.Tensor, chi_S_vec: torch.Tensor,
    m: float, kappa: float,
    mdl: ModelParams, saem: SAEMParams, mcmc: MCMCParams
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Infinite-width, infinite-data effective single-neuron density after integrating out 'a':
      g_w^2 = sigma_w / d
      g_a^2 = sigma_a / N^gamma
      A     = 1 / g_a^2  = N^gamma / sigma_a
      D∞    = A*kappa^2 + Sigma
      For parity teacher (single mode): J_Y = J_S, and sum_A m_A J_A = m J_S
      logp  = -||w||^2/(2 g_w^2) - 0.5*log D∞ + ((J_Y - m J_S)^2) / (2 kappa^2 D∞)
    """
    gw2 = mdl.sigma_w / mdl.d
    ga2 = mdl.sigma_a / (mdl.N ** mdl.gamma)
    A   = 1.0 / ga2

    w = w.detach().requires_grad_(True)

    J, Sigma = compute_J_Sigma(w, X, chi_S_vec, mdl.act)
    D = A * (kappa**2) + Sigma                      # <-- no 1/N correction
    D_safe = torch.clamp(D, min=saem.eps_D)

    # Parity teacher: J_Y = J
    JY_minus_mJ = J - m * J                         # = (1 - m)*J

    # log prior: - ||w||^2 / (2 g_w^2)
    prior_term = -0.5 * (w * w).sum(dim=1) / gw2

    # terms from integrating out a (no TAP corrections)
    log_det_term = -0.5 * torch.log(D_safe)
    data_term    = (JY_minus_mJ * JY_minus_mJ) / (2.0 * (kappa**2) * D_safe)

    logp = prior_term + log_det_term + data_term

    grad = torch.autograd.grad(logp.sum(), w, create_graph=False, retain_graph=False)[0]
    if mcmc.grad_clip and mcmc.grad_clip > 0:
        gn = grad.norm(dim=1, keepdim=True).clamp_min(1e-12)
        grad = grad * (mcmc.grad_clip / gn).clamp(max=1.0)
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    return logp.detach(), grad.detach(), J.detach(), Sigma.detach()

# ----------------------------- Sampler -----------------------------

def mcmc_sgld(
    w: torch.Tensor, X: torch.Tensor, chi_S_vec: torch.Tensor,
    m: float, kappa: float,
    mdl: ModelParams, saem: SAEMParams, mcmc: MCMCParams
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    step = mcmc.step_size
    for _ in range(mcmc.n_steps):
        _, grad, J, Sigma = compute_logp_and_grad(
            w, X, chi_S_vec, m, kappa, mdl, saem, mcmc
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
    return w.detach(), J.detach(), Sigma.detach()

# ----------------------------- Feasibility (trivial here) -----------------------------

def feasible_project(m: float, mdl: ModelParams, saem: SAEMParams):
    """
    In the infinite-width/no-TAP setting D∞ = A*kappa^2 + Σ >= A*kappa^2 > 0,
    so we only need 0 <= m <= 1.
    Return shape kept similar to original (bound is dummy 'inf').
    """
    m_proj = min(max(m, 0.0), 1.0)
    return m_proj, float("inf")

# ----------------------------- SAEM Loop (single fixed-point for m) -----------------------------

def saem_solve_for_kappa(
    kappa: float, devices: List[int],
    BASE: Dict, SAVE_DIR: str, run_tag: str = ""
):
    # Unpack params
    mdl = ModelParams(
        d=BASE["d"], N=BASE["N"], k=BASE["k"],
        sigma_a=BASE["σa"], sigma_w=BASE["σw"],
        gamma=BASE["γ"], act=BASE.get("act", "relu")
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
        damping=BASE.get("damping", 1.0),
        eps_D=BASE.get("eps_D", 1e-6),
        eps_proj=BASE.get("eps_proj", 1e-3),
        print_every=BASE.get("print_every", 50),
    )

    # Teacher indices
    S = make_parity_indices(mdl.d, mdl.k, seed=BASE.get("teacher_seed", 0))

    # Per-device state
    per_dev = []
    total_chains = 0
    for di in devices if len(devices) > 0 else [-1]:
        device = torch.device(f"cuda:{di}") if di >= 0 and torch.cuda.is_available() else torch.device("cpu")
        X = make_boolean_batch(mcmc.R_inputs, mdl.d, device, seed=BASE.get("data_seed", 0))
        chi_vec = chi_S_of_X(X, S)
        chains = mcmc.n_chains_per_device

        # init w with variance g_w^2 = σ_w/d
        gw2 = BASE["σw"] / BASE["d"]
        w = torch.randn(chains, mdl.d, device=device) * math.sqrt(gw2)

        per_dev.append({"device": device, "X": X, "chi_vec": chi_vec, "w": w})
        total_chains += chains

    # Init order parameter and PR average
    m = BASE.get("m_init", 0.0)
    m_bar = m

    history = []
    t_start = time.time()

    for it in range(1, saem.max_iters + 1):
        # accumulators (pooled over all chains / devices)
        g1_sum = 0.0
        g2_sum = 0.0
        n_sum  = 0

        # E-step: refresh chains & collect pooled stats
        for slot in per_dev:
            device = slot["device"]
            X, chi_vec, w = slot["X"], slot["chi_vec"], slot["w"]

            w, J, Sigma = mcmc_sgld(w, X, chi_vec, m, kappa, mdl, saem, mcmc)
            slot["w"] = w

            # A, D∞ with corrected scalings (no TAP)
            ga2 = mdl.sigma_a / (mdl.N ** mdl.gamma)
            A = 1.0 / ga2
            D = A * (kappa**2) + Sigma
            D = torch.clamp(D, min=saem.eps_D)

            # pooled statistics for fixed-point
            g1 = (J * J) / D
            # g2 kept for logging parity with TAP code (not used in update here)
            g2 = ((J * J) * (J * J)) / ((kappa**2) * (D * D))

            g1_sum += g1.sum().item()
            g2_sum += g2.sum().item()
            n_sum  += g1.numel()

        # finalize pooled averages
        g1_mean = g1_sum / max(1, n_sum)
        g2_mean = g2_sum / max(1, n_sum)

        # Robbins–Monro step size
        a_t = saem.a0 / (it + saem.t0)

        # Fixed-point residual (single equation)
        F1 = m - mdl.N * (1.0 - m) * g1_mean

        # Parameter update for m only
        m_new = m - saem.damping * a_t * F1

        # Projection 0<=m<=1 (keep interface similar)
        m, _m_bound = feasible_project(m_new, mdl, saem)

        # Polyak–Ruppert average (optional, kept for logging)
        m_bar = ((it-1)*m_bar + m) / it

        if it % saem.print_every == 0 or it == 1:
            dt = time.time() - t_start
            msg = {
                "iter": it, "kappa": kappa,
                "m": m, "m_bar": m_bar,
                "g1_mean": g1_mean, "g2_mean": g2_mean,
                "time_s": round(dt, 2)
            }
            print(json.dumps(msg))
            history.append(msg)

    # Final snapshot
    result = {
        "kappa": kappa,
        "m_final": m,
        "m_bar": m_bar,
        "history": history[-10:],  # last few for quick glance
        "BASE": BASE
    }

    os.makedirs(SAVE_DIR, exist_ok=True)
    tag = run_tag if run_tag else f"kappa_{kappa:.6f}"
    out_path = os.path.join(SAVE_DIR, f"saem_infwidth_result_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[saved] {out_path}")
    return result

# ----------------------------- Main -----------------------------

if __name__ == "__main__":
    set_seed(42)
    devices = check_gpu()

    BASE = dict(
        d=40, N=1024, k=4,
        σa=1.0, σw=1.0, γ=1.0,
        act="relu",
        opt_steps=4000,
        # MCMC controls
        R_inputs=8192,
        chains_per_device=8192,
        mcmc_steps=30,
        mcmc_step_size=5e-3,
        mcmc_step_decay=0.999,
        langevin_sqrt2=True,
        grad_clip=10.0,
        clamp_w=10.0,
        # SAEM controls
        a0=0.5, t0=100.0, damping=1.0,
        eps_D=1e-6, eps_proj=1e-3, print_every=50,
        # Seeds
        teacher_seed=0, data_seed=0,
        # init
        m_init=0.2,
    )

    # Warm start from large -> small kappa (you can reverse if you prefer)
    kappa_list = np.logspace(np.log10(1e-4), np.log10(5e1), 30)

    # Save directory
    save_dir = "/home/goring/mean_field_langevin/MCMC_Pinf_1N_sparse/results/1808_d40k4_nospecialst_infN"
    os.makedirs(save_dir, exist_ok=True)

    run_tag_prefix = time.strftime("%Y%m%d_%H%M%S")

    # Warm start across kappas
    m_ws = BASE.get("m_init", 0.0)

    for idx, kappa in enumerate(kappa_list):
        print(f"\n=== SAEM (∞-width) for kappa={kappa} ===")
        BASE["m_init"] = m_ws
        tag = f"{run_tag_prefix}_k{idx}_{kappa:.6f}"
        result = saem_solve_for_kappa(
            kappa=kappa,
            devices=devices,
            BASE=BASE,
            SAVE_DIR=save_dir,
            run_tag=tag
        )
        m_ws = result["m_final"]
