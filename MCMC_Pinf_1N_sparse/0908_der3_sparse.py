import os, json, time, math, random
from typing import Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
import torch

# ============================= Utils =============================

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

# ========================= Core math =============================

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
    sigma_a: float = 1.0     # σ_a
    sigma_w: float = 1.0     # σ_w
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

@dataclass
class SparsityParams:
    beta_a: float = 1.0   # Beta prior for π (a=1,b=1 => uniform)
    beta_b: float = 1.0
    pi_floor: float = 1e-6

# ========================= J, Σ, responsibilities =================

def compute_J_Sigma(
    w: torch.Tensor, X: torch.Tensor, chi_S_vec: torch.Tensor, act: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    z = X @ w.T                # (R,B)
    phi = activation(z, act)   # (R,B)
    J = (phi * chi_S_vec[:, None]).mean(dim=0)  # (B,)
    Sigma = (phi * phi).mean(dim=0)             # (B,)
    return J, Sigma

# ====================== log π(w) and gradient =====================

def compute_logp_and_grad_spikeslab(
    w: torch.Tensor, X: torch.Tensor, chi_S_vec: torch.Tensor,
    m: float, chi: float, kappa: float, pi: float,
    mdl: ModelParams, saem: SAEMParams, sp: SparsityParams, mcmc: MCMCParams
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Spike-and-slab integration over a, with unknown π (learned self-consistently).
    g_w^2 = σ_w / d
    g_a^2 = σ_a / N^γ
    A     = 1 / g_a^2 = N^γ / σ_a
    D     = A*kappa^2 + Sigma - (chi/N) J^2
    log p(w) = -||w||^2/(2 g_w^2) + log( (1-π) + π * α^{-1/2} * exp(β^2/(2α)) )
      where α = D / κ^2, β = (1-m) J / κ^2
    """
    gw2 = mdl.sigma_w / mdl.d
    ga2 = mdl.sigma_a / (mdl.N ** mdl.gamma)
    A   = 1.0 / ga2

    w = w.detach().requires_grad_(True)

    J, Sigma = compute_J_Sigma(w, X, chi_S_vec, mdl.act)
    D = A * (kappa**2) + Sigma - (chi / mdl.N) * (J * J)
    D_safe = torch.clamp(D, min=saem.eps_D)

    # α, β for the "on" Gaussian slab
    alpha = D_safe / (kappa**2)
    beta  = (1.0 - m) * J / (kappa**2)

    # tensor π on the right device/dtype
    pi_t = torch.tensor(pi, device=w.device, dtype=w.dtype)
    pi_t = torch.clamp(pi_t, sp.pi_floor, 1.0 - sp.pi_floor)

    # prior over w
    prior_term = -0.5 * (w * w).sum(dim=1) / gw2

    # log-sum over z∈{0,1}: log( (1-π) + π * α^{-1/2} * exp(β^2/(2α)) )
    log_on  = -0.5 * torch.log(alpha) + 0.5 * (beta * beta) / alpha
    log_off = torch.zeros_like(log_on)
    log1m_pi = torch.log1p(-pi_t)
    log_pi   = torch.log(pi_t)

    log_mix = torch.logaddexp(log1m_pi + log_off, log_pi + log_on)
    logp = prior_term + log_mix

    # SGLD gradient wrt w
    grad = torch.autograd.grad(logp.sum(), w, create_graph=False, retain_graph=False)[0]
    if mcmc.grad_clip and mcmc.grad_clip > 0:
        gn = grad.norm(dim=1, keepdim=True).clamp_min(1e-12)
        grad = grad * (mcmc.grad_clip / gn).clamp(max=1.0)
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

    # responsibilities r = P(z=1 | w, π) = sigmoid( log π + log_on - log(1-π) )
    logits = log_pi + log_on - log1m_pi
    r = torch.sigmoid(logits)

    return logp.detach(), grad.detach(), J.detach(), Sigma.detach(), r.detach()

# ============================= Sampler ============================

def mcmc_sgld_spikeslab(
    w: torch.Tensor, X: torch.Tensor, chi_S_vec: torch.Tensor,
    m: float, chi: float, kappa: float, pi: float,
    mdl: ModelParams, saem: SAEMParams, sp: SparsityParams, mcmc: MCMCParams
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    step = mcmc.step_size
    r_out = None
    for _ in range(mcmc.n_steps):
        _, grad, J, Sigma, r = compute_logp_and_grad_spikeslab(
            w, X, chi_S_vec, m, chi, kappa, pi, mdl, saem, sp, mcmc
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
        r_out = r  # last responsibilities
    return w.detach(), J.detach(), Sigma.detach(), r_out.detach()

# ===================== Feasibility (D>0) ==========================

def feasible_project(m: float, chi: float, mdl: ModelParams, rho_max_obs: float, saem: SAEMParams):
    """
    Enforce positivity of D: chi <= (1 - eps_proj) * N / rho_max
    and 0 <= m <= 1.
    """
    if rho_max_obs <= 0:
        bound = float("inf")
    else:
        bound = (1.0 - saem.eps_proj) * mdl.N / rho_max_obs
    chi_proj = min(max(chi, 0.0), bound)
    m_proj = min(max(m, 0.0), 1.0)
    return m_proj, chi_proj, bound

# ========================= SAEM main loop =========================

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
    sp = SparsityParams(
        beta_a=BASE.get("beta_a", 1.0),
        beta_b=BASE.get("beta_b", 1.0),
        pi_floor=BASE.get("pi_floor", 1e-6)
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

    # Init order parameters (and π) + PR averages
    m   = BASE.get("m_init", 0.0)
    chi = BASE.get("chi_init", 1e-6)
    pi  = BASE.get("pi_init", 0.01)   # initial guess; will be learned

    m_bar, chi_bar, pi_bar = m, chi, pi

    rho_max_obs = 1e-6
    history = []
    t_start = time.time()

    for it in range(1, saem.max_iters + 1):
        # accumulators (pooled over all chains / devices)
        g1_sum = 0.0
        g2_sum = 0.0
        r_sum  = 0.0
        n_sum  = 0
        max_rho_batch = 0.0

        # E-step-ish: refresh chains & collect pooled stats
        for slot in per_dev:
            X, chi_vec, w = slot["X"], slot["chi_vec"], slot["w"]

            w, J, Sigma, r = mcmc_sgld_spikeslab(
                w, X, chi_vec, m, chi, kappa, pi, mdl, saem, sp, mcmc
            )
            slot["w"] = w

            # scalars used in g1/g2
            ga2 = mdl.sigma_a / (mdl.N ** mdl.gamma)
            A = 1.0 / ga2
            D = A * (kappa**2) + Sigma - (chi / mdl.N) * (J * J)
            D = torch.clamp(D, min=saem.eps_D)

            g1 = r * (J * J) / D
            g2 = r * ((J * J) * (J * J)) / ((kappa**2) * (D * D))

            g1_sum += g1.sum().item()
            g2_sum += g2.sum().item()
            r_sum  += r.sum().item()
            n_sum  += g1.numel()

            # rho for chi bound (use J^2/Sigma)
            rho = (J * J / torch.clamp(Sigma, min=1e-12)).max().item()
            max_rho_batch = max(max_rho_batch, rho)

        rho_max_obs = max(rho_max_obs, max_rho_batch)

        # finalize pooled averages
        g1_mean = g1_sum / max(1, n_sum)
        g2_mean = g2_sum / max(1, n_sum)
        r_mean  = r_sum  / max(1, n_sum)

        # Robbins–Monro step size
        a_t = saem.a0 / (it + saem.t0)

        # Fixed-point residuals (weighted by r inside g1,g2)
        F1 = m   - mdl.N * (1.0 - m) * g1_mean
        F2 = chi - mdl.N * (1.0 - m) * (1.0 - m) * g2_mean - mdl.N * g1_mean

        # Parameter updates
        m_new   = m   - saem.damping * a_t * F1
        chi_new = chi - saem.damping * a_t * F2

        # π update (M-step with Beta prior)
        # π* = (a-1 + Σ r) / (a+b-2 + n)
        num = (sp.beta_a - 1.0) + r_sum
        den = (sp.beta_a + sp.beta_b - 2.0) + n_sum
        pi_star = num / max(den, 1e-12)
        # damp for stability
        pi_new = (1.0 - saem.damping * a_t) * pi + (saem.damping * a_t) * float(pi_star)
        pi_new = float(min(max(pi_new, sp.pi_floor), 1.0 - sp.pi_floor))

        # Projections for feasibility (D>0) and bounds on m, chi
        m, chi, chi_bound = feasible_project(m_new, chi_new, mdl, rho_max_obs, saem)
        pi = pi_new

        # Polyak–Ruppert averages (optional)
        m_bar   = ((it-1)*m_bar   + m  ) / it
        chi_bar = ((it-1)*chi_bar + chi) / it
        pi_bar  = ((it-1)*pi_bar  + pi ) / it

        if it % saem.print_every == 0 or it == 1:
            dt = time.time() - t_start
            msg = {
                "iter": it, "kappa": kappa,
                "m": m, "chi": chi, "pi": pi,
                "m_bar": m_bar, "chi_bar": chi_bar, "pi_bar": pi_bar,
                "g1_mean": g1_mean, "g2_mean": g2_mean, "r_mean": r_mean,
                "rho_max_obs": rho_max_obs, "chi_bound": chi_bound,
                "time_s": round(dt, 2)
            }
            print(json.dumps(msg))
            history.append(msg)

    # Final snapshot
    result = {
        "kappa": kappa,
        "m_final": m, "chi_final": chi, "pi_final": pi,
        "m_bar": m_bar, "chi_bar": chi_bar, "pi_bar": pi_bar,
        "rho_max_obs": rho_max_obs,
        "history": history[-10:],  # last few iterations
        "BASE": BASE
    }

    os.makedirs(SAVE_DIR, exist_ok=True)
    tag = run_tag if run_tag else f"kappa_{kappa:.6f}"
    out_path = os.path.join(SAVE_DIR, f"saem_spikeslab_result_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[saved] {out_path}")
    return result

# ============================== Main =============================

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
        # Inits
        m_init=0.5, chi_init=1e-6, pi_init=0.01,
        # π prior (optional): Beta(a,b); (1,1) means π ≈ mean(r)
        beta_a=1.0, beta_b=1.0,
        pi_floor=1e-6,
    )

    # Sweep κ from large → small (warm starts)
    kappa_list = np.logspace(np.log10(1e-4), np.log10(5e1), 30)

    save_dir = "/home/goring/mean_field_langevin/MCMC_Pinf_1N_sparse/results/1808_d40k4_nospecialst_sparse"
    os.makedirs(save_dir, exist_ok=True)

    run_tag_prefix = time.strftime("%Y%m%d_%H%M%S")

    # Warm start across κ
    m_ws  = BASE.get("m_init", 0.0)
    chi_ws= BASE.get("chi_init", 1e-6)
    pi_ws = BASE.get("pi_init", 0.01)

    for idx, kappa in enumerate(kappa_list):
        print(f"\n=== SAEM (spike&slab, self-consistent π) for kappa={kappa} ===")
        BASE["m_init"]  = m_ws
        BASE["chi_init"] = chi_ws
        BASE["pi_init"]  = pi_ws
        tag = f"{run_tag_prefix}_k{idx}_{kappa:.6f}"
        result = saem_solve_for_kappa(
            kappa=kappa,
            devices=devices,
            BASE=BASE,
            SAVE_DIR=save_dir,
            run_tag=tag
        )
        m_ws, chi_ws, pi_ws = result["m_final"], result["chi_final"], result["pi_final"]
