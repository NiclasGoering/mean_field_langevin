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
    sigma_a: float = 1.0     # note: these are the σ's in your paper
    sigma_w: float = 1.0
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
    z = X @ w.T                # (R,B)
    phi = activation(z, act)   # (R,B)
    J = (phi * chi_S_vec[:, None]).mean(dim=0)  # (B,)
    Sigma = (phi * phi).mean(dim=0)             # (B,)
    return J, Sigma

# ----------------------------- E|N(μ,σ²)| helper -----------------------------

def expected_abs_normal(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """E|X| for X ~ N(mu, sigma^2)."""
    sigma_safe = torch.clamp(sigma, min=1e-12)
    alpha = torch.abs(mu) / (math.sqrt(2.0) * sigma_safe)
    term1 = sigma_safe * math.sqrt(2.0 / math.pi) * torch.exp(-alpha * alpha)
    term2 = torch.abs(mu) * torch.erf(alpha)
    return term1 + term2

# ----------------------------- logπ and grad -----------------------------

def compute_logp_and_grad(
    w: torch.Tensor, X: torch.Tensor, chi_S_vec: torch.Tensor,
    m: float, chi: float, kappa: float,
    mdl: ModelParams, saem: SAEMParams, mcmc: MCMCParams
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Correct scalings:
      g_w^2 = sigma_w / d
      g_a^2 = sigma_a / N^gamma
      A     = 1 / g_a^2  = N^gamma / sigma_a
      D     = A*kappa^2 + Sigma - (chi/N) J^2
      prior = - ||w||^2 / (2 g_w^2)
    """
    gw2 = mdl.sigma_w / mdl.d
    ga2 = mdl.sigma_a / (mdl.N ** mdl.gamma)
    A   = 1.0 / ga2

    w = w.detach().requires_grad_(True)

    J, Sigma = compute_J_Sigma(w, X, chi_S_vec, mdl.act)
    D = A * (kappa**2) + Sigma - (chi / mdl.N) * (J * J)
    D_safe = torch.clamp(D, min=saem.eps_D)

    # log prior: - ||w||^2 / (2 g_w^2)
    prior_term = -0.5 * (w * w).sum(dim=1) / gw2

    # terms from integrating out a
    log_det_term = -0.5 * torch.log(D_safe)
    data_term    = ((1.0 - m)**2) / (2.0 * (kappa**2)) * (J * J) / D_safe

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
    m: float, chi: float, kappa: float,
    mdl: ModelParams, saem: SAEMParams, mcmc: MCMCParams
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    step = mcmc.step_size
    for _ in range(mcmc.n_steps):
        _, grad, J, Sigma = compute_logp_and_grad(
            w, X, chi_S_vec, m, chi, kappa, mdl, saem, mcmc
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

# ----------------------------- 1D k-means splitter -----------------------------

@torch.no_grad()
def split_specialists(s: torch.Tensor, max_iter: int = 15) -> Tuple[torch.Tensor, torch.Tensor, float, Tuple[float,float]]:
    """
    1-D k-means on s (B,) with k=2.
    Returns boolean masks (specialist, background), threshold tau, and centers (c_low,c_high).
    """
    if s.numel() < 4:
        mask_s = s > s.mean()
        return mask_s, ~mask_s, float(s.mean().item()), (float(s.min()), float(s.max()))

    c1 = torch.quantile(s, 0.2)
    c2 = torch.quantile(s, 0.8)
    for _ in range(max_iter):
        d1 = (s - c1).abs()
        d2 = (s - c2).abs()
        assign2 = (d2 < d1)  # True -> cluster 2 (high)
        c2_new = s[assign2].mean() if assign2.any() else c2
        c1_new = s[~assign2].mean() if (~assign2).any() else c1
        if torch.allclose(c1, c1_new) and torch.allclose(c2, c2_new):
            break
        c1, c2 = c1_new, c2_new

    c_low = min(c1.item(), c2.item())
    c_high = max(c1.item(), c2.item())
    tau = 0.5 * (c_low + c_high)
    mask_s = s >= tau
    return mask_s, ~mask_s, float(tau), (float(c_low), float(c_high))

# ----------------------------- SAEM Loop (mixture) -----------------------------

def feasible_project(m: float, chi: float, mdl: ModelParams, rho_max_obs: float, saem: SAEMParams):
    if rho_max_obs <= 0:
        bound = float("inf")
    else:
        bound = (1.0 - saem.eps_proj) * mdl.N / rho_max_obs
    chi_proj = min(max(chi, 0.0), bound)
    m_proj = min(max(m, 0.0), 1.0)
    return m_proj, chi_proj, bound

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

        # init w with variance g_w^2
        gw2 = BASE["σw"] / BASE["d"]  # g_w^2
        w = torch.randn(chains, mdl.d, device=device) * math.sqrt(gw2)

        per_dev.append({"device": device, "X": X, "chi_vec": chi_vec, "w": w})
        total_chains += chains

    # Init params and PR averages
    m = BASE.get("m_init", 0.0)
    chi = BASE.get("chi_init", 1e-6)
    p = BASE.get("p_init", 0.01)  # small specialist fraction
    m_bar, chi_bar, p_bar = m, chi, p

    rho_max_obs = 1e-6
    history = []
    t_start = time.time()

    for it in range(1, saem.max_iters + 1):
        # accumulators per basin
        sum_g1_s = 0.0; sum_g2_s = 0.0; cnt_s = 0
        sum_g1_b = 0.0; sum_g2_b = 0.0; cnt_b = 0
        max_rho_batch = 0.0
        taus = []; centers = []

        # E-step: refresh chains & collect stats
        for slot in per_dev:
            device = slot["device"]
            X, chi_vec, w = slot["X"], slot["chi_vec"], slot["w"]
            w, J, Sigma = mcmc_sgld(w, X, chi_vec, m, chi, kappa, mdl, saem, mcmc)
            slot["w"] = w

            # A, D with corrected scalings
            ga2 = mdl.sigma_a / (mdl.N ** mdl.gamma)
            A = 1.0 / ga2
            D = A * (kappa**2) + Sigma - (chi / mdl.N) * (J * J)
            D = torch.clamp(D, min=saem.eps_D)

            # core statistics for TAP
            g1 = (J * J) / D
            g2 = ((J * J) * (J * J)) / ((kappa**2) * (D * D))

            # conditional a|w moments
            mu = (1.0 - m) * J / D
            sig2 = (kappa**2) / D
            sigma = torch.sqrt(torch.clamp(sig2, min=1e-24))
            E_abs_a = expected_abs_normal(mu, sigma)

            # split statistic based on *effective* contribution
            t = torch.abs(J) * E_abs_a

            mask_s, mask_b, tau, (c_low, c_high) = split_specialists(t)
            taus.append(tau); centers.append((c_low, c_high))

            if mask_s.any():
                sum_g1_s += g1[mask_s].sum().item()
                sum_g2_s += g2[mask_s].sum().item()
                cnt_s += int(mask_s.sum().item())
            if mask_b.any():
                sum_g1_b += g1[mask_b].sum().item()
                sum_g2_b += g2[mask_b].sum().item()
                cnt_b += int(mask_b.sum().item())

            # rho for chi bound (use J^2/Sigma)
            rho = (J * J / torch.clamp(Sigma, min=1e-12)).max().item()
            max_rho_batch = max(max_rho_batch, rho)

        rho_max_obs = max(rho_max_obs, max_rho_batch)

        # finalize per-basin averages
        g1_s = (sum_g1_s / cnt_s) if cnt_s > 0 else 0.0
        g2_s = (sum_g2_s / cnt_s) if cnt_s > 0 else 0.0
        g1_b = (sum_g1_b / cnt_b) if cnt_b > 0 else 0.0
        g2_b = (sum_g2_b / cnt_b) if cnt_b > 0 else 0.0

        p_emp = cnt_s / max(1, (cnt_s + cnt_b))

        # mixture TAP residuals
        g1_mix = p * g1_s + (1.0 - p) * g1_b
        g2_mix = p * g2_s + (1.0 - p) * g2_b

        a_t = saem.a0 / (it + saem.t0)

        F1 = m   - mdl.N * (1.0 - m) * g1_mix
        F2 = chi - mdl.N * (1.0 - m) * (1.0 - m) * g2_mix - mdl.N * g1_mix

        # lever rule for p (from F1=0): m = N(1-m)(p g1_s + (1-p) g1_b)
        denom = max(abs(g1_s - g1_b), 1e-12)
        p_star = ((m / max(1e-12, (mdl.N * (1.0 - m)))) - g1_b) / denom
        p_star = float(min(max(p_star, 0.0), 1.0))
        Fp = p - p_star

        # RM updates
        m_new   = m   - saem.damping * a_t * F1
        chi_new = chi - saem.damping * a_t * F2
        p_new   = p   - saem.damping * a_t * Fp

        # Projections
        m, chi, chi_bound = feasible_project(m_new, chi_new, mdl, rho_max_obs, saem)
        p = float(min(max(p_new, 0.0), 1.0))

        # PR averages
        m_bar   = ((it-1)*m_bar   + m  ) / it
        chi_bar = ((it-1)*chi_bar + chi) / it
        p_bar   = ((it-1)*p_bar   + p  ) / it

        if it % saem.print_every == 0 or it == 1:
            dt = time.time() - t_start
            msg = {
                "iter": it, "kappa": kappa,
                "m": m, "chi": chi, "p": p,
                "m_bar": m_bar, "chi_bar": chi_bar, "p_bar": p_bar,
                "g1_s": g1_s, "g1_b": g1_b, "g2_s": g2_s, "g2_b": g2_b,
                "g1_mix": g1_mix, "g2_mix": g2_mix,
                "p_emp": p_emp, "p_star": p_star,
                "tau_mean": float(sum(taus)/max(1,len(taus))),
                "centers_mean": tuple(float(sum(c[i] for c in centers)/max(1,len(centers))) for i in (0,1)),
                "rho_max_obs": rho_max_obs, "chi_bound": chi_bound,
                "time_s": round(dt, 2)
            }
            print(json.dumps(msg))
            history.append(msg)

    # Final snapshot
    result = {
        "kappa": kappa,
        "m_final": m, "chi_final": chi, "p_final": p,
        "m_bar": m_bar, "chi_bar": chi_bar, "p_bar": p_bar,
        "rho_max_obs": rho_max_obs,
        "history": history[-10:],
        "BASE": BASE
    }

    os.makedirs(SAVE_DIR, exist_ok=True)
    tag = run_tag if run_tag else f"kappa_{kappa:.6f}"
    out_path = os.path.join(SAVE_DIR, f"saem_mix_result_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[saved] {out_path}")
    return result

# ----------------------------- Main -----------------------------

if __name__ == "__main__":
    devices = check_gpu()

    BASE = dict(
        d=30, N=1024, k=4,
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
        # inits
        m_init=0.2, chi_init=1e-6, p_init=0.01,
    )

    # Warm start from large -> small kappa
    kappa_list = sorted(np.logspace(np.log10(1e-5), np.log10(5e1), 15), reverse=True)

    save_dir = "/home/goring/mean_field_langevin/MCMC_Pinf_1N_sparse/results/0708_d30k4_diagosntic4_mix3"
    os.makedirs(save_dir, exist_ok=True)

    run_tag_prefix = time.strftime("%Y%m%d_%H%M%S")

    # Warm start across kappas
    m_ws  = BASE.get("m_init", 0.0)
    chi_ws= BASE.get("chi_init", 1e-6)
    p_ws  = BASE.get("p_init", 0.01)

    for idx, kappa in enumerate(kappa_list):
        print(f"\n=== SAEM (mixture) for kappa={kappa} ===")
        BASE["m_init"] = m_ws
        BASE["chi_init"] = chi_ws
        BASE["p_init"] = p_ws
        tag = f"{run_tag_prefix}_k{idx}_{kappa:.6f}"
        result = saem_solve_for_kappa(
            kappa=kappa,
            devices=devices,
            BASE=BASE,
            SAVE_DIR=save_dir,
            run_tag=tag
        )
        m_ws, chi_ws, p_ws = result["m_final"], result["chi_final"], result["p_final"]
