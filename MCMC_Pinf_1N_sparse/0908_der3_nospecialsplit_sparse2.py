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
    damping_m: float = 1.0
    damping_chi: float = 2.0
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

# ----------------------------- logπ and grad (UNGATED) -----------------------------

def compute_logp_and_grad(
    w: torch.Tensor, X: torch.Tensor, chi_S_vec: torch.Tensor,
    m: float, chi: float, kappa: float,
    mdl: ModelParams, saem: SAEMParams, mcmc: MCMCParams
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Gibbs density for w after integrating out a:
      g_w^2 = sigma_w / d
      g_a^2 = sigma_a / N^gamma
      A     = 1 / g_a^2  = N^gamma / sigma_a
      D     = A*kappa^2 + Sigma - (chi/N) J^2
      logp  = -||w||^2/(2 g_w^2) - 0.5*log D + ((1-m)^2/(2 kappa^2)) * (J^2 / D)
    """
    gw2 = mdl.sigma_w / mdl.d
    ga2 = mdl.sigma_a / (mdl.N ** mdl.gamma)
    A   = 1.0 / ga2

    w = w.detach().requires_grad_(True)

    J, Sigma = compute_J_Sigma(w, X, chi_S_vec, mdl.act)
    D = A * (kappa**2) + Sigma - (chi / mdl.N) * (J * J)
    D_safe = torch.clamp(D, min=saem.eps_D)

    prior_term   = -0.5 * (w * w).sum(dim=1) / gw2
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

# ----------------------------- Feasibility (D>0) -----------------------------

def feasible_project(m: float, chi: float, mdl: ModelParams, rho_cap: float, saem: SAEMParams):
    """
    Enforce positivity of D using a per-iteration cap on rho:
        chi <= (1 - eps_proj) * N / rho_cap
    and 0 <= m <= 1.
    """
    import math

    if (rho_cap is None) or (rho_cap <= 0.0) or (not math.isfinite(rho_cap)):
        bound = float("inf")
    else:
        bound = (1.0 - saem.eps_proj) * mdl.N / rho_cap

    m_proj = min(max(m, 0.0), 1.0)
    if math.isfinite(bound):
        chi_proj = min(max(chi, 0.0), bound)
    else:
        chi_proj = max(chi, 0.0)

    return m_proj, chi_proj, bound

# ----------------------------- Sparsity helpers (u, r, beta) -----------------------------

def compute_u_s4(
    J: torch.Tensor, Sigma: torch.Tensor,
    m: float, chi: float, kappa: float, A: float, N: int, eps_D: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (u, D_safe, s4) where
      D  = A*kappa^2 + Sigma - (chi/N) * J^2
      u  = ((1-m)^2 / (2 kappa^2)) * (J^2 / D) - 0.5*log D
      s4 = (J^4) / (kappa^2 D^2)
    Operates on CPU tensors (callers can .to('cpu')).
    """
    JJ = J * J
    D  = A * (kappa**2) + Sigma - (chi / N) * JJ
    D_safe = torch.clamp(D, min=eps_D)
    u = ((1.0 - m)**2) / (2.0 * (kappa**2)) * (JJ / D_safe) - 0.5 * torch.log(D_safe)
    s4 = (JJ * JJ) / ((kappa**2) * (D_safe * D_safe))
    return u, D_safe, s4

def softmax_stable(x: torch.Tensor, beta: float) -> torch.Tensor:
    y = x * beta
    y = y - y.max()
    w = torch.exp(y)
    w_sum = w.sum()
    if w_sum <= 0.0 or not torch.isfinite(w_sum):
        # fallback to uniform
        return torch.full_like(x, 1.0 / x.numel())
    return w / w_sum

def replicon_value(beta: float, u_all: torch.Tensor, s4_all: torch.Tensor, N: int, m: float) -> float:
    # Use UN-SCALED softmax here (sum=1); replicon uses r^2, so absolute scale matters.
    r_sm = softmax_stable(u_all, beta)
    val = N * (1.0 - m) * (1.0 - m) * torch.mean((r_sm * r_sm) * s4_all)
    return float(val.item())

def solve_beta_replicon(u_all: torch.Tensor, s4_all: torch.Tensor, N: int, m: float,
                        beta_lo: float = 0.0, beta_hi: float = 200.0, tol: float = 1e-3) -> float:
    """
    Solve 1 = N(1-m)^2 * E[(r_sm)^2 * s4] for beta using bisection with r_sm = softmax(beta u).
    If even beta_hi cannot reach 1, clamp to beta_hi (max concentration allowed).
    """
    f_lo = replicon_value(beta_lo, u_all, s4_all, N, m)
    if f_lo >= 1.0:
        return beta_lo
    f_hi = replicon_value(beta_hi, u_all, s4_all, N, m)
    if f_hi <= 1.0:
        return beta_hi
    lo, hi = beta_lo, beta_hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f_mid = replicon_value(mid, u_all, s4_all, N, m)
        if abs(f_mid - 1.0) <= tol:
            return mid
        if f_mid < 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

# ----------------------------- SAEM Loop (pooled estimator with sparsity weighting) -----------------------------

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
        damping_m=BASE.get("damping_m", 1.0),
        damping_chi=BASE.get("damping_chi", 2.0),
        eps_D=BASE.get("eps_D", 1e-6),
        eps_proj=BASE.get("eps_proj", 1e-3),
        print_every=BASE.get("print_every", 50),
    )

    # robust ρ settings
    rho_quantile = float(BASE.get("rho_quantile", 0.999))
    use_rho_ema = bool(BASE.get("use_rho_ema", True))
    rho_ema_beta = float(BASE.get("rho_ema_beta", 0.9))

    # beta search bounds
    beta_hi_cfg = float(BASE.get("beta_hi", 200.0))
    beta_tol = float(BASE.get("beta_tol", 1e-3))

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

    # Init order parameters and PR averages
    m = BASE.get("m_init", 0.0)
    chi = BASE.get("chi_init", 1e-6)
    m_bar, chi_bar = m, chi

    history = []
    t_start = time.time()

    # Constants for D / rho calculation
    ga2 = mdl.sigma_a / (mdl.N ** mdl.gamma)
    A = 1.0 / ga2

    # EMA state
    rho_cap_prev = None
    last_rho_cap = None

    for it in range(1, saem.max_iters + 1):

        # ---- SGLD refresh (UNGATED dynamics) & collect J,Sigma ----
        g1_sum = 0.0
        g2_sum = 0.0
        n_sum  = 0
        rho_q_vals = []

        J_list = []
        Sigma_list = []

        for slot in per_dev:
            device = slot["device"]
            X, chi_vec, w = slot["X"], slot["chi_vec"], slot["w"]
            w, J, Sigma = mcmc_sgld(w, X, chi_vec, m, chi, kappa, mdl, saem, mcmc)
            slot["w"] = w
            slot["_last_J"] = J
            slot["_last_Sigma"] = Sigma

            J_list.append(J.detach().to("cpu"))
            Sigma_list.append(Sigma.detach().to("cpu"))

            # rho for chi bound (independent of sparsity)
            rho_vals = (J * J) / (A * (kappa**2) + Sigma)
            try:
                rho_q = torch.quantile(rho_vals, rho_quantile).item()
            except Exception:
                rho_q = rho_vals.max().item()
            rho_q_vals.append(rho_q)

        # Concatenate across devices on CPU
        J_all = torch.cat(J_list, dim=0)
        Sigma_all = torch.cat(Sigma_list, dim=0)
        B_total = J_all.numel()

        # ---- E-step sparsity: compute u, solve beta (replicon), form responsibilities ----
        u_all, D_all, s4_all = compute_u_s4(J_all, Sigma_all, m, chi, kappa, A, mdl.N, saem.eps_D)

        # Solve beta on UN-SCALED softmax (sum=1)
        beta = solve_beta_replicon(u_all, s4_all, mdl.N, m, beta_lo=0.0, beta_hi=beta_hi_cfg, tol=beta_tol)

        # r_sm sums to 1; then SCALE to mean 1 for TAP moments: r = B * r_sm
        r_sm = softmax_stable(u_all, beta)                # (B_total,), sum=1
        H = float(torch.sum(r_sm * r_sm).item())          # Herfindahl, in [1/B, 1]
        rho_eff = 1.0 / (B_total * H)                     # effective active fraction
        r_all = (B_total * r_sm).detach()                 # mean(r_all)=1

        # ---- Compute weighted TAP moments with scaled r ----
        JJ = J_all * J_all
        D_safe = D_all
        g1 = r_all * (JJ / D_safe)
        g2 = r_all * ((JJ * JJ) / ((kappa**2) * (D_safe * D_safe)))

        g1_mean = float(g1.mean().item())   # E[r * J^2 / D]
        g2_mean = float(g2.mean().item())   # E[r * J^4 / (k^2 D^2)]

        # Robbins–Monro step size
        a_t = saem.a0 / (it + saem.t0)

        # Fixed-point residuals with r-weighted stats
        F1 = m   - mdl.N * (1.0 - m) * g1_mean
        F2 = chi - mdl.N * (1.0 - m) * (1.0 - m) * g2_mean - mdl.N * g1_mean

        # Proposals
        m_new   = m   - saem.damping_m  * a_t * F1
        chi_new = chi - saem.damping_chi * a_t * F2

        # robust per-iteration rho cap (quantile), optional EMA
        rho_cap_cur = max(rho_q_vals) if len(rho_q_vals) > 0 else 0.0
        if use_rho_ema and (rho_cap_prev is not None):
            rho_cap = rho_ema_beta * rho_cap_prev + (1.0 - rho_ema_beta) * rho_cap_cur
        else:
            rho_cap = rho_cap_cur
        rho_cap_prev = rho_cap
        last_rho_cap = rho_cap

        # Projection for feasibility (D>0)
        m, chi, chi_bound = feasible_project(m_new, chi_new, mdl, rho_cap, saem)

        # Post-projection safety backoff
        for _ in range(5):
            any_bad = False
            for slot in per_dev:
                J = slot["_last_J"]
                Sigma = slot["_last_Sigma"]
                Dchk = A * (kappa**2) + Sigma - (chi / mdl.N) * (J * J)
                if (Dchk <= saem.eps_D).any():
                    any_bad = True
                    break
            if not any_bad:
                break
            chi *= 0.95

        # Polyak–Ruppert averages
        m_bar   = ((it-1)*m_bar   + m  ) / it
        chi_bar = ((it-1)*chi_bar + chi) / it

        # diagnostics
        Ng1 = mdl.N * g1_mean
        m_fixed = Ng1 / (1.0 + Ng1)

        # Replicon value on unscaled softmax (should be ~1 at solution)
        repl = replicon_value(beta, u_all, s4_all, mdl.N, m)

        if it % saem.print_every == 0 or it == 1:
            dt = time.time() - t_start
            msg = {
                "iter": it, "kappa": kappa,
                "m": m, "chi": chi,
                "m_bar": m_bar, "chi_bar": chi_bar,
                "g1_mean": g1_mean, "g2_mean": g2_mean,
                "Ng1": Ng1, "m_fixed": m_fixed,
                "beta": beta, "rho_eff": rho_eff, "H": H,
                "replicon": repl,
                "rho_cap": last_rho_cap, "rho_quantile": rho_quantile,
                "chi_bound": chi_bound,
                "chi_at_bound": float(1.0 if (chi_bound < float("inf") and chi >= 0.999999 * chi_bound) else 0.0),
                "time_s": round(dt, 2)
            }
            print(json.dumps(msg))
            history.append(msg)

    # Final snapshot
    result = {
        "kappa": kappa,
        "m_final": m, "chi_final": chi,
        "m_bar": m_bar, "chi_bar": chi_bar,
        "history": history[-10:],  # last few for quick glance
        "BASE": BASE
    }

    os.makedirs(SAVE_DIR, exist_ok=True)
    tag = run_tag if run_tag else f"kappa_{kappa:.6f}"
    out_path = os.path.join(SAVE_DIR, f"saem_sparseE_result_{tag}.json")
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
        opt_steps=1000,
        # MCMC controls
        R_inputs=16384,
        chains_per_device=4096,
        mcmc_steps=600,
        mcmc_step_size=3e-3,
        mcmc_step_decay=0.999,
        langevin_sqrt2=True,
        grad_clip=1e5,
        clamp_w=100.0,
        # SAEM controls
        a0=0.5, t0=100.0, damping_m=1.0, damping_chi=2.0,
        eps_D=1e-6, eps_proj=1e-3, print_every=5,
        # Sparsity (replicon) options
        beta_hi=200.0,
        beta_tol=1e-3,
        # Robust feasibility
        rho_quantile=0.999, use_rho_ema=True, rho_ema_beta=0.9,
        # Seeds
        teacher_seed=0, data_seed=0,
        # inits
        m_init=0.5, chi_init=1e-6,
    )

    # Warm start from large -> small kappa
    kappa_list = sorted(np.logspace(np.log10(1e-4), np.log10(5e1), 15), reverse=True)

    save_dir = "/home/goring/mean_field_langevin/MCMC_Pinf_1N_sparse/results/2008_d25k4_sparseE_only"
    os.makedirs(save_dir, exist_ok=True)

    run_tag_prefix = time.strftime("%Y%m%d_%H%M%S")

    # Warm start across kappas
    m_ws  = BASE.get("m_init", 0.0)
    chi_ws= BASE.get("chi_init", 1e-6)

    for idx, kappa in enumerate(kappa_list):
        print(f"\n=== SAEM (sparse E-step) for kappa={kappa} ===")
        BASE["m_init"] = m_ws
        BASE["chi_init"] = chi_ws
        tag = f"{run_tag_prefix}_k{idx}_{kappa:.6f}"
        result = saem_solve_for_kappa(
            kappa=kappa,
            devices=devices,
            BASE=BASE,
            SAVE_DIR=save_dir,
            run_tag=tag
        )
        m_ws, chi_ws = result["m_final"], result["chi_final"]
