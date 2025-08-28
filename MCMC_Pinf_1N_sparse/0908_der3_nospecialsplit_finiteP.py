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
    R_inputs: int = 8192                 # Monte Carlo samples for E_x[·]
    n_chains_per_device: int = 4096
    n_steps: int = 50
    step_size_w: float = 2e-3            # separate stepsizes (a is stiffer)
    step_size_a: float = 2e-4
    step_decay: float = 0.999
    langevin_sqrt2: bool = True
    grad_clip_w: float = 10.0
    grad_clip_a: float = 10.0
    clamp_w: float = 10.0
    clamp_a: float = 10.0

@dataclass
class SAEMParams:
    max_iters: int = 2000
    a0: float = 0.5
    t0: float = 100.0
    damping_m: float = 1.0
    print_every: int = 50
    resample_every: int = 0   # 0 = fixed MC pool; else refresh every K iters

# ----------------------------- E_x moments for χ_S -----------------------------

def compute_all_moments(
    w: torch.Tensor, X: torch.Tensor, chi_S_vec: torch.Tensor, act: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns per-chain tensors of shape (B,):
      J     = E_x[phi * chi_S]
      Sigma = E_x[phi^2]
      C     = Cov_x(chi_S*phi, phi^2)
      V2    = Var_x(phi^2)
    """
    z = X @ w.T                      # (R,B)
    phi = activation(z, act)         # (R,B)
    phi2 = phi * phi                 # (R,B)

    # Means
    J = (phi * chi_S_vec[:, None]).mean(dim=0)  # E[phi chi_S]
    Sigma = phi2.mean(dim=0)                    # E[phi^2]

    # Cov(chi_S*phi, phi^2)
    A = chi_S_vec[:, None] * phi                # (R,B)
    B = phi2                                    # (R,B)
    C = (A * B).mean(dim=0) - A.mean(dim=0) * B.mean(dim=0)

    # Var(phi^2)
    V2 = (phi2 * phi2).mean(dim=0) - Sigma * Sigma

    return J, Sigma, C, V2

# ----------------------------- Full finite-P logπ and grads (joint w, a) -----------------------------

def compute_logp_and_grads_full(
    w: torch.Tensor, a: torch.Tensor, X: torch.Tensor, chi_S_vec: torch.Tensor,
    m: float, kappa: float, P: int,
    mdl: ModelParams, mcmc: MCMCParams
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Dataset-free finite-P action (no TAP), simplified for χ_S:
      S = (N^γ/(2σ_a)) a^2 + (d/(2σ_w))||w||^2
          + (a^2/(2κ^2)) Σ - (a/(κ^2))(1-m) J
          - (a^2/(2 κ^4 P)) (1-m)^2 (Σ - J^2)
          + (a^3/(2 κ^4 P)) (1-m) C
          - (a^4/(8 κ^4 P)) V2
    log π = -S (per chain)
    """
    gw2 = mdl.sigma_w / mdl.d
    ga2 = mdl.sigma_a / (mdl.N ** mdl.gamma)
    Acoef = (mdl.N ** mdl.gamma) / mdl.sigma_a  # 1/ga2

    # Enable grads
    w = w.detach().requires_grad_(True)
    a = a.detach().requires_grad_(True)

    # Moments
    J, Sigma, C, V2 = compute_all_moments(w, X, chi_S_vec, mdl.act)

    # Priors
    S_prior = 0.5 * Acoef * (a * a) + 0.5 * (w * w).sum(dim=1) * (mdl.d / mdl.sigma_w)

    # Infinite-data mean-field
    S_inf = 0.5 * (a * a) * (Sigma / (kappa**2)) - (a * (1.0 - m) * J) / (kappa**2)

    # O(1/P) corrections
    bracket = (1.0 - m) ** 2 * (Sigma - J * J)
    S_a2 = -0.5   * (a * a)       * bracket / (kappa**4 * P)
    S_a3 = +0.5   * (a * a * a)   * (1.0 - m) * C / (kappa**4 * P)
    S_a4 = -0.125 * (a ** 4)      * V2 / (kappa**4 * P)

    S_total = S_prior + S_inf + S_a2 + S_a3 + S_a4
    logp = -S_total

    # grads
    grad_w, grad_a = torch.autograd.grad(
        logp.sum(), [w, a], create_graph=False, retain_graph=False
    )

    # clipping
    if mcmc.grad_clip_w and mcmc.grad_clip_w > 0:
        gnw = grad_w.norm(dim=1, keepdim=True).clamp_min(1e-12)
        grad_w = grad_w * (mcmc.grad_clip_w / gnw).clamp(max=1.0)
    if mcmc.grad_clip_a and mcmc.grad_clip_a > 0:
        gna = grad_a.abs().clamp_min(1e-12)
        grad_a = grad_a * (mcmc.grad_clip_a / gna).clamp(max=1.0)

    diag = {"J": J.detach(), "Sigma": Sigma.detach(), "C": C.detach(), "V2": V2.detach()}
    return logp.detach(), grad_w.detach(), grad_a.detach(), diag

# ----------------------------- Joint SGLD -----------------------------

def mcmc_sgld_joint(
    w: torch.Tensor, a: torch.Tensor, X: torch.Tensor, chi_S_vec: torch.Tensor,
    m: float, kappa: float, P: int,
    mdl: ModelParams, mcmc: MCMCParams
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    step_w = mcmc.step_size_w
    step_a = mcmc.step_size_a
    diag_last = {}
    for _ in range(mcmc.n_steps):
        _, gw, ga, diag = compute_logp_and_grads_full(
            w, a, X, chi_S_vec, m, kappa, P, mdl, mcmc
        )

        if mcmc.langevin_sqrt2:
            w = w + step_w * gw + torch.randn_like(w) * math.sqrt(2.0 * step_w)
            a = a + step_a * ga + torch.randn_like(a) * math.sqrt(2.0 * step_a)
        else:
            w = w + 0.5 * step_w * gw + torch.randn_like(w) * math.sqrt(step_w)
            a = a + 0.5 * step_a * ga + torch.randn_like(a) * math.sqrt(step_a)

        if mcmc.clamp_w:
            w = torch.clamp(w, -mcmc.clamp_w, mcmc.clamp_w)
        if mcmc.clamp_a:
            a = torch.clamp(a, -mcmc.clamp_a, mcmc.clamp_a)

        step_w *= mcmc.step_decay
        step_a *= mcmc.step_decay
        diag_last = diag
    return w.detach(), a.detach(), diag_last

# ----------------------------- SAEM Loop (m_S only) -----------------------------

def saem_solve_for_kappa_fullP(
    kappa: float, P: int, devices: List[int],
    BASE: Dict, SAVE_DIR: str, run_tag: str = ""
):
    # Unpack
    mdl = ModelParams(
        d=BASE["d"], N=BASE["N"], k=BASE["k"],
        sigma_a=BASE["σa"], sigma_w=BASE["σw"],
        gamma=BASE["γ"], act=BASE.get("act", "relu")
    )
    mcmc = MCMCParams(
        R_inputs=BASE.get("R_inputs", 8192),
        n_chains_per_device=BASE.get("chains_per_device", 4096),
        n_steps=BASE.get("mcmc_steps", 50),
        step_size_w=BASE.get("step_size_w", 2e-3),
        step_size_a=BASE.get("step_size_a", 2e-4),
        step_decay=BASE.get("mcmc_step_decay", 0.999),
        langevin_sqrt2=BASE.get("langevin_sqrt2", True),
        grad_clip_w=BASE.get("grad_clip_w", 10.0),
        grad_clip_a=BASE.get("grad_clip_a", 10.0),
        clamp_w=BASE.get("clamp_w", 10.0),
        clamp_a=BASE.get("clamp_a", 10.0),
    )
    saem = SAEMParams(
        max_iters=BASE.get("opt_steps", 2000),
        a0=BASE.get("a0", 0.5),
        t0=BASE.get("t0", 100.0),
        damping_m=BASE.get("damping_m", 1.0),
        print_every=BASE.get("print_every", 50),
        resample_every=BASE.get("resample_every", 0),
    )

    # Teacher
    S = make_parity_indices(mdl.d, mdl.k, seed=BASE.get("teacher_seed", 0))

    # Per-device state (dataset-free MC pool for expectations)
    per_dev = []
    for di in devices if len(devices) > 0 else [-1]:
        device = torch.device(f"cuda:{di}") if di >= 0 and torch.cuda.is_available() else torch.device("cpu")
        X = make_boolean_batch(mcmc.R_inputs, mdl.d, device, seed=BASE.get("data_seed", 0))
        chi_vec = chi_S_of_X(X, S)

        # init w ~ N(0, g_w^2 I), a ~ N(0, g_a^2)
        gw2 = BASE["σw"] / BASE["d"]
        ga2 = BASE["σa"] / (BASE["N"] ** BASE["γ"])
        chains = mcmc.n_chains_per_device
        w = torch.randn(chains, mdl.d, device=device) * math.sqrt(gw2)
        a = torch.randn(chains, device=device) * math.sqrt(ga2)

        per_dev.append({"device": device, "X": X, "chi_vec": chi_vec, "w": w, "a": a, "S": S})

    # m_S init
    m = BASE.get("m_init", 0.0)
    m_bar = m
    history = []
    t_start = time.time()

    for it in range(1, saem.max_iters + 1):
        # optional MC pool refresh
        if saem.resample_every and (it % saem.resample_every == 0):
            for slot in per_dev:
                slot["X"] = make_boolean_batch(mcmc.R_inputs, mdl.d, slot["device"], seed=BASE.get("data_seed", 0) + it)
                slot["chi_vec"] = chi_S_of_X(slot["X"], S)

        # E-step: sample (w,a); collect stats for m update
        aJ_sum, n_sum = 0.0, 0
        for slot in per_dev:
            X, chi_vec, w, a = slot["X"], slot["chi_vec"], slot["w"], slot["a"]
            w, a, _ = mcmc_sgld_joint(w, a, X, chi_vec, m, kappa, P, mdl, mcmc)
            slot["w"], slot["a"] = w, a

            with torch.no_grad():
                J, _, _, _ = compute_all_moments(w, X, chi_vec, mdl.act)
                aJ_sum += (a * J).sum().item()
                n_sum  += a.numel()

        aJ_mean = aJ_sum / max(1, n_sum)

        # Robbins–Monro step size
        a_t = saem.a0 / (it + saem.t0)

        # Update m via m* = N E[a J]
        target_m = mdl.N * aJ_mean
        m_new = (1 - saem.damping_m * a_t) * m + saem.damping_m * a_t * target_m
        m = float(min(max(m_new, 0.0), 1.0))
        m_bar = ((it-1)*m_bar + m) / it

        if it % saem.print_every == 0 or it == 1:
            dt = time.time() - t_start
            print(json.dumps({"iter": it, "kappa": kappa, "P": P, "m": m, "m_bar": m_bar, "time_s": round(dt,2)}))
            history.append({"iter": it, "m": m, "m_bar": m_bar, "time_s": round(dt,2)})

    # save
    result = {
        "kappa": kappa, "P": P,
        "m_final": m, "m_bar": m_bar,
        "history": history[-10:], "BASE": BASE
    }
    os.makedirs(SAVE_DIR, exist_ok=True)
    tag = run_tag if run_tag else f"kappa_{kappa:.6f}_P_{P}"
    out_path = os.path.join(SAVE_DIR, f"saem_fullFiniteP_result_{tag}.json")
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
        # MC pool for expectations (dataset-free)
        R_inputs=16384,
        resample_every=3,            # set >0 to refresh MC pool periodically
        # Chains / SGLD controls
        chains_per_device=2048,
        mcmc_steps=600,
        step_size_w=2e-4,
        step_size_a=5e-6,
        mcmc_step_decay=0.999,
        langevin_sqrt2=True,
        grad_clip_w=1e5,
        grad_clip_a=1e5,
        clamp_w=100.0,
        clamp_a=100.0,
        # SAEM
        a0=0.5, t0=100.0, damping_m=1.0, print_every=10,
        # Seeds
        teacher_seed=0, data_seed=0,
        # init
        m_init=0.2,
    )

    # choose grid(s)
    P_list = [10,100,250,500,750,1000,1500,2000,2500]
    kappa_list = [1e-3,1e-2]

    save_dir = "/home/goring/mean_field_langevin/MCMC_Pinf_1N_sparse/results/21_08_finiteP_nn"
    os.makedirs(save_dir, exist_ok=True)
    run_tag_prefix = time.strftime("%Y%m%d_%H%M%S")

    for kappa in kappa_list:
        m_ws = BASE.get("m_init", 0.0)
        for P in P_list:
            print(f"\n=== SAEM full finite-P (disorder-avg) for kappa={kappa}, P={P} ===")
            BASE["m_init"] = m_ws
            tag = f"{run_tag_prefix}_kap{kappa:.3e}_P{P}"
            res = saem_solve_for_kappa_fullP(
                kappa=kappa, P=P,
                devices=devices,
                BASE=BASE,
                SAVE_DIR=save_dir,
                run_tag=tag
            )
            m_ws = res["m_final"]
