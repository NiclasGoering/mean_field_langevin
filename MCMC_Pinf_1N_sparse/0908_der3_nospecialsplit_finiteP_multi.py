# multimode_quenched_feature_learning.py
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
    A_list = []
    seen = set()
    S_set = set(S.tolist())
    while len(A_list) < M:
        A = torch.randperm(d, generator=g)[:k].sort().values
        key = tuple(A.tolist())
        if key in seen:
            continue
        seen.add(key)
        if set(A.tolist()) != S_set:
            A_list.append(A)
    return A_list

def make_boolean_batch(R: int, d: int, device: torch.device, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.randint(low=0, high=2, size=(R, d), generator=g, device=device, dtype=torch.int8)
    return X.to(torch.float32) * 2.0 - 1.0

def chi_of_modes(X: torch.Tensor, modes: List[torch.Tensor]) -> torch.Tensor:
    """Return (R, M) matrix with columns chi_A(x) for each A in modes (modes[0] should be S)."""
    cols = [X[:, A.long()].prod(dim=1) for A in modes]
    return torch.stack(cols, dim=1)  # (R,M)

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
    R_inputs: int = 8192                 # MC samples for E_x[·]
    n_chains_per_device: int = 2048
    n_steps: int = 100
    step_size_w: float = 2e-4
    step_size_a: float = 1e-6
    step_decay: float = 0.999
    langevin_sqrt2: bool = True
    grad_clip_w: float = 10.0
    grad_clip_a: float = 10.0
    clamp_w: float = 5.0
    clamp_a: float = 3.0

@dataclass
class SAEMParams:
    max_iters: int = 1200
    a0: float = 0.5
    t0: float = 100.0
    damping_m: float = 1.0
    print_every: int = 50
    resample_every: int = 0   # 0 = fixed MC pool; else refresh every K iters

# ----------------------------- Multi-mode moments -----------------------------

@torch.no_grad()
def compute_diag_ratio(
    a: torch.Tensor, Sigma: torch.Tensor, J_S: torch.Tensor, sum_mJ: torch.Tensor,
    Var_g: torch.Tensor, Cov_g2: torch.Tensor, Var_phi2: torch.Tensor,
    kappa: float, P: int
) -> float:
    """Diagnostic: magnitude of 1/P corrections vs base MF terms."""
    base = torch.abs(0.5*(a*a)*Sigma/(kappa**2)) + torch.abs(a*(J_S - sum_mJ)/(kappa**2))
    corr = torch.abs(0.5*(a*a)*Var_g/(kappa**4*P)) \
         + torch.abs(0.5*(a*a*a)*Cov_g2/(kappa**4*P)) \
         + torch.abs(0.125*(a**4)*Var_phi2/(kappa**4*P))
    num = corr.mean().item()
    den = (base.mean().item() + 1e-12)
    return num/den

def compute_moments_multimode(
    w: torch.Tensor,                       # (B,d)
    X: torch.Tensor,                       # (R,d)
    chi_mat: torch.Tensor,                 # (R,M) with modes[0]=S
    m_vec: torch.Tensor,                   # (M,)
    act: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      J_mat    : (B, M) with J_A = E_x[phi * chi_A]
      Sigma    : (B,)   E_x[phi^2]
      Var_g    : (B,)   Var_x[phi * beta(x)]      where beta = chi_S - sum_A m_A chi_A
      Cov_g2   : (B,)   Cov_x[phi*beta, phi^2]
      Var_phi2 : (B,)   Var_x[phi^2]
    """
    R = X.shape[0]
    z   = X @ w.T                   # (R,B)
    phi = activation(z, act)        # (R,B)
    phi2 = phi * phi                # (R,B)

    # E[phi * chi_A] for all A: (B,M)
    J_mat = (phi.T @ chi_mat) / float(R)

    # E[phi^2]: (B,)
    Sigma = phi2.mean(dim=0)

    # residual beta(x)
    beta = chi_mat[:, 0] - (chi_mat @ m_vec)       # (R,)

    # g = phi * beta
    g = phi * beta[:, None]                        # (R,B)

    g_mean = g.mean(dim=0)                         # (B,)
    Var_g = (g * g).mean(dim=0) - g_mean * g_mean  # (B,)

    phi2_mean = Sigma
    Cov_g2 = (g * phi2).mean(dim=0) - g_mean * phi2_mean  # (B,)

    Var_phi2 = (phi2 * phi2).mean(dim=0) - phi2_mean * phi2_mean  # (B,)

    return J_mat, Sigma, Var_g, Cov_g2, Var_phi2

# ----------------------------- Log-density and grads (multimode; quenched/annealed toggle) -----------------------------

def compute_logp_and_grads_multimode(
    w: torch.Tensor, a: torch.Tensor,
    X: torch.Tensor, chi_mat: torch.Tensor,  # chi_mat has modes[0]=S
    m_vec: torch.Tensor,                     # (M,)
    kappa: float, P: int,
    mdl: ModelParams, mcmc: MCMCParams,
    correction_mode: str = "quenched"        # "quenched" (default) or "annealed"
):
    """
    Base MF:
      S_prior = (N^γ/(2σ_a)) a^2 + (d/(2σ_w))||w||^2
      S_inf   = (a^2/(2 κ^2)) Σ - (a/κ^2) [J_S - sum_A m_A J_A]
    Finite-P corrections (computed on population pool):
      residual β = χ_S - Σ_A m_A χ_A
      define g = φ β, then
      Var_g    = Var[g], Cov_g2 = Cov[g, φ^2], Var_phi2 = Var[φ^2]
    Signs:
      - "quenched":  +1/2 a^2 Var_g  -1/2 a^3 Cov_g2  +1/8 a^4 Var_phi2
      - "annealed": -1/2 a^2 Var_g  +1/2 a^3 Cov_g2  -1/8 a^4 Var_phi2
      (all divided by κ^4 P)
    """
    # constants
    Acoef = (mdl.N ** mdl.gamma) / mdl.sigma_a  # 1/g_a^2
    gw_inv2 = (mdl.d / mdl.sigma_w)             # 1/g_w^2

    w = w.detach().requires_grad_(True)
    a = a.detach().requires_grad_(True)

    # moments (multi-mode)
    J_mat, Sigma, Var_g, Cov_g2, Var_phi2 = compute_moments_multimode(
        w, X, chi_mat, m_vec, mdl.act
    )
    J_S = J_mat[:, 0]                            # (B,)
    sum_mJ = (J_mat * m_vec[None, :]).sum(dim=1) # (B,)

    # prior + infinite-data mean field
    S_prior = 0.5 * Acoef * (a * a) + 0.5 * gw_inv2 * (w * w).sum(dim=1)
    S_inf   = 0.5 * (a * a) * (Sigma / (kappa**2)) - (a * (J_S - sum_mJ)) / (kappa**2)

    # finite-P corrections (toggle)
    if correction_mode == "quenched":
        S_a2 = +0.5   * (a * a)     * (Var_g     / (kappa**4 * P))
        S_a3 = -0.5   * (a * a * a) * (Cov_g2    / (kappa**4 * P))
        S_a4 = +0.125 * (a ** 4)    * (Var_phi2  / (kappa**4 * P))
    elif correction_mode == "annealed":
        S_a2 = -0.5   * (a * a)     * (Var_g     / (kappa**4 * P))
        S_a3 = +0.5   * (a * a * a) * (Cov_g2    / (kappa**4 * P))
        S_a4 = -0.125 * (a ** 4)    * (Var_phi2  / (kappa**4 * P))
    else:
        raise ValueError("correction_mode must be 'quenched' or 'annealed'")

    S_total = S_prior + S_inf + S_a2 + S_a3 + S_a4
    logp = -S_total

    grad_w, grad_a = torch.autograd.grad(logp.sum(), [w, a], create_graph=False, retain_graph=False)

    # optional mild clipping (keeps target invariant measure; avoids blow-ups)
    if mcmc.grad_clip_w and mcmc.grad_clip_w > 0:
        gnw = grad_w.norm(dim=1, keepdim=True).clamp_min(1e-12)
        grad_w = grad_w * (mcmc.grad_clip_w / gnw).clamp(max=1.0)
    if mcmc.grad_clip_a and mcmc.grad_clip_a > 0:
        gna = grad_a.abs().clamp_min(1e-12)
        grad_a = grad_a * (mcmc.grad_clip_a / gna).clamp(max=1.0)

    # diagnostics for regime validity
    corr_ratio = compute_diag_ratio(a.detach(), Sigma.detach(), J_S.detach(), sum_mJ.detach(),
                                    Var_g.detach(), Cov_g2.detach(), Var_phi2.detach(), kappa, P)

    diag = {
        "J_mat": J_mat.detach(),
        "Sigma": Sigma.detach(),
        "Var_g": Var_g.detach(),
        "Cov_g2": Cov_g2.detach(),
        "Var_phi2": Var_phi2.detach(),
        "corr_ratio": corr_ratio
    }
    return logp.detach(), grad_w.detach(), grad_a.detach(), diag

# ----------------------------- Joint SGLD (constant metric) -----------------------------

def mcmc_sgld_joint_multimode(
    w: torch.Tensor, a: torch.Tensor,
    X: torch.Tensor, chi_mat: torch.Tensor,
    m_vec: torch.Tensor, kappa: float, P: int,
    mdl: ModelParams, mcmc: MCMCParams,
    correction_mode: str = "quenched"
):
    step_w = mcmc.step_size_w
    step_a = mcmc.step_size_a
    diag_last = {}
    for _ in range(mcmc.n_steps):
        _, gw, ga, diag = compute_logp_and_grads_multimode(
            w, a, X, chi_mat, m_vec, kappa, P, mdl, mcmc, correction_mode
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

# ----------------------------- SAEM loop (multimode) -----------------------------

def saem_multimode_for_kappa(
    kappa: float, P: int, M_noise: int, devices: List[int],
    BASE: Dict, SAVE_DIR: str, run_tag: str = ""
):
    mdl = ModelParams(
        d=_get(BASE, "d", 30),
        N=_get(BASE, "N", 1024),
        k=_get(BASE, "k", 4),
        sigma_a=_get(BASE, "sigma_a", 1.0),
        sigma_w=_get(BASE, "sigma_w", 1.0),
        gamma=_get(BASE, "gamma", 1.0),
        act=_get(BASE, "act", "relu"),
    )
    mcmc = MCMCParams(
        R_inputs=_get(BASE, "R_inputs", 8192),
        n_chains_per_device=_get(BASE, "chains_per_device", 2048),
        n_steps=_get(BASE, "mcmc_steps", 100),
        step_size_w=_get(BASE, "step_size_w", 2e-4),
        step_size_a=_get(BASE, "step_size_a", 1e-6),
        step_decay=_get(BASE, "mcmc_step_decay", 0.999),
        langevin_sqrt2=_get(BASE, "langevin_sqrt2", True),
        grad_clip_w=_get(BASE, "grad_clip_w", 10.0),
        grad_clip_a=_get(BASE, "grad_clip_a", 10.0),
        clamp_w=_get(BASE, "clamp_w", 5.0),
        clamp_a=_get(BASE, "clamp_a", 3.0),
    )
    saem = SAEMParams(
        max_iters=_get(BASE, "opt_steps", 1200),
        a0=_get(BASE, "a0", 0.5),
        t0=_get(BASE, "t0", 100.0),
        damping_m=_get(BASE, "damping_m", 1.0),
        print_every=_get(BASE, "print_every", 50),
        resample_every=_get(BASE, "resample_every", 0),
    )
    correction_mode = _get(BASE, "correction_mode", "quenched")  # "quenched" or "annealed"

    # Teacher signal and noise modes
    S = make_parity_indices(mdl.d, mdl.k, seed=_get(BASE, "teacher_seed", 0))
    A_noise = make_k_subsets(mdl.d, mdl.k, M_noise, S, seed=_get(BASE, "noise_seed", 1))
    modes = [S] + A_noise  # modes[0] == S
    Mtot = 1 + M_noise

    per_dev = []
    for di in devices if len(devices) > 0 else [-1]:
        device = torch.device(f"cuda:{di}") if di >= 0 and torch.cuda.is_available() else torch.device("cpu")

        # population MC pool for expectations (theory setting)
        X = make_boolean_batch(mcmc.R_inputs, mdl.d, device, seed=_get(BASE, "pool_seed", 0))
        chi_mat = chi_of_modes(X, modes)  # (R,Mtot)

        # init w ~ N(0, g_w^2 I), a ~ N(0, g_a^2)
        gw2 = mdl.sigma_w / mdl.d
        ga2 = mdl.sigma_a / (mdl.N ** mdl.gamma)
        chains = mcmc.n_chains_per_device
        w = torch.randn(chains, mdl.d, device=device) * math.sqrt(gw2)
        a = torch.randn(chains, device=device) * math.sqrt(ga2)

        per_dev.append({"device": device, "X": X, "chi_mat": chi_mat, "w": w, "a": a})

    # m-vector init (unbiased): all zeros
    m_vec = torch.zeros(Mtot, dtype=torch.float32)
    m_vec[0] = 0.5  # m_S = 1.0 (S is the teacher signal)
    m_bar = m_vec.clone()
    history = []
    t_start = time.time()

    for it in range(1, saem.max_iters + 1):
        # optional resample of MC pool
        if saem.resample_every and (it % saem.resample_every == 0):
            for slot in per_dev:
                slot["X"] = make_boolean_batch(
                    mcmc.R_inputs, mdl.d, slot["device"], seed=_get(BASE, "pool_seed", 0) + it
                )
                slot["chi_mat"] = chi_of_modes(slot["X"], modes)

        # E-step
        aJ_sum = torch.zeros(Mtot)
        n_sum = 0
        corr_ratios = []
        for slot in per_dev:
            X, chi_mat, w, a = slot["X"], slot["chi_mat"], slot["w"], slot["a"]
            w, a, diag = mcmc_sgld_joint_multimode(
                w, a, X, chi_mat, m_vec.to(X.device), kappa, P, mdl, mcmc, correction_mode
            )
            slot["w"], slot["a"] = w, a

            with torch.no_grad():
                # reuse last diag's J_mat to avoid recompute: but re-evaluate once for accuracy
                J_mat, _, _, _, _ = compute_moments_multimode(
                    w, X, chi_mat, m_vec.to(X.device), mdl.act
                )
                aJ_sum += (a[:, None] * J_mat).mean(dim=0).cpu()  # (M,)
                n_sum += 1

                if "corr_ratio" in diag:
                    corr_ratios.append(diag["corr_ratio"])

        aJ_mean = aJ_sum / max(1, n_sum)              # (M,)
        # Robbins–Monro rate
        a_t = saem.a0 / (it + saem.t0)

        # M-step: m_B = N E[a J_B]
        target_m = mdl.N * aJ_mean                     # (M,)
        m_vec = (1 - saem.damping_m * a_t) * m_vec + saem.damping_m * a_t * target_m
        # clip only m_S into [0,1] (interpretability of proxy); others free
        m_vec[0] = float(min(max(m_vec[0].item(), 0.0), 1.0))
        m_bar = (m_bar * (it - 1) + m_vec) / it

        if it % saem.print_every == 0 or it == 1:
            ge = 0.5 * (1.0 - float(m_vec[0].item()))**2  # generalization error proxy
            dt = time.time() - t_start
            corr_info = float(np.mean(corr_ratios)) if corr_ratios else 0.0
            msg = {
                "iter": it, "kappa": kappa, "P": P, "mode": correction_mode,
                "m_S": round(float(m_vec[0].item()), 6),
                "gen_err": round(ge, 6),
                "m_noise_rms": round(float(m_vec[1:].pow(2).mean().sqrt().item()), 6) if Mtot > 1 else 0.0,
                "corr_ratio": round(corr_info, 4),
                "time_s": round(dt, 2)
            }
            if corr_info > 0.1:
                msg["warn"] = "1/P corrections not small; expansion may be invalid."
            print(json.dumps(msg))
            history.append(msg)

    # save
    result = {
        "kappa": kappa, "P": P, "mode": correction_mode,
        "m_final": float(m_vec[0].item()),
        "gen_err": 0.5 * (1.0 - float(m_vec[0].item()))**2,
        "m_noise_rms": float(m_vec[1:].pow(2).mean().sqrt().item()) if Mtot > 1 else 0.0,
        "history": history[-10:], "BASE": BASE
    }
    os.makedirs(SAVE_DIR, exist_ok=True)
    tag = run_tag if run_tag else f"kappa_{kappa:.6f}_P_{P}"
    out_path = os.path.join(SAVE_DIR, f"saem_multimode_quenched_{tag}.json")
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
        sigma_a=1.0, sigma_w=1.0, gamma=1.0,
        act="relu",
        correction_mode="quenched",  # "quenched" (recommended) or "annealed"
        opt_steps=1000,
        # MC pool (population expectations)
        R_inputs=16384,
        resample_every=5,            # set >0 (e.g., 20) to refresh MC pool
        # Chains / SGLD
        chains_per_device=2048,
        mcmc_steps=400,
        step_size_w=2e-4,
        step_size_a=1e-6,
        mcmc_step_decay=0.999,
        langevin_sqrt2=True,
        grad_clip_w=10.0,
        grad_clip_a=10.0,
        clamp_w=5.0,
        clamp_a=5.0,
        # SAEM
        a0=0.5, t0=100.0, damping_m=1.0, print_every=10,
        # Seeds
        teacher_seed=0, pool_seed=0, noise_seed=1,
    )

    save_dir = "/home/goring/mean_field_langevin/MCMC_Pinf_1N_sparse/results/2108_finiteP_multi_nn"
    os.makedirs(save_dir, exist_ok=True)
    run_tag_prefix = time.strftime("%Y%m%d_%H%M%S")

    # example sweep
    P_list = [10,100,500,1000,1500,5000,10000]
    kappa_list = [1e-3,1e-2]
    M_noise = 1000   # increase for stronger degeneracy competition

    for kappa in kappa_list:
        for P in P_list:
            print(f"\n=== multimode SAEM (mode={BASE['correction_mode']}) for κ={kappa}, P={P}, M_noise={M_noise} ===")
            tag = f"{run_tag_prefix}_kap{kappa:.3e}_P{P}_M{M_noise}"
            _ = saem_multimode_for_kappa(
                kappa=kappa, P=P, M_noise=M_noise,
                devices=devices, BASE=BASE, SAVE_DIR=save_dir,
                run_tag=tag
            )
