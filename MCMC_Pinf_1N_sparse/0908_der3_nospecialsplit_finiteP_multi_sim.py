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
    cols = [X[:, A.long()].prod(dim=1) for A in modes]
    return torch.stack(cols, dim=1)  # (R,M)

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
    sigma_a: float = 1.0     # σ_a
    sigma_w: float = 1.0     # σ_w
    gamma: float = 1.0
    act: str = "relu"

@dataclass
class MCMCParams:
    n_chains_per_device: int = 2048
    n_steps: int = 200
    step_size: float = 5e-3
    step_decay: float = 0.999
    langevin_sqrt2: bool = True
    grad_clip: float = 1e5
    clamp_w: float = 10.0

@dataclass
class SAEMParams:
    max_iters: int = 1200
    a0: float = 0.5
    t0: float = 100.0
    damping_m: float = 1.0
    eps_D: float = 1e-9
    print_every: int = 20
    project_mS: bool = True
    m_noise_clip: float = 3.0
    ridge_lambda: float = 1e-6  # tiny ridge for the SPD linear system

# ----------------------------- Dataset moments (means!) -----------------------------

def compute_J_Sigma(
    w: torch.Tensor, X: torch.Tensor, chi_mat: torch.Tensor, act: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    J_mat: (B,M), J_A = mean_p[phi(x_p;w) * chi_A(x_p)]
    Sigma: (B,),  mean_p[phi^2]
    """
    P = X.shape[0]
    z   = X @ w.T                   # (P,B)
    phi = activation(z, act)        # (P,B)
    phi2 = phi * phi
    J_mat = (phi.T @ chi_mat) / float(P)     # (B,M)
    Sigma = phi2.mean(dim=0)                 # (B,)
    return J_mat, Sigma

# ----------------------------- logπ(w) and grad (a integrated out) -----------------------------

def compute_logp_and_grad_w(
    w: torch.Tensor,
    X: torch.Tensor, chi_mat: torch.Tensor,
    m_vec: torch.Tensor, kappa: float,
    mdl: ModelParams, saem: SAEMParams
):
    """
    Exact finite-P objective with a integrated out:
      g_w^2 = σ_w / d,  g_a^2 = σ_a / N^γ,  A = 1/g_a^2 = N^γ/σ_a
      D = A κ^2 + Σ
      J_β = J_S - m^T J
      log p(w) = -||w||^2/(2g_w^2) - 1/2 log D + (J_β^2)/(2 κ^2 D)
    """
    gw2 = mdl.sigma_w / mdl.d
    ga2 = mdl.sigma_a / (mdl.N ** mdl.gamma)
    A   = 1.0 / ga2

    w = w.detach().requires_grad_(True)
    J_mat, Sigma = compute_J_Sigma(w, X, chi_mat, mdl.act)
    J_beta = J_mat[:, 0] - (J_mat * m_vec[None, :]).sum(dim=1)  # (B,)

    D = A * (kappa**2) + Sigma
    D_safe = torch.clamp(D, min=saem.eps_D)

    prior_term = -0.5 * (w * w).sum(dim=1) / gw2
    log_det    = -0.5 * torch.log(D_safe)
    data_term  = 0.5 * (J_beta * J_beta) / ((kappa**2) * D_safe)
    logp = prior_term + log_det + data_term

    grad_w = torch.autograd.grad(logp.sum(), w, create_graph=False, retain_graph=False)[0]
    if not torch.isfinite(grad_w).all():
        grad_w = torch.where(torch.isfinite(grad_w), grad_w, torch.zeros_like(grad_w))
    return logp.detach(), grad_w.detach(), J_mat.detach(), Sigma.detach()

# ----------------------------- SGLD over w -----------------------------

def mcmc_sgld_w(
    w: torch.Tensor, X: torch.Tensor, chi_mat: torch.Tensor,
    m_vec: torch.Tensor, kappa: float,
    mdl: ModelParams, saem: SAEMParams, mcmc: MCMCParams
):
    step = mcmc.step_size
    diag_last = {}
    for _ in range(mcmc.n_steps):
        _, grad_w, J_mat, Sigma = compute_logp_and_grad_w(
            w, X, chi_mat, m_vec.to(w.device), kappa, mdl, saem
        )
        # optional clipping
        if mcmc.grad_clip and mcmc.grad_clip > 0:
            gn = grad_w.norm(dim=1, keepdim=True).clamp_min(1e-12)
            grad_w = grad_w * (mcmc.grad_clip / gn).clamp(max=1.0)

        if mcmc.langevin_sqrt2:
            w = w + step * grad_w + torch.randn_like(w) * math.sqrt(2.0 * step)
        else:
            w = w + 0.5 * step * grad_w + torch.randn_like(w) * math.sqrt(step)

        if mcmc.clamp_w:
            w = torch.clamp(w, -mcmc.clamp_w, mcmc.clamp_w)

        step *= mcmc.step_decay
        diag_last = {"J_mat": J_mat, "Sigma": Sigma}
    return w.detach(), diag_last

# ----------------------------- SAEM loop with linear m-step -----------------------------

def saem_multimode_quenched_exact_linear_m(
    kappa: float, P: int, M_noise: int, devices: List[int],
    BASE: Dict, SAVE_DIR: str, run_tag: str = ""
):
    mdl = ModelParams(
        d=_get(BASE, "d", 25), N=_get(BASE, "N", 1024), k=_get(BASE, "k", 4),
        sigma_a=_get(BASE, "σa", 1.0), sigma_w=_get(BASE, "σw", 1.0),
        gamma=_get(BASE, "γ", 1.0), act=_get(BASE, "act", "relu")
    )
    mcmc = MCMCParams(
        n_chains_per_device=_get(BASE, "chains_per_device", 2048),
        n_steps=_get(BASE, "mcmc_steps", 200),
        step_size=_get(BASE, "mcmc_step_size", 5e-3),
        step_decay=_get(BASE, "mcmc_step_decay", 0.999),
        langevin_sqrt2=_get(BASE, "langevin_sqrt2", True),
        grad_clip=_get(BASE, "grad_clip", 1e5),
        clamp_w=_get(BASE, "clamp_w", 10.0),
    )
    saem = SAEMParams(
        max_iters=_get(BASE, "opt_steps", 1200),
        a0=_get(BASE, "a0", 0.5),
        t0=_get(BASE, "t0", 100.0),
        damping_m=_get(BASE, "damping_m", 1.0),
        eps_D=_get(BASE, "eps_D", 1e-9),
        print_every=_get(BASE, "print_every", 20),
        project_mS=_get(BASE, "project_mS", True),
        m_noise_clip=_get(BASE, "m_noise_clip", 3.0),
        ridge_lambda=_get(BASE, "ridge_lambda", 1e-6),
    )

    # teacher + noise modes
    S = make_parity_indices(mdl.d, mdl.k, seed=_get(BASE, "teacher_seed", 0))
    A_noise = make_k_subsets(mdl.d, mdl.k, M_noise, S, seed=_get(BASE, "noise_seed", 1))
    modes = [S] + A_noise
    Mtot = 1 + M_noise

    # per-device state (fixed quenched dataset of size P)
    per_dev = []
    for di in devices if len(devices) > 0 else [-1]:
        device = torch.device(f"cuda:{di}") if di >= 0 and torch.cuda.is_available() else torch.device("cpu")
        X = make_boolean_batch(P, mdl.d, device, seed=_get(BASE, "data_seed", 0))
        chi_mat = chi_of_modes(X, modes)  # (P,Mtot)

        # init w ~ N(0, g_w^2 I)
        gw2 = mdl.sigma_w / mdl.d
        chains = mcmc.n_chains_per_device
        w = torch.randn(chains, mdl.d, device=device) * math.sqrt(gw2)

        per_dev.append({"device": device, "X": X, "chi_mat": chi_mat, "w": w})

    # init m-vector
    m_vec = torch.zeros(Mtot, dtype=torch.float32)
    m_vec[0] = _get(BASE, "m_init", 0.5)
    m_bar = m_vec.clone()

    history = []
    t_start = time.time()

    Acoef = (mdl.N**mdl.gamma) / mdl.sigma_a  # so D = Acoef*kappa^2 + Sigma

    for it in range(1, saem.max_iters + 1):
        # CPU master buffers for linear system (avoid device mismatch)
        B_mat = torch.zeros((Mtot, Mtot), dtype=torch.float64, device="cpu")
        b_vec = torch.zeros(Mtot, dtype=torch.float64, device="cpu")
        n_slots = 0

        # E-step: update w by SGLD, accumulate B,b
        for slot in per_dev:
            X, chi_mat, w = slot["X"], slot["chi_mat"], slot["w"]
            w, _ = mcmc_sgld_w(w, X, chi_mat, m_vec.to(w.device), kappa, mdl, saem, mcmc)
            slot["w"] = w

            with torch.no_grad():
                # all tensors below are on this device
                J_mat, Sigma = compute_J_Sigma(w, X, chi_mat, mdl.act)   # (B,M),(B,)
                D = Acoef * (kappa**2) + Sigma                           # (B,)
                D = torch.clamp(D, min=saem.eps_D)
                invD = 1.0 / D                                           # (B,)

                # Compute per-device contributions (on device), then move to CPU
                # B = E[(1/D) JJ^T] = (J^T @ (J * invD[:,None])) / B
                B_contrib = (J_mat.T @ (J_mat * invD[:, None])) / J_mat.shape[0]
                # b = E[(1/D) J_S J] = (J^T @ (J_S * invD)) / B
                b_contrib = (J_mat.T @ (J_mat[:, 0] * invD)) / J_mat.shape[0]

                B_mat += B_contrib.double().cpu()
                b_vec += b_contrib.double().cpu()
                n_slots += 1

        # average across devices if multiple
        if n_slots > 1:
            B_mat /= n_slots
            b_vec /= n_slots

        # Solve (I + N * B) m* = N * b   with ridge (CPU, double)
        I = torch.eye(Mtot, dtype=torch.float64, device="cpu")
        lhs = I + (mdl.N * B_mat) + saem.ridge_lambda * I
        rhs = (mdl.N * b_vec)

        try:
            L = torch.linalg.cholesky(lhs)
            m_star = torch.cholesky_solve(rhs.unsqueeze(1), L).squeeze(1)  # (M,)
        except Exception:
            m_star = torch.linalg.solve(lhs, rhs)

        m_star = m_star.to(m_vec.dtype)  # CPU float32

        # SAEM averaging (on CPU)
        a_t = saem.a0 / (it + saem.t0)
        m_vec = (1 - saem.damping_m * a_t) * m_vec + saem.damping_m * a_t * m_star

        # keep teacher overlap in [0,1]; softly clip noise
        if saem.project_mS:
            m_vec[0] = float(min(max(m_vec[0].item(), 0.0), 1.0))
        if saem.m_noise_clip and saem.m_noise_clip > 0:
            m_vec[1:] = m_vec[1:].clamp(-saem.m_noise_clip, saem.m_noise_clip)

        m_bar = (m_bar * (it - 1) + m_vec) / it

        if it % saem.print_every == 0 or it == 1:
            ge = 0.5 * (1.0 - float(m_vec[0].item()))**2
            dt = time.time() - t_start
            msg = {
                "iter": it, "kappa": kappa, "P": P,
                "m_S": round(float(m_vec[0].item()), 6),
                "gen_err": round(ge, 6),
                "m_noise_rms": round(float(m_vec[1:].pow(2).mean().sqrt().item()), 6) if Mtot > 1 else 0.0,
                "time_s": round(dt, 2)
            }
            print(json.dumps(msg))
            history.append(msg)

    result = {
        "kappa": kappa, "P": P, "mode": "multi-quenched-exact-linearM",
        "m_final": float(m_vec[0].item()),
        "gen_err": 0.5 * (1.0 - float(m_vec[0].item()))**2,
        "m_noise_rms": float(m_vec[1:].pow(2).mean().sqrt().item()) if Mtot > 1 else 0.0,
        "history_tail": history[-10:], "BASE": BASE
    }
    os.makedirs(SAVE_DIR, exist_ok=True)
    tag = run_tag if run_tag else f"kappa_{kappa:.6f}_P_{P}"
    out_path = os.path.join(SAVE_DIR, f"saem_multimode_quenched_exact_linearM_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[saved] {out_path}")
    return result

# ----------------------------- Learning curve helper -----------------------------

def run_learning_curve(
    kappa: float, P_list: List[int], n_replicates: int, M_noise: int,
    BASE: Dict, devices: List[int], SAVE_DIR: str, run_tag_prefix: str
):
    curve = []
    for P in P_list:
        reps = []
        for r in range(n_replicates):
            BASE_rep = dict(BASE)
            BASE_rep["data_seed"] = _get(BASE, "data_seed", 0) + 1000*r
            tag = f"{run_tag_prefix}_kap{kappa:.3e}_P{P}_M{M_noise}_rep{r}"
            res = saem_multimode_quenched_exact_linear_m(
                kappa=kappa, P=P, M_noise=M_noise, devices=devices,
                BASE=BASE_rep, SAVE_DIR=SAVE_DIR, run_tag=tag
            )
            reps.append(res)

        ge = [x["gen_err"] for x in reps]
        ms = [x["m_final"] for x in reps]
        mn = [x["m_noise_rms"] for x in reps]
        summary = {
            "kappa": kappa, "P": P, "M_noise": M_noise, "n_replicates": n_replicates,
            "gen_err_mean": float(np.mean(ge)),
            "gen_err_std": float(np.std(ge, ddof=1)) if n_replicates > 1 else 0.0,
            "mS_mean": float(np.mean(ms)),
            "mS_std": float(np.std(ms, ddof=1)) if n_replicates > 1 else 0.0,
            "m_noise_rms_mean": float(np.mean(mn)),
            "m_noise_rms_std": float(np.std(mn, ddof=1)) if n_replicates > 1 else 0.0
        }
        print("[curve point]", json.dumps(summary))
        curve.append(summary)

    curve_path = os.path.join(SAVE_DIR, f"learning_curve_{run_tag_prefix}_kap{kappa:.3e}_M{M_noise}.json")
    with open(curve_path, "w") as f:
        json.dump(curve, f, indent=2)
    print(f"[saved learning curve] {curve_path}")
    return curve

# ----------------------------- Main -----------------------------

if __name__ == "__main__":
    set_seed(42)
    devices = check_gpu()

    BASE = dict(
        d=25, N=1024, k=4,
        σa=1.0, σw=1.0, γ=1.0,
        act="relu",
        opt_steps=5000,
        # SGLD over w
        chains_per_device=2048*2,
        mcmc_steps=800,
        mcmc_step_size=5e-3,
        mcmc_step_decay=0.999,
        langevin_sqrt2=True,
        grad_clip=1e5,
        clamp_w=20.0,
        # SAEM / linear M-step
        a0=0.5, t0=100.0, damping_m=1.0,
        eps_D=1e-9, print_every=20,
        ridge_lambda=5e-5,
        # Seeds
        teacher_seed=0, noise_seed=1, data_seed=0,
        # init
        m_init=0.5,
        project_mS=True,
        m_noise_clip=5.0,
    )

    save_dir = "/home/goring/mean_field_langevin/MCMC_Pinf_1N_sparse/results/2108_finiteP_multi_nn_linear1"
    os.makedirs(save_dir, exist_ok=True)
    run_tag_prefix = time.strftime("%Y%m%d_%H%M%S")

    # Example sweep (learning curve)
    kappa = 1e-3
    P_list = [750]
    M_noise = 512
    n_replicates = 1

    _ = run_learning_curve(
        kappa=kappa, P_list=P_list, n_replicates=n_replicates,
        M_noise=M_noise, BASE=BASE, devices=devices,
        SAVE_DIR=save_dir, run_tag_prefix=run_tag_prefix
    )
