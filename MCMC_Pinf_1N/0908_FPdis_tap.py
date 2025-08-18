
import os, json, time, math, random
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

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
    torch.cuda.manual_seed_all(seed)


# ----------------------------- Core math -----------------------------

def make_parity_indices(d: int, k: int, seed: int = 0) -> torch.Tensor:
    """Choose the k indices for the parity teacher (Walsh set S)."""
    g = torch.Generator().manual_seed(seed)
    S = torch.randperm(d, generator=g)[:k]
    return S.sort().values  # ascending

def make_boolean_batch(R: int, d: int, device: torch.device, seed: int = 0) -> torch.Tensor:
    """Generate R iid inputs in {+1,-1}^d as a tensor (R,d)."""
    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.randint(low=0, high=2, size=(R, d), generator=g, device=device, dtype=torch.int8)
    X = X.to(torch.float32) * 2.0 - 1.0
    return X

def chi_S_of_X(X: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """Compute the Walsh parity χ_S(x) for a batch X: returns (R,) float32 in {+1,-1}."""
    # X: (R,d), S: (k,)
    # product over the selected coordinates
    chi = X[:, S.long()].prod(dim=1)
    return chi

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
    sigma_a: float = 1.0
    sigma_w: float = 1.0
    gamma: float = 1.0
    act: str = "relu"

@dataclass
class MCMCParams:
    R_inputs: int = 8192           # number of Monte Carlo inputs per device
    n_chains_per_device: int = 8192
    n_steps: int = 30              # SGLD steps per SAEM iteration
    step_size: float = 5e-3
    step_decay: float = 0.999
    langevin_sqrt2: bool = True    # use sqrt(2*η) noise (ULA)
    grad_clip: float = 10.0

@dataclass
class SAEMParams:
    max_iters: int = 4000
    a0: float = 0.5
    t0: float = 100.0
    damping: float = 1.0           # extra damping on parameter update (0,1]
    eps_D: float = 1e-6
    eps_proj: float = 1e-3
    print_every: int = 50

# ----------------------------- J, Σ, D, logπ -----------------------------

def compute_J_Sigma(
    w: torch.Tensor,         # (B,d)
    X: torch.Tensor,         # (R,d)
    chi_S_vec: torch.Tensor, # (R,)
    act: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Monte Carlo estimates:
      J(w)   = E_x [ φ(w^T x) χ_S(x) ]
      Σ(w)   = E_x [ φ(w^T x)^2 ]
    Returns J: (B,), Sigma: (B,)
    """
    # z = X @ w^T => (R,B)
    z = X @ w.T
    phi = activation(z, act)
    # χ_S broadcast to (R,1)
    J = (phi * chi_S_vec[:, None]).mean(dim=0)   # (B,)
    Sigma = (phi * phi).mean(dim=0)              # (B,)
    return J, Sigma

def compute_logp_and_grad(
    w: torch.Tensor, X: torch.Tensor, chi_S_vec: torch.Tensor,
    m: float, chi: float, kappa: float,
    mdl: ModelParams, saem: SAEMParams, mcmc: MCMCParams
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      logp:   (B,)
      grad:   (B,d)  gradient of logp wrt w
      J, Sigma: (B,), (B,)
    """
    d = mdl.d
    sigma_w = mdl.sigma_w
    A = mdl.N**mdl.gamma / mdl.sigma_a  # A = N^γ / σ_a
    w = w.detach().requires_grad_(True)

    J, Sigma = compute_J_Sigma(w, X, chi_S_vec, mdl.act)
    D = A * (kappa**2) + Sigma - (chi / mdl.N) * (J * J)
    D_safe = torch.clamp(D, min=saem.eps_D)

    prior_term = - (d / (2.0 * (sigma_w**2))) * (w * w).sum(dim=1)
    log_det_term = -0.5 * torch.log(D_safe)
    data_term = ((1.0 - m)**2) / (2.0 * (kappa**2)) * (J * J) / D_safe
    logp = prior_term + log_det_term + data_term

    # get grad directly as a tensor
    grad = torch.autograd.grad(logp.sum(), w, create_graph=False, retain_graph=False)[0]

    # per-sample norm clip (no .grad accesses)
    if mcmc.grad_clip is not None and mcmc.grad_clip > 0:
        gn = grad.norm(dim=1, keepdim=True).clamp_min(1e-12)
        scale = (mcmc.grad_clip / gn).clamp(max=1.0)
        grad = grad * scale

    # be paranoid about NaNs/infs from log/clamp boundaries
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

    return logp.detach(), grad.detach(), J.detach(), Sigma.detach()
# ----------------------------- Sampler -----------------------------

def mcmc_sgld(
    w: torch.Tensor, X: torch.Tensor, chi_S_vec: torch.Tensor,
    m: float, chi: float, kappa: float,
    mdl: ModelParams, saem: SAEMParams, mcmc: MCMCParams
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One block of SGLD steps (unadjusted Langevin) to refresh chains.
    Returns new w, and the last J, Sigma computed.
    """
    step = mcmc.step_size
    for _ in range(mcmc.n_steps):
        logp, grad, J, Sigma = compute_logp_and_grad(
            w, X, chi_S_vec, m, chi, kappa, mdl, saem, mcmc
        )
        if mcmc.langevin_sqrt2:
            noise = torch.randn_like(w) * math.sqrt(2.0 * step)
            w = w + step * grad + noise
        else:
            noise = torch.randn_like(w) * math.sqrt(step)
            w = w + 0.5 * step * grad + noise  # MALA-like update without MH correction

        # small shrinkage to avoid runaway (helps with precision issues)
        # this does not change the target in the small-step limit
        w = torch.clamp(w, min=-10.0, max=10.0)

        step *= mcmc.step_decay
    return w.detach(), J.detach(), Sigma.detach()

# ----------------------------- SAEM Loop -----------------------------

def feasible_project(m: float, chi: float, mdl: ModelParams, rho_max_obs: float, saem: SAEMParams):
    # Project chi into [0, (1 - eps) N / rho_max_obs]
    if rho_max_obs <= 0:
        bound = float("inf")
    else:
        bound = (1.0 - saem.eps_proj) * mdl.N / rho_max_obs
    chi_proj = min(max(chi, 0.0), bound)
    m_proj = min(max(m, 0.0), 1.0)  # keep m within [0,1]
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
        grad_clip=BASE.get("grad_clip", 10.0)
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
    for di in devices if len(devices)>0 else [-1]:
        device = torch.device(f"cuda:{di}") if di >= 0 and torch.cuda.is_available() else torch.device("cpu")
        # inputs
        X = make_boolean_batch(mcmc.R_inputs, mdl.d, device, seed=BASE.get("data_seed", 0))
        chi_vec = chi_S_of_X(X, S)
        # chains
        chains = mcmc.n_chains_per_device
        w = torch.randn(chains, mdl.d, device=device) * math.sqrt(mdl.sigma_w**2 / mdl.d)
        per_dev.append({"device": device, "X": X, "chi_vec": chi_vec, "w": w})
        total_chains += chains

    # Init params and PR averages
    m = BASE.get("m_init", 0.0)
    chi = BASE.get("chi_init", 1e-6)
    m_bar, chi_bar = m, chi

    # Running diagnostics
    rho_max_obs = 1e-6
    history = []
    t_start = time.time()

    for it in range(1, saem.max_iters + 1):
        g1_acc = 0.0
        g2_acc = 0.0
        n_acc = 0
        max_rho_batch = 0.0

        # MCMC E-step across devices
        for slot in per_dev:
            device = slot["device"]
            X, chi_vec, w = slot["X"], slot["chi_vec"], slot["w"]
            w, J, Sigma = mcmc_sgld(w, X, chi_vec, m, chi, kappa, mdl, saem, mcmc)
            slot["w"] = w  # update state

            # compute expectations on this device
            A = mdl.N**mdl.gamma / mdl.sigma_a
            D = A * (kappa**2) + Sigma - (chi / mdl.N) * (J * J)
            D = torch.clamp(D, min=saem.eps_D)

            g1 = (J * J / D).mean()
            g2 = ((J * J) * (J * J) / ( (kappa**2) * (D * D) )).mean()
            g1_acc += g1.item() * w.shape[0]
            g2_acc += g2.item() * w.shape[0]
            n_acc += w.shape[0]

            rho = (J * J / torch.clamp(Sigma, min=1e-12)).max().item()
            if rho > max_rho_batch:
                max_rho_batch = rho

        g1_hat = g1_acc / max(1, n_acc)
        g2_hat = g2_acc / max(1, n_acc)
        rho_max_obs = max(rho_max_obs, max_rho_batch)

        # Robbins–Monro step
        a_t = saem.a0 / (it + saem.t0)
        F1 = m - mdl.N * (1.0 - m) * g1_hat
        F2 = chi - mdl.N * (1.0 - m) * (1.0 - m) * g2_hat - mdl.N * g1_hat
        m_new = m - saem.damping * a_t * F1
        chi_new = chi - saem.damping * a_t * F2

        # Project to feasible set
        m, chi, chi_bound = feasible_project(m_new, chi_new, mdl, rho_max_obs, saem)

        # Polyak–Ruppert averaging
        m_bar = ( (it-1)*m_bar + m ) / it
        chi_bar = ( (it-1)*chi_bar + chi ) / it

        if it % saem.print_every == 0 or it == 1:
            dt = time.time() - t_start
            msg = {
                "iter": it, "kappa": kappa, "m": m, "chi": chi,
                "m_bar": m_bar, "chi_bar": chi_bar,
                "g1_hat": g1_hat, "g2_hat": g2_hat,
                "rho_max_obs": rho_max_obs, "chi_bound": chi_bound,
                "time_s": round(dt, 2)
            }
            print(json.dumps(msg))
            history.append(msg)

    # Final snapshot
    result = {
        "kappa": kappa,
        "m_final": m, "chi_final": chi,
        "m_bar": m_bar, "chi_bar": chi_bar,
        "rho_max_obs": rho_max_obs,
        "history": history[-10:],  # keep last few for the JSON size
        "BASE": BASE
    }

    # Save
    os.makedirs(SAVE_DIR, exist_ok=True)
    tag = run_tag if run_tag else f"kappa_{kappa:.6f}"
    out_path = os.path.join(SAVE_DIR, f"saem_result_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[saved] {out_path}")
    return result


# ----------------------------- Main -----------------------------

if __name__ == "__main__":
    # Example configuration; edit here (no argparse).
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
        # SAEM controls
        a0=0.5, t0=100.0, damping=1.0,
        eps_D=1e-6, eps_proj=1e-3, print_every=50,
        # Seeds
        teacher_seed=0, data_seed=0,
        # init
        m_init=0.8, chi_init=1e-6,
    )

    kappa_list = sorted([0.001, 0.0075, 0.005, 0.0025, 0.001, 1e-4], reverse=True)

    save_dir = "/home/goring/mean_field_langevin/MCMC_Pinf_1N/results/0708_d30k4_diagosntic4"
    os.makedirs(save_dir, exist_ok=True)

    run_tag_prefix = time.strftime("%Y%m%d_%H%M%S")

    # Warm start across kappas
    m_ws, chi_ws = BASE.get("m_init", 0.0), BASE.get("chi_init", 1e-6)

    for idx, kappa in enumerate(kappa_list):
        print(f"\n=== SAEM for kappa={kappa} ===")
        # Update inits for warm start
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
        # Warm-start next
        m_ws, chi_ws = result["m_final"], result["chi_final"]
