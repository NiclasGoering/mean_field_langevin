# cavity_functional_selfconsistency.py
import os, json, time, math, random
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import torch._dynamo as dynamo

# ----------------------------- Utilities -----------------------------
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
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def activation(z: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "relu": return torch.relu(z)
    if kind == "tanh": return torch.tanh(z)
    raise ValueError(f"Unknown activation: {kind}")

def make_parity_indices(d: int, k: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    S = torch.randperm(d, generator=g)[:k]
    return S.sort().values  # indices used by teacher

def parity_labels(X_pm1: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    # X in {±1}, labels ±1 from product over subset S
    feats = X_pm1[:, S]          # (P, k)
    y = feats.prod(dim=1)        # (P,)
    return y.to(torch.float32)

# ----------------------------- Configs -----------------------------
@dataclass
class ModelParams:
    d: int = 25
    N: int = 1024
    k: int = 4
    sigma_a: float = 1.0
    sigma_w: float = 1.0
    gamma: float = 1.0
    act: str = "relu"

@dataclass
class MCMCParams:
    B: int = 8192
    steps: int = 50
    step_size: float = 2e-4
    step_decay: float = 0.999
    grad_clip: float = 1e15
    clamp_w: float = 10.0
    langevin_sqrt2: bool = True
    autocast: Optional[bool] = None  # None -> auto

@dataclass
class SolveParams:
    outer_steps: int = 300
    saem_a0: float = 1.0
    saem_t0: float = 150.0
    saem_damping: float = 3.0
    print_every: int = 10

@dataclass
class DataParams:
    P_train: int = 2000
    P_eval: int = 100000  # large held-out for unbiased metrics
    seed_data: int = 0

# ----------------------------- SGLD over w (integrate out a) -----------------------------
@dynamo.disable
def sgld_sample_w(
    w: torch.Tensor,        # (B,d)
    X: torch.Tensor,        # (P,d), entries ±1
    y: torch.Tensor,        # (P,), ±1
    f_mean: torch.Tensor,   # (P,), current mean function on train set
    kappa: float,
    mdl: ModelParams,
    mcmc: MCMCParams,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    SGLD for the single-neuron posterior with 'a' analytically integrated.
    Returns:
      w_new: (B,d)
      phi: (P,B) activations on training set at final state
      D_oracle: (B,) = κ²*A + Σ(w)
      JY: (B,)
      Jm: (B,)
      Sigma: (B,)
    """
    device = w.device
    P = X.shape[0]

    # Prior scales
    gw2   = mdl.sigma_w / mdl.d                 # variance per weight coord
    Acoef = (mdl.N ** mdl.gamma) / mdl.sigma_a  # from Gaussian prior on a

    step = mcmc.step_size
    autocast_enabled = mcmc.autocast if (mcmc.autocast is not None) else w.is_cuda

    # pre-cast once
    y_f  = y.view(-1).to(torch.float32)
    m_f  = f_mean.view(-1).to(torch.float32)

    for _ in range(mcmc.steps):
        w = w.detach().requires_grad_(True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            z = X @ w.t().contiguous()         # (P,B)
            phi = activation(z, mdl.act)

        g = phi.to(torch.float32)
        Sigma = (g*g).mean(dim=0)              # (B,)
        JY = (g.t() @ y_f) / float(P)          # (B,)
        Jm = (g.t() @ m_f) / float(P)          # (B,)
        D_internal = (Acoef + Sigma / (kappa**2)).clamp_min(1e-9)  # α(w)

        # Energy U(w) (a integrated out)
        prior = 0.5 * (w*w).sum(dim=1) / gw2
        log_det = 0.5 * torch.log(D_internal)
        data_quad = (-0.5) * (JY - Jm).pow(2) / ((kappa**4) * D_internal)
        U = prior + log_det + data_quad

        grad = torch.autograd.grad(U.sum(), w, retain_graph=False, create_graph=False)[0]
        grad = torch.where(torch.isfinite(grad), grad, torch.zeros_like(grad))
        if mcmc.grad_clip and mcmc.grad_clip > 0:
            gn = grad.norm(dim=1, keepdim=True).clamp_min(1e-12)
            grad = grad * (mcmc.grad_clip/gn).clamp(max=1.0)

        noise = torch.randn_like(w)
        if mcmc.langevin_sqrt2:
            w = w - step * grad + noise * math.sqrt(2.0*step)
        else:
            w = w - 0.5*step*grad + noise * math.sqrt(step)
        if mcmc.clamp_w:
            w = torch.clamp(w, -mcmc.clamp_w, mcmc.clamp_w)
        step *= mcmc.step_decay

    # final recompute for outputs
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        z = X @ w.t().contiguous()
        phi = activation(z, mdl.act).detach()
    g = phi.to(torch.float32)
    Sigma = (g*g).mean(dim=0)
    JY = (g.t() @ y_f) / float(P)
    Jm = (g.t() @ m_f) / float(P)
    D_oracle = (Acoef * (kappa**2) + Sigma).clamp_min(1e-9)   # κ²*A + Σ

    return w.detach(), phi.detach(), D_oracle.detach(), JY.detach(), Jm.detach(), Sigma.detach()

# ----------------------------- Functional self-consistency loop -----------------------------
class FunctionalCavitySolver:
    def __init__(self, mdl: ModelParams, mcmc: MCMCParams, sol: SolveParams,
                 kappa: float, device: torch.device):
        self.mdl = mdl
        self.mcmc = mcmc
        self.sol = sol
        self.kappa = kappa
        self.device = device

    @torch.no_grad()
    def predict_mean_f_on(self, X_eval: torch.Tensor, W: torch.Tensor, c: torch.Tensor,
                          chunk_B: int = 1024) -> torch.Tensor:
        """
        Compute ⟨f(x)⟩ ≈ N * mean_b [ c_b * φ(w_b^T x) ] on an arbitrary set X_eval.
        Chunked over B to limit memory.
        """
        P_eval = X_eval.shape[0]
        B = W.shape[0]
        f = torch.zeros(P_eval, device=X_eval.device, dtype=torch.float32)
        for start in range(0, B, chunk_B):
            stop = min(start + chunk_B, B)
            Wb = W[start:stop]                 # (Bb, d)
            cb = c[start:stop].view(1, -1)     # (1, Bb)
            z = X_eval @ Wb.t().contiguous()   # (P_eval, Bb)
            gb = activation(z, self.mdl.act).to(torch.float32)  # (P_eval, Bb)
            f += (gb * cb).mean(dim=1) * (stop - start) / B     # weighted average
        f = f * float(self.mdl.N)
        return f

    def run(self, X_train: torch.Tensor, y_train: torch.Tensor,
            X_eval: torch.Tensor, y_eval: torch.Tensor,
            outer_steps: Optional[int] = None,
            log_dir: str = "./results_cavity", run_tag: str = "") -> Dict:

        os.makedirs(log_dir, exist_ok=True)
        if outer_steps is None: outer_steps = self.sol.outer_steps

        P_train, d = X_train.shape
        device = self.device

        # init mean function on train set
        f_mean = torch.zeros(P_train, device=device, dtype=torch.float32)

        # init SGLD chains for w
        W = torch.randn(self.mcmc.B, d, device=device) * math.sqrt(self.mdl.sigma_w/self.mdl.d)

        traj = {
            "iter": [], "time_s": [],
            "train_mse": [], "train_corr": [],
            "eval_mS": [], "eval_noise_norm2": [], "eval_err01_direct": [], "eval_err01_clt": []
        }
        t0 = time.time()

        for it in range(1, outer_steps+1):
            # --- E-step: sample w from single-neuron posterior (given current f_mean) ---
            W, phi_train, D_or, JY, Jm, Sigma = sgld_sample_w(
                W, X_train, y_train, f_mean, self.kappa, self.mdl, self.mcmc
            )

            # compute coefficients c_b = (JY - Jm)/D_oracle
            c = (JY - Jm) / D_or            # (B,)

            # --- Build new mean function on training set ---
            # f_new_train = N * mean_b [ c_b * φ_b(x) ]
            f_new_train = (phi_train.to(torch.float32) * c.view(1, -1)).mean(dim=1) * float(self.mdl.N)

            # SAEM-style damping for stability
            a_t = self.sol.saem_a0 / (it + self.sol.saem_t0)
            f_mean = (1 - self.sol.saem_damping*a_t) * f_mean + self.sol.saem_damping*a_t * f_new_train

            # training diagnostics
            train_mse = float(((y_train - f_mean)**2).mean().item())
            train_corr = float((y_train * f_mean).mean().item())  # ⟨y f⟩ on train

            # --- Unbiased evaluation on large held-out set ---
            # Use the *same* posterior samples and coefficients (they depend on train residuals),
            # and evaluate ⟨f⟩ on X_eval.
            with torch.no_grad():
                f_eval = self.predict_mean_f_on(X_eval, W, c, chunk_B=1024)
                # alignment and noise (orthogonal residual to teacher)
                m_S = float((f_eval * y_eval).mean().item())           # since ⟨y^2⟩=1 for ±1 labels
                r = f_eval - m_S * y_eval
                noise_norm2 = float((r*r).mean().item())
                # 0-1 direct
                err01_direct = float(((f_eval * y_eval) <= 0).float().mean().item())
                # CLT-style estimate from m_S and noise variance
                sigma = max(math.sqrt(noise_norm2), 1e-12)
                z = -m_S / sigma
                if z > 12.0:  # numeric guard
                    err01_clt = 1.0
                elif z < -12.0:
                    err01_clt = 0.0
                else:
                    err01_clt = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

            traj["iter"].append(it)
            traj["time_s"].append(time.time() - t0)
            traj["train_mse"].append(train_mse)
            traj["train_corr"].append(train_corr)
            traj["eval_mS"].append(m_S)
            traj["eval_noise_norm2"].append(noise_norm2)
            traj["eval_err01_direct"].append(err01_direct)
            traj["eval_err01_clt"].append(err01_clt)

            if it % self.sol.print_every == 1 or it == outer_steps:
                print(json.dumps({
                    "iter": it,
                    "train_mse": train_mse,
                    "train_corr": train_corr,
                    "eval_mS": m_S,
                    "eval_noise_norm2": noise_norm2,
                    "eval_err01_direct": err01_direct,
                    "eval_err01_clt": err01_clt,
                    "elapsed_s": round(traj["time_s"][-1], 2),
                    "B": self.mcmc.B, "P_train": P_train, "P_eval": X_eval.shape[0]
                }))

        # final snapshot
        out = {
            "summary": {
                "P_train": int(P_train),
                "P_eval": int(X_eval.shape[0]),
                "d": self.mdl.d,
                "k": self.mdl.k,
                "N": self.mdl.N,
                "kappa": self.kappa,
                "act": self.mdl.act,
                "mS_last": traj["eval_mS"][-1],
                "noise_norm2_last": traj["eval_noise_norm2"][-1],
                "err01_direct_last": traj["eval_err01_direct"][-1],
                "err01_clt_last": traj["eval_err01_clt"][-1],
            },
            "traj": traj,
            "config": {
                "mdl": vars(self.mdl),
                "mcmc": vars(self.mcmc),
                "sol": vars(self.sol),
                "kappa": self.kappa
            }
        }
        tag = run_tag or time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(log_dir, f"cavity_func_{tag}_P{P_train}_Neval{X_eval.shape[0]}_kap{float(self.kappa):.3e}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {path}")
        return out

# ----------------------------- Data generation -----------------------------
def generate_dense_parity_data(P: int, d: int, k: int, S_idx: torch.Tensor, device) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(0)
    X = (torch.randint(0, 2, (P, d), generator=g, device=device, dtype=torch.int8).float() * 2.0 - 1.0)
    y = parity_labels(X, S_idx)
    return X, y

# ----------------------------- Entrypoint -----------------------------
if __name__ == "__main__":
    set_seed(42)
    _ = check_gpu()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---- knobs ----
    d = 10
    k = 2
    P_train = 1000
    P_eval  = 100000   # large held-out for unbiased estimates
    N = 1024
    kappa = 1e-2

    mdl  = ModelParams(d=d, N=N, k=k, sigma_a=1.0, sigma_w=1.0, gamma=1.0, act="relu")
    mcmc = MCMCParams(B=8192*4, steps=400, step_size=2e-6, step_decay=0.999,
                      grad_clip=1e3, clamp_w=20.0, autocast=True)
    sol  = SolveParams(outer_steps=10000, saem_a0=1.0, saem_t0=150.0, saem_damping=200.2,
                       print_every=5)

    # teacher subset and data
    S_idx = make_parity_indices(d, k, seed=0).to(device)
    X_train, y_train = generate_dense_parity_data(P_train, d, k, S_idx, device)
    X_eval,  y_eval  = generate_dense_parity_data(P_eval,  d, k, S_idx, device)

    solver = FunctionalCavitySolver(mdl, mcmc, sol, kappa=kappa, device=device)
    _ = solver.run(X_train, y_train, X_eval, y_eval,
                   log_dir="./results_cavity", run_tag="")
