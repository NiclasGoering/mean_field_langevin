# cavity_functional_selfconsistency_gamma_sweep.py
import os, json, time, math, random
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

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
    sigma_a: float = 1.0   # amplitude prior VARIANCE (no N scaling): a ~ N(0, sigma_a)
    sigma_w: float = 1.0   # weight prior variance (per neuron)
    gamma: float = 1.0     # f(x) = N^{-gamma} * sum a_i phi(w_i^T x)
    act: str = "relu"

@dataclass
class MCMCParams:
    B: int = 8192
    steps: int = 200
    step_size: float = 5e-3
    step_decay: float = 0.999
    grad_clip: float = 1e8
    clamp_w: float = 20.0
    langevin_sqrt2: bool = True
    autocast: Optional[bool] = None  # None -> auto (enabled on CUDA)

@dataclass
class SolveParams:
    outer_steps: int = 500
    saem_a0: float = 1.0
    saem_t0: float = 20.0
    saem_damping: float = 1.0
    print_every: int = 10
    save_every: int = 100   # <-- only save trajectory every this many iterations

@dataclass
class CorrectionParams:
    # Master switch: turn the 1/N correction on/off
    use_1_over_N: bool = False
    # Inducing set (subset of training inputs) for low-rank, quenched susceptibility
    M_inducing: int = 128
    rank: int = 32                 # top eigenpairs to keep (<= M_inducing)
    refresh_every: int = 5         # rebuild susceptibility every this many outer its
    ridge: float = 1e-6            # small ridge for numerical stability
    clip_ratio: float = 1.0        # enforce 0 <= R(w) <= clip_ratio * Sigma(w)
    inducing_seed: int = 0         # deterministic inducing subset
    # Whether to stop gradient through the reaction term inside SGLD (recommended)
    stopgrad_reaction: bool = True

# Internal container for precomputed correction state
class CorrectionState:
    def __init__(self):
        self.U: Optional[torch.Tensor] = None          # (M,d)
        self.Q: Optional[torch.Tensor] = None          # (M,r) eigenvectors
        self.lam: Optional[torch.Tensor] = None        # (r,)   eigenvalues (>=0)
        self.M: int = 0
        self.r: int = 0
        self.last_built_iter: int = 0

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
    corr_params: Optional[CorrectionParams] = None,
    corr_state: Optional[CorrectionState] = None,
):
    device = w.device
    P = X.shape[0]
    N = float(mdl.N)
    gamma = float(mdl.gamma)

    gw2 = mdl.sigma_w / mdl.d
    Acoef = 1.0 / mdl.sigma_a
    N2g = N ** (2.0 * gamma)

    step = mcmc.step_size
    autocast_enabled = mcmc.autocast if (mcmc.autocast is not None) else w.is_cuda

    y_f = y.view(-1).to(torch.float32)
    m_f = f_mean.view(-1).to(torch.float32)

    use_corr = bool(corr_params and corr_params.use_1_over_N and corr_state and corr_state.Q is not None)

    def reaction_R_for_W(Wmat: torch.Tensor) -> torch.Tensor:
        if not use_corr:
            return torch.zeros(Wmat.shape[0], device=Wmat.device, dtype=torch.float32)
        U = corr_state.U
        Q = corr_state.Q
        lam = corr_state.lam
        W_use = Wmat.detach() if corr_params.stopgrad_reaction else Wmat
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            zU = U @ W_use.t().contiguous()         # (M,B)
            phiU = activation(zU, mdl.act)          # (M,B)
        V = phiU.t().to(torch.float32)              # (B,M)
        t = V @ Q                                   # (B,r)
        R = (t * t) @ lam                           # (B,)
        return R

    for _ in range(mcmc.steps):
        w = w.detach().requires_grad_(True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            z = X @ w.t().contiguous()         # (P,B)
            phi = activation(z, mdl.act)

        g = phi.to(torch.float32)
        Sigma = (g * g).mean(dim=0)            # (B,)
        JY    = (g.t() @ y_f) / float(P)       # (B,)
        Jm    = (g.t() @ m_f) / float(P)       # (B,)
        Jr    = JY - Jm                        # (B,)

        if use_corr:
            R = reaction_R_for_W(w)            # (B,)
            max_R = (Sigma * float(corr_params.clip_ratio)).to(R.dtype)
            R = R.clamp_min(0.0)
            R = torch.minimum(R, max_R)
            Sigma_eff = Sigma - (R / float(mdl.N))
        else:
            Sigma_eff = Sigma

        D_internal = (Acoef + Sigma_eff / (kappa * kappa * N2g)).clamp_min(1e-12)

        prior   = 0.5 * (w * w).sum(dim=1) / gw2
        log_det = 0.5 * torch.log(D_internal)
        data_quad = (-0.5) * (Jr * Jr) / ((kappa ** 4) * N2g * D_internal + 1e-30)
        U = prior + log_det + data_quad

        grad = torch.autograd.grad(U.sum(), w, retain_graph=False, create_graph=False)[0]
        grad = torch.where(torch.isfinite(grad), grad, torch.zeros_like(grad))
        if mcmc.grad_clip and mcmc.grad_clip > 0:
            gn = grad.norm(dim=1, keepdim=True).clamp_min(1e-12)
            grad = grad * (mcmc.grad_clip / gn).clamp(max=1.0)

        noise = torch.randn_like(w)
        if mcmc.langevin_sqrt2:
            w = w - step * grad + noise * math.sqrt(2.0 * step)
        else:
            w = w - 0.5 * step * grad + noise * math.sqrt(step)
        if mcmc.clamp_w:
            w = torch.clamp(w, -mcmc.clamp_w, mcmc.clamp_w)
        step *= mcmc.step_decay

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        z = X @ w.t().contiguous()
        phi = activation(z, mdl.act).detach()
    g = phi.to(torch.float32)
    Sigma = (g * g).mean(dim=0)
    JY    = (g.t() @ y_f) / float(P)
    Jm    = (g.t() @ m_f) / float(P)

    if use_corr:
        R_final = reaction_R_for_W(w)
        max_Rf = (Sigma * float(corr_params.clip_ratio)).to(R_final.dtype)
        R_final = R_final.clamp_min(0.0)
        R_final = torch.minimum(R_final, max_Rf)
        Sigma_eff_final = Sigma - (R_final / float(mdl.N))
        D_oracle = ((kappa ** 2) * N2g) * Acoef + Sigma_eff_final
    else:
        D_oracle = ((kappa ** 2) * N2g) * Acoef + Sigma

    return w.detach(), phi.detach(), D_oracle.detach(), JY.detach(), Jm.detach(), Sigma.detach()


# ----------------------------- Functional self-consistency loop -----------------------------
class FunctionalCavitySolver:
    def __init__(self, mdl: ModelParams, mcmc: MCMCParams, sol: SolveParams,
                 kappa: float, device: torch.device,
                 corr: Optional[CorrectionParams] = None):
        self.mdl = mdl
        self.mcmc = mcmc
        self.sol = sol
        self.kappa = kappa
        self.device = device
        self.corr = corr if corr is not None else CorrectionParams()
        self._corr_state = CorrectionState()

    @torch.no_grad()
    def predict_mean_f_on(self, X_eval: torch.Tensor, W: torch.Tensor, c: torch.Tensor,
                          chunk_B: int = 1024) -> torch.Tensor:
        """
        Compute ⟨f(x)⟩ ≈ mean_b [ c_b * φ(w_b^T x) ].
        No extra N factor: c already includes the N^{-gamma} effect.
        """
        P_eval = X_eval.shape[0]
        f = torch.zeros(P_eval, device=X_eval.device, dtype=torch.float32)
        B = W.shape[0]
        for start in range(0, B, chunk_B):
            stop = min(start + chunk_B, B)
            Wb = W[start:stop]                 # (Bb, d)
            cb = c[start:stop].view(1, -1)     # (1, Bb)
            z = X_eval @ Wb.t().contiguous()   # (P_eval, Bb)
            gb = activation(z, self.mdl.act).to(torch.float32)
            # accumulate average over B in a numerically stable way
            f += (gb * cb).mean(dim=1) * (stop - start) / B
        return f

    @torch.no_grad()
    def _maybe_init_inducing(self, X_train: torch.Tensor):
        if self._corr_state.U is not None:
            return
        M = min(self.corr.M_inducing, X_train.shape[0])
        # Deterministic subset for "quenched" inducing points
        g = torch.Generator(device=X_train.device).manual_seed(self.corr.inducing_seed)
        idx = torch.randperm(X_train.shape[0], generator=g, device=X_train.device)[:M]
        self._corr_state.U = X_train[idx].to(self.device)  # (M,d)
        self._corr_state.M = M

    @torch.no_grad()
    def _refresh_correction(self, W: torch.Tensor, Sigma: torch.Tensor, c: torch.Tensor):
        """
        Build low-rank susceptibility on inducing set:
        chi ≈ (N^{2γ}/κ^2) * Cov_b[ (a/N^γ) φ_u ]  with
        E[(a/N^γ) φ_u] ≈ mean_b c_b φ_b(u), and Var(a/N^γ) = 1 / ( N^{2γ}A + Σ/κ^2 ).
        Then eigendecompose and keep top-`rank` modes.
        """
        if not self.corr.use_1_over_N:
            self._corr_state.Q = None
            self._corr_state.lam = None
            return

        U = self._corr_state.U  # (M,d)
        if U is None:
            return

        B = W.shape[0]
        M = U.shape[0]
        N2g = float(self.mdl.N ** (2.0 * self.mdl.gamma))
        Acoef = 1.0 / self.mdl.sigma_a
        kappa2 = float(self.kappa ** 2)

        # Posterior variance of a/N^γ given w (no reaction inside var; O(1/N) difference)
        var_anorm = 1.0 / (N2g * Acoef + (Sigma / kappa2))  # (B,)

        # φ(w^T U) for all chains
        zU = U @ W.t().contiguous()               # (M,B)
        phiU = activation(zU, self.mdl.act).to(torch.float32).t()  # (B,M)

        # First and second moments across chains
        # S1_j = E_b[ c_b * v_bj ]
        S1 = (phiU * c.view(B, 1)).mean(dim=0)    # (M,)
        # S2_{jk} = E_b[ (c_b^2 + var_b) * v_bj * v_bk ]
        q = (c * c + var_anorm).view(B, 1)        # (B,1)
        S2 = (phiU.t() @ (phiU * q)) / float(B)   # (M,M)

        Cov = S2 - S1.view(M, 1) @ S1.view(1, M)  # (M,M)
        # Scale to susceptibility tilde{chi}
        chi = (N2g / kappa2) * Cov
        # Numerical stabilization
        chi = 0.5 * (chi + chi.t())
        if self.corr.ridge > 0:
            chi = chi + float(self.corr.ridge) * torch.eye(M, device=chi.device, dtype=chi.dtype)

        # Eigendecomposition and truncation
        # eigh returns ascending eigenvalues
        evals, evecs = torch.linalg.eigh(chi)     # (M,), (M,M)
        # Keep top-r nonnegative modes
        r = min(self.corr.rank, M)
        top = evals[-r:]
        Q = evecs[:, -r:]
        # Clamp negatives to 0 (PSD guarantee)
        lam = torch.clamp(top, min=0.0).to(torch.float32)
        Q = Q.to(torch.float32)

        self._corr_state.Q = Q
        self._corr_state.lam = lam
        self._corr_state.r = r
        self._corr_state.last_built_iter = 0  # updated in run()

    def run(self, X_train: torch.Tensor, y_train: torch.Tensor,
            X_test: torch.Tensor, y_test: torch.Tensor,
            outer_steps: Optional[int] = None,
            log_dir: str = "./results_cavity_gamma", run_tag: str = "") -> Dict:

        os.makedirs(log_dir, exist_ok=True)
        if outer_steps is None: outer_steps = self.sol.outer_steps

        P_train, d = X_train.shape

        # init mean function on train set
        f_mean = torch.zeros(P_train, device=self.device, dtype=torch.float32)

        # init SGLD chains for w
        W = torch.randn(self.mcmc.B, d, device=self.device) * math.sqrt(self.mdl.sigma_w / self.mdl.d)

        traj = {
            "iter": [], "time_s": [],
            "train_mse": [], "train_corr": [],
            "test_mS": [], "test_noise_norm2": [], "test_err01_direct": [], "test_err01_clt": [],
            "test_mse": [],
            "mean_Sigma": [], "mean_abs_Jr": [], "kappa2N2A": [], "mean_c": [],
            "use_1_over_N": [], "M_inducing": [], "rank": []
        }
        t0 = time.time()

        Acoef = 1.0 / self.mdl.sigma_a
        kappa2N2A = (self.kappa ** 2) * (self.mdl.N ** (2.0 * self.mdl.gamma)) * Acoef

        # Prepare inducing set if needed
        if self.corr.use_1_over_N:
            self._maybe_init_inducing(X_train)

        initial_metrics_recorded = False
        initial_metrics = {}
        final_metrics = {}

        for it in range(1, outer_steps + 1):
            # --- E-step: sample w from single-neuron posterior (given current f_mean) ---
            W, phi_train, D_or, JY, Jm, Sigma = sgld_sample_w(
                W, X_train, y_train, f_mean, self.kappa, self.mdl, self.mcmc,
                corr_params=self.corr,
                corr_state=self._corr_state
            )
            Jr = (JY - Jm).to(torch.float32)

            # c_b = Jr / D_oracle  (already includes N^{-gamma} normalization consistently)
            c = (Jr / D_or).to(torch.float32)           # (B,)

            # --- Build new mean function on training set ---
            f_new_train = (phi_train.to(torch.float32) * c.view(1, -1)).mean(dim=1)

            # SAEM-style damping for stability
            a_t = self.sol.saem_a0 / (it + self.sol.saem_t0)
            f_mean = (1 - self.sol.saem_damping * a_t) * f_mean + self.sol.saem_damping * a_t * f_new_train

            # training diagnostics
            train_mse = float(((y_train - f_mean) ** 2).mean().item())
            train_corr = float((y_train * f_mean).mean().item())  # ⟨y f⟩ on train

            # --- (Optional) refresh low-rank susceptibility on inducing set ---
            if self.corr.use_1_over_N and ((it == 1) or (it % max(1, self.corr.refresh_every) == 0)):
                self._refresh_correction(W, Sigma, c)
                self._corr_state.last_built_iter = it

            # --- Unbiased evaluation on held-out TEST set (global) ---
            with torch.no_grad():
                f_test = self.predict_mean_f_on(X_test, W, c, chunk_B=1024*16)
                m_S = float((f_test * y_test).mean().item())      # since ⟨y^2⟩=1 for ±1 labels
                r = f_test - m_S * y_test
                noise_norm2 = float((r * r).mean().item())
                err01_direct = float(((f_test * y_test) <= 0).float().mean().item())
                sigma = max(math.sqrt(noise_norm2), 1e-12)
                z = -m_S / sigma
                if z > 12.0:
                    err01_clt = 1.0
                elif z < -12.0:
                    err01_clt = 0.0
                else:
                    err01_clt = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
                test_mse = float(((y_test - f_test) ** 2).mean().item())

            # Record initial metrics at the first iteration snapshot
            if not initial_metrics_recorded:
                initial_metrics = {
                    "iter": it,
                    "train_mse": train_mse,
                    "train_corr": train_corr,
                    "test_mS": m_S,
                    "test_noise_norm2": noise_norm2,
                    "test_err01_direct": err01_direct,
                    "test_err01_clt": err01_clt,
                    "test_mse": test_mse,
                    "mean_Sigma": float(Sigma.mean().item()),
                    "mean_abs_Jr": float(Jr.abs().mean().item()),
                    "kappa2N2A": float(kappa2N2A),
                    "mean_c": float(c.mean().item()),
                    "use_1_over_N": bool(self.corr.use_1_over_N),
                    "M_inducing": int(self._corr_state.M if self._corr_state.U is not None else 0),
                    "rank": int(self._corr_state.r if self._corr_state.Q is not None else 0),
                    "elapsed_s": round(time.time() - t0, 2)
                }
                initial_metrics_recorded = True

            # Save trajectory sparsely
            if (it % max(1, self.sol.save_every) == 0) or (it == 1) or (it == outer_steps):
                traj["iter"].append(it)
                traj["time_s"].append(time.time() - t0)
                traj["train_mse"].append(train_mse)
                traj["train_corr"].append(train_corr)
                traj["test_mS"].append(m_S)
                traj["test_noise_norm2"].append(noise_norm2)
                traj["test_err01_direct"].append(err01_direct)
                traj["test_err01_clt"].append(err01_clt)
                traj["test_mse"].append(test_mse)
                traj["mean_Sigma"].append(float(Sigma.mean().item()))
                traj["mean_abs_Jr"].append(float(Jr.abs().mean().item()))
                traj["kappa2N2A"].append(float(kappa2N2A))
                traj["mean_c"].append(float(c.mean().item()))
                traj["use_1_over_N"].append(bool(self.corr.use_1_over_N))
                traj["M_inducing"].append(int(self._corr_state.M if self._corr_state.U is not None else 0))
                traj["rank"].append(int(self._corr_state.r if self._corr_state.Q is not None else 0))

            # Printing
            if it % self.sol.print_every == 1 or it == outer_steps:
                print(json.dumps({
                    "iter": it,
                    "train_mse": train_mse,
                    "train_corr": train_corr,
                    "test_mS": m_S,
                    "test_noise_norm2": noise_norm2,
                    "test_err01_direct": err01_direct,
                    "test_err01_clt": err01_clt,
                    "test_mse": test_mse,
                    "mean_Sigma": float(Sigma.mean().item()),
                    "mean_abs_Jr": float(Jr.abs().mean().item()),
                    "kappa2N2A": float(kappa2N2A),
                    "mean_c": float(c.mean().item()),
                    "elapsed_s": round(time.time() - t0, 2),
                    "B": self.mcmc.B, "P_train": P_train, "P_test": X_test.shape[0],
                    "use_1_over_N": bool(self.corr.use_1_over_N),
                    "M_inducing": int(self._corr_state.M if self._corr_state.U is not None else 0),
                    "rank": int(self._corr_state.r if self._corr_state.Q is not None else 0)
                }))

            # Always update final_metrics (overwrites until last iter)
            final_metrics = {
                "iter": it,
                "train_mse": train_mse,
                "train_corr": train_corr,
                "test_mS": m_S,
                "test_noise_norm2": noise_norm2,
                "test_err01_direct": err01_direct,
                "test_err01_clt": err01_clt,
                "test_mse": test_mse,
                "mean_Sigma": float(Sigma.mean().item()),
                "mean_abs_Jr": float(Jr.abs().mean().item()),
                "kappa2N2A": float(kappa2N2A),
                "mean_c": float(c.mean().item()),
                "use_1_over_N": bool(self.corr.use_1_over_N),
                "M_inducing": int(self._corr_state.M if self._corr_state.U is not None else 0),
                "rank": int(self._corr_state.r if self._corr_state.Q is not None else 0),
                "elapsed_s": round(time.time() - t0, 2)
            }

        # final snapshot / summary
        out = {
            "summary": {
                "P_train": int(P_train),
                "P_test": int(X_test.shape[0]),
                "d": self.mdl.d,
                "k": self.mdl.k,
                "N": self.mdl.N,
                "gamma": self.mdl.gamma,
                "kappa": self.kappa,
                "act": self.mdl.act,
                "sigma_a": self.mdl.sigma_a,
                "use_1_over_N": bool(self.corr.use_1_over_N),
                "M_inducing": int(self._corr_state.M if self._corr_state.U is not None else 0),
                "rank": int(self._corr_state.r if self._corr_state.Q is not None else 0),
                "initial_metrics": initial_metrics,
                "final_metrics": final_metrics
            },
            "traj": traj,
            "config": {
                "mdl": vars(self.mdl),
                "mcmc": vars(self.mcmc),
                "sol": vars(self.sol),
                "kappa": self.kappa,
                "corr": vars(self.corr)
            }
        }
        # smart file name including key sweep params
        tag = run_tag or time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(
            log_dir,
            f"cavity_func_gamma_P{P_train}_N{self.mdl.N}_kap{float(self.kappa):.3e}_gamma{self.mdl.gamma}_{tag}.json"
        )
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

# ----------------------------- Entrypoint with N/kappa/P sweeps -----------------------------
if __name__ == "__main__":
    set_seed(42)
    _ = check_gpu()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---- Teacher/data ----
    d = 15
    k = 4

    log_dir = "/home/goring/mean_field_langevin/MCMC_finiteP/resutls_mf1/d15_k4_1"

     
    # These lists make N, kappa, and P_train iterable. Adjust as you like.
    N_list: List[int] = [1024]            # e.g., [256, 512, 1024]
    kappa_list: List[float] = [1e-3] #[1e-2,7.5e-3,1e-1,1e-3,5e-3,5e-2,5e-4]     # e.g., [3e-4, 1e-3, 3e-3]
    P_train_list: List[int] = [10,100, 500,1000,2500,5000,10000,20000]    # e.g., [10000, 25000, 50000]

    P_test = 20000  # global test-set size (formerly P_eval)
    gamma = 1.0     # set to 1.0 for 1/N, 0.5 for 1/sqrt(N), 0.0 for no N scaling

    S_idx = make_parity_indices(d, k, seed=0).to(device)

    # Global TEST set (fixed across all runs for comparability)
    X_test, y_test = generate_dense_parity_data(P_test, d, k, S_idx, device)

    # ---- MCMC/Solver knobs ----
    # You can tweak these once; they apply to all runs in the sweep
    base_mcmc = MCMCParams(
        B=16384, steps=200, step_size=5e-3, step_decay=0.999,
        grad_clip=1e8, clamp_w=20.0, autocast=True
    )
    base_sol  = SolveParams(
        outer_steps=1000, saem_a0=1.0, saem_t0=20.0, saem_damping=1.0,
        print_every=50, save_every=100  # <-- trajectory saved every 100 iterations
    )

    # ---- 1/N correction controls ----
    corr = CorrectionParams(
        use_1_over_N=False,        # set False to disable and recover original behavior
        M_inducing=512,
        rank=128,
        refresh_every=5,
        ridge=1e-6,
        clip_ratio=1.0,
        inducing_seed=0,
        stopgrad_reaction=False
    )


    # ---------------------- Sweep over (P_train, N, kappa) ----------------------
    for P_train in P_train_list:
        # Training data regenerated per P_train
        X_train, y_train = generate_dense_parity_data(P_train, d, k, S_idx, device)

        for N in N_list:
            for kappa in kappa_list:
                mdl  = ModelParams(d=d, N=N, k=k, sigma_a=1.0, sigma_w=1.0, gamma=gamma, act="relu")
                # fresh copies so runs are independent
                mcmc = MCMCParams(**vars(base_mcmc))
                sol  = SolveParams(**vars(base_sol))

                run_tag = f"P{P_train}_N{N}_kap{float(kappa):.3e}"

                solver = FunctionalCavitySolver(mdl, mcmc, sol, kappa=kappa, device=device, corr=corr)
                _ = solver.run(X_train, y_train, X_test, y_test,
                               log_dir=log_dir, run_tag=run_tag)
