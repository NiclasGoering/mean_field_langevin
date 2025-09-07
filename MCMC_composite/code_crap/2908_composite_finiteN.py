# cavity_functional_selfconsistency_gamma_composite.py
# Basis-free 1/N correction (Option B, low-rank probe chains) with stability fixes:
#   • Connected susceptibility (variance-only) in reaction by default
#   • Center + whiten features → use correlation ρ^2 (bounded)
#   • Per-chain ratio clip: R_divN ≤ rho_clip * Σ
#   • Strong damping (EMA) + annealed λ_react
#   • Floor clamp for Σ_eff ≥ clamp_sigma_eff_min
#
# Prints detailed 1/N diagnostics.

import os, json, time, math, random, re
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

def parse_composite_spec(spec: str) -> List[List[int]]:
    sets = re.findall(r"\{([^}]*)\}", spec)
    out: List[List[int]] = []
    for s in sets:
        if s.strip() == "":
            out.append([])
            continue
        elems = [int(tok.strip()) for tok in s.split(",") if tok.strip() != ""]
        out.append(sorted(elems))
    if len(out) == 0:
        raise ValueError(f"Failed to parse composite spec: {spec!r}")
    return out

def parity_character(X_pm1: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    if S.numel() == 0:
        return torch.ones(X_pm1.shape[0], device=X_pm1.device, dtype=torch.float32)
    feats = X_pm1[:, S]
    return feats.prod(dim=1).to(torch.float32)

def compute_characters_matrix(X_pm1: torch.Tensor, sets: List[torch.Tensor]) -> torch.Tensor:
    P = X_pm1.shape[0]
    M = len(sets)
    C = torch.empty(P, M, device=X_pm1.device, dtype=torch.float32) if M > 0 \
        else torch.zeros(P, 0, device=X_pm1.device, dtype=torch.float32)
    for j, Sj in enumerate(sets):
        C[:, j] = parity_character(X_pm1, Sj)
    return C

# ----------------------------- Configs -----------------------------
@dataclass
class ModelParams:
    d: int = 25
    N: int = 1024
    k: int = 4
    sigma_a: float = 1.0            # std of a
    sigma_w: float = 1.0            # std of ||w|| scaled by sqrt(d)
    gamma: float = 1.0              # f(x) = N^{-γ} * sum_i a_i φ(w_i^T x)
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
    autocast: Optional[bool] = False

@dataclass
class SolveParams:
    outer_steps: int = 500
    saem_a0: float = 1.0
    saem_t0: float = 20.0
    saem_damping: float = 1.0
    print_every: int = 10

@dataclass
class ReactionParams:
    # Low-rank probe-chains reaction parameters
    r: int = 1024
    resample_every: int = 1
    # Connected χ (variance-only) vs full (μ^2+σ^2)
    a2_mode: str = "var"          # "var" (default) or "full"
    # Use whitened correlations instead of raw dot products
    use_corr: bool = True
    center_phi: bool = True
    # Damping + anneal
    lambda_react: float = 0.05     # small by default; set higher once stable
    ema: float = 0.95
    anneal_iters: int = 100        # ramp λ from 0 to target over these iters
    # Trust-region & stability
    rho_clip: float = 0.25         # cap R_divN <= rho_clip * Σ (per chain)
    clamp_sigma_eff_min: float = 1e-3

# ----------------------------- SGLD (lag-1 reaction) -----------------------------
@dynamo.disable
def sgld_sample_w(
    w: torch.Tensor,        # (B,d)
    X: torch.Tensor,        # (P,d)
    y: torch.Tensor,        # (P,)
    f_mean: torch.Tensor,   # (P,)
    kappa: float,
    mdl: ModelParams,
    mcmc: MCMCParams,
    R_divN: Optional[torch.Tensor] = None,  # (B,)
    clamp_sigma_eff_min: float = 1e-3,
):
    """
    α(w)  = 1/(2 σ_a^2) + Σ_eff / (κ^2 N^{2γ}), where Σ_eff = max(Σ - R_divN, clamp)
    D_or  = κ^2 N^{2γ} α = κ^2 N^{2γ} * (1/(2 σ_a^2)) + Σ_eff
    """
    device = w.device
    P = X.shape[0]
    N = float(mdl.N)
    N2g = N ** (2.0 * float(mdl.gamma))

    var_w_per_coord = (mdl.sigma_w ** 2) / mdl.d
    Acoef = 1.0 / (2.0 * (mdl.sigma_a ** 2))  # 1/(2 σ_a^2)

    step = mcmc.step_size
    autocast_enabled = mcmc.autocast if (mcmc.autocast is not None) else w.is_cuda

    y_f = y.view(-1).to(torch.float32)
    m_f = f_mean.view(-1).to(torch.float32)

    if R_divN is None:
        R_divN = torch.zeros(w.shape[0], device=device, dtype=torch.float32)
    else:
        R_divN = R_divN.to(torch.float32)

    for _ in range(mcmc.steps):
        w = w.detach().requires_grad_(True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            z = X @ w.t().contiguous()         # (P,B)
            phi = activation(z, mdl.act)
        g = phi.to(torch.float32)              # (P,B)

        Sigma = (g * g).mean(dim=0)            # (B,)
        Sigma_eff = Sigma - R_divN
        if clamp_sigma_eff_min is not None:
            Sigma_eff = torch.maximum(Sigma_eff, torch.tensor(clamp_sigma_eff_min, device=device, dtype=torch.float32))

        JY = (g.t() @ y_f) / float(P)
        Jm = (g.t() @ m_f) / float(P)
        Jr = JY - Jm

        D_internal = (Acoef + Sigma_eff / (kappa * kappa * N2g)).clamp_min(1e-12)  # α

        prior   = 0.5 * (w * w).sum(dim=1) / var_w_per_coord
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
    Sigma_eff = Sigma - R_divN
    if clamp_sigma_eff_min is not None:
        Sigma_eff = torch.maximum(Sigma_eff, torch.tensor(clamp_sigma_eff_min, device=device, dtype=torch.float32))
    JY    = (g.t() @ y_f) / float(P)
    Jm    = (g.t() @ m_f) / float(P)
    Jr    = JY - Jm

    D_oracle = ((kappa ** 2) * N2g * Acoef + Sigma_eff).clamp_min(1e-12)
    return (
        w.detach(),            # weights
        phi.detach(),          # φ on train (P,B)
        D_oracle.detach(),     # (B,)
        JY.detach(),
        Jm.detach(),
        Sigma.detach(),        # Σ (B,)
        Sigma_eff.detach(),    # Σ_eff (B,)
    )

# ----------------------------- Solver -----------------------------
class FunctionalCavitySolver:
    def __init__(self, mdl: ModelParams, mcmc: MCMCParams, sol: SolveParams,
                 kappa: float, device: torch.device,
                 teacher_sets: Optional[List[torch.Tensor]] = None,
                 react: Optional[ReactionParams] = None):
        self.mdl = mdl
        self.mcmc = mcmc
        self.sol = sol
        self.kappa = kappa
        self.device = device
        self.teacher_sets = teacher_sets or []
        self.react = react or ReactionParams()
        # lag-1 state
        self._R_divN_prev: Optional[torch.Tensor] = None
        self._J_indices: Optional[torch.Tensor] = None

    @torch.no_grad()
    def predict_mean_f_on(self, X_eval: torch.Tensor, W: torch.Tensor, c: torch.Tensor,
                          chunk_B: int = 1024) -> torch.Tensor:
        P_eval = X_eval.shape[0]
        f = torch.zeros(P_eval, device=X_eval.device, dtype=torch.float32)
        B = W.shape[0]
        for start in range(0, B, chunk_B):
            stop = min(start + chunk_B, B)
            Wb = W[start:stop]
            cb = c[start:stop].view(1, -1)
            z = X_eval @ Wb.t().contiguous()
            gb = activation(z, self.mdl.act).to(torch.float32)
            f += (gb * cb).mean(dim=1) * (stop - start) / B
        return self.mdl.N * f

    @torch.no_grad()
    def compute_susceptibility_AA(
        self,
        phi_train: torch.Tensor,     # (P, B)
        c: torch.Tensor,             # (B,)
        D_oracle: torch.Tensor,      # (B,)
        C_train: torch.Tensor,       # (P, M)
        chunk_B: int = 2048
    ) -> torch.Tensor:
        if C_train.numel() == 0:
            return torch.zeros(0, device=phi_train.device, dtype=torch.float32)

        P, B = phi_train.shape
        M = C_train.shape[1]
        device = phi_train.device

        N = float(self.mdl.N)
        gamma = float(self.mdl.gamma)
        N2g = N ** (2.0 * gamma)

        a2 = N2g * (c.to(torch.float32) ** 2 + (self.kappa ** 2) / D_oracle.to(torch.float32))

        accum = torch.zeros(M, device=device, dtype=torch.float32)
        Ct = C_train.to(torch.float32).t()

        for start in range(0, B, chunk_B):
            stop = min(start + chunk_B, B)
            g_chunk = phi_train[:, start:stop].to(torch.float32)
            J_chunk = (Ct @ g_chunk) / float(P)
            term = (a2[start:stop].view(1, -1) * (J_chunk * J_chunk))
            accum += term.sum(dim=1)

        chi_AA = (N / (self.kappa ** 2)) * (accum / float(B))
        return chi_AA

    @torch.no_grad()
    def _sample_J_indices(self, B: int) -> torch.Tensor:
        r = min(self.react.r, B)
        g = torch.Generator(device=self.device).manual_seed(torch.randint(0, 2**31-1, (1,), device=self.device).item())
        perm = torch.randperm(B, generator=g, device=self.device)
        return perm[:r]

    @torch.no_grad()
    def compute_reaction_lowrank(
        self,
        phi_train: torch.Tensor,  # (P,B)
        Sigma_raw: torch.Tensor,  # (B,) uncentered Σ = E[φ^2]
        c: torch.Tensor,          # (B,)
        D_oracle: torch.Tensor,   # (B,)
        J_indices: torch.Tensor,  # (r,)
        outer_iter: int           # for annealing
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        R_divN[b] ≈ λ_eff * (1/κ^2) * (1/r) * Σ_{j∈J} a2_react[j] * ρ_{bj}^2,
        with ρ computed from centered+whitened features and
        a2_react = N^{2γ} * (κ^2 / D_oracle)   if a2_mode="var"
                 = N^{2γ} * (c^2 + κ^2 / D_oracle) if a2_mode="full"
        finally: ratio-clip R_divN ≤ rho_clip * Σ_raw[b].
        """
        P, B = phi_train.shape
        r = J_indices.numel()
        device = phi_train.device
        if r == 0:
            return torch.zeros(B, device=device, dtype=torch.float32), {"lambda_eff":0.0}

        # Center (along dataset)
        phi_f = phi_train.to(torch.float32)                 # (P,B)
        if self.react.center_phi:
            mu = phi_f.mean(dim=0, keepdim=True)            # (1,B)
            phi_c = phi_f - mu
        else:
            phi_c = phi_f

        # Variance (per chain) for whitening
        var_b = (phi_c * phi_c).mean(dim=0).clamp_min(1e-12)   # (B,)
        std_b = var_b.sqrt()                                    # (B,)

        # Select probe set
        Phi_J = phi_c[:, J_indices]                          # (P,r)
        var_j = (Phi_J * Phi_J).mean(dim=0).clamp_min(1e-12) # (r,)
        std_j = var_j.sqrt()

        # Build whitened features: ϕ̂_b = ϕ_c_b / std_b, ϕ̂_J = ϕ_c_J / std_j
        if self.react.use_corr:
            Phi_hat = phi_c / std_b.view(1, -1)              # (P,B)
            Phi_J_hat = Phi_J / std_j.view(1, -1)            # (P,r)
            # Correlations ρ = E[ϕ̂_b ϕ̂_j] (bounded in [-1,1])
            Rho = (Phi_hat.t() @ Phi_J_hat) / float(P)       # (B,r)
            M2 = Rho.square()                                # (B,r)
        else:
            # Raw inner products (already centered), normalized by P
            M2 = ((phi_c.t() @ Phi_J) / float(P)).square()   # (B,r)

        # a2 for reaction: variance-only or full
        N2g = (float(self.mdl.N) ** (2.0 * float(self.mdl.gamma)))
        if self.react.a2_mode.lower() == "full":
            a2_all = N2g * (c.to(torch.float32) ** 2 + (self.kappa ** 2) / D_oracle.to(torch.float32))
        else:  # "var" (connected)
            a2_all = N2g * ((self.kappa ** 2) / D_oracle.to(torch.float32))
        a2_J = a2_all[J_indices]                             # (r,)

        # Annealed λ
        if self.react.anneal_iters and self.react.anneal_iters > 0:
            anneal = min(1.0, max(0.0, outer_iter / float(self.react.anneal_iters)))
        else:
            anneal = 1.0
        lambda_eff = float(self.react.lambda_react) * anneal

        # Raw reaction estimate
        R_divN_raw = (M2 @ a2_J) * (lambda_eff / (self.kappa ** 2 * float(r)))  # (B,)

        # Per-chain ratio clip: R_divN ≤ rho_clip * Σ_raw
        if self.react.rho_clip is not None and self.react.rho_clip > 0:
            cap = self.react.rho_clip * Sigma_raw.to(torch.float32)
            R_divN = torch.minimum(R_divN_raw, cap)
        else:
            R_divN = R_divN_raw

        stats = {
            "lambda_eff": lambda_eff,
            "mean_rsq": float(M2.mean().item()),
            "mean_a2_react": float(a2_J.mean().item()),
            "raw_mean_R_divN": float(R_divN_raw.mean().item()),
            "clipped_mean_R_divN": float(R_divN.mean().item()),
        }
        return torch.clamp(R_divN, min=0.0), stats

    def run(self, X_train: torch.Tensor, y_train: torch.Tensor,
            X_eval: torch.Tensor, y_eval: torch.Tensor,
            outer_steps: Optional[int] = None,
            log_dir: str = "./results_cavity_gamma", run_tag: str = "") -> Dict:

        os.makedirs(log_dir, exist_ok=True)
        if outer_steps is None: outer_steps = self.sol.outer_steps

        P_train, d = X_train.shape
        f_mean = torch.zeros(P_train, device=self.device, dtype=torch.float32)

        # init chains
        W = torch.randn(self.mcmc.B, d, device=self.device) * (self.mdl.sigma_w / math.sqrt(self.mdl.d))

        # optional characters (for your parity diagnostics)
        teacher_sets = self.teacher_sets
        M_comp = len(teacher_sets)
        C_train = compute_characters_matrix(X_train, teacher_sets) if M_comp > 0 else torch.zeros(P_train, 0, device=self.device)
        C_eval  = compute_characters_matrix(X_eval,  teacher_sets) if M_comp > 0 else torch.zeros(X_eval.shape[0], 0, device=self.device)
        true_coeff = torch.ones(M_comp, device=self.device, dtype=torch.float32)

        traj = {
            "iter": [], "time_s": [],
            "train_mse": [], "train_corr": [], "train_y2": [], "train_f2": [],
            "eval_mS_vec": [], "eval_mS_norm2": [],
            "eval_coeff_mse": [], "eval_coeff_mae": [], "eval_coeff_sign_acc": [],
            "eval_noise_norm2": [], "eval_mse_total": [], "eval_corr_yf": [], "eval_R2": [],
            "mean_Sigma": [], "mean_abs_Jr": [], "kappa2N2A": [], "mean_c": [],
            "chi_AA_vec": [], "chi_AA_norm1": [], "chi_AA_norm2": [],
            # 1/N diagnostics:
            "mean_R_divN": [], "mean_R_over_Sigma": [],
            "mean_Sigma_eff": [], "min_Sigma_eff": [], "frac_Sigma_eff_le0": [],
            "probe_rank_r": [], "rho_clip": [], "lambda_react": [], "lambda_eff": [], "ema": [],
            "center_phi": [], "use_corr": [], "a2_mode": [], "clamp_sigma_eff_min": [],
            "rsq_mean": [], "a2_react_mean": [], "raw_mean_R_divN": [], "clipped_mean_R_divN": []
        }
        t0 = time.time()

        Acoef = 1.0 / (2.0 * (self.mdl.sigma_a ** 2))
        kappa2N2A = (self.kappa ** 2) * (self.mdl.N ** (2.0 * self.mdl.gamma)) * Acoef

        B = self.mcmc.B
        if self._R_divN_prev is None or self._R_divN_prev.numel() != B:
            self._R_divN_prev = torch.zeros(B, device=self.device, dtype=torch.float32)
        if (self._J_indices is None) or (self._J_indices.numel() != min(self.react.r, B)):
            self._J_indices = self._sample_J_indices(B)

        for it in range(1, outer_steps + 1):
            # SGLD with lag-1 reaction & Σ_eff clamp
            W, phi_train, D_or_old, JY, Jm, Sigma_raw, Sigma_eff_lag = sgld_sample_w(
                W, X_train, y_train, f_mean, self.kappa, self.mdl, self.mcmc,
                R_divN=self._R_divN_prev, clamp_sigma_eff_min=self.react.clamp_sigma_eff_min
            )
            Jr = (JY - Jm).to(torch.float32)

            # possibly resample probe set
            if (self.react.resample_every is not None) and (self.react.resample_every > 0):
                if (it % self.react.resample_every) == 0:
                    self._J_indices = self._sample_J_indices(B)

            # first pass c_lag from old D_or
            c_lag = (Jr / D_or_old).to(torch.float32)

            # new reaction (low-rank, connected by default), with annealed λ and ratio clip
            R_divN_calc, rstats = self.compute_reaction_lowrank(
                phi_train=phi_train,
                Sigma_raw=Sigma_raw,
                c=c_lag,
                D_oracle=D_or_old,
                J_indices=self._J_indices,
                outer_iter=it
            )

            # EMA smoothing
            if self.react.ema is not None and 0.0 < self.react.ema < 1.0:
                R_divN_new = self.react.ema * self._R_divN_prev + (1.0 - self.react.ema) * R_divN_calc
            else:
                R_divN_new = R_divN_calc

            # corrected D_or & c (using new reaction)
            Sigma_eff_now = Sigma_raw - R_divN_new
            if self.react.clamp_sigma_eff_min is not None:
                Sigma_eff_now = torch.maximum(Sigma_eff_now, torch.tensor(self.react.clamp_sigma_eff_min, device=self.device, dtype=torch.float32))
            D_or_corr = (kappa2N2A + Sigma_eff_now).clamp_min(1e-12)
            c = (Jr / D_or_corr).to(torch.float32)

            # update lag-1 reaction
            self._R_divN_prev = R_divN_new.detach()

            # mean function update: ⟨f⟩ = N * E_b[c φ]
            f_chunk_mean = (phi_train.to(torch.float32) * c.view(1, -1)).mean(dim=1)
            f_new_train = self.mdl.N * f_chunk_mean

            a_t = self.sol.saem_a0 / (it + self.sol.saem_t0)
            f_mean = (1 - self.sol.saem_damping * a_t) * f_mean + self.sol.saem_damping * a_t * f_new_train

            # training diag
            train_mse = float(((y_train - f_mean) ** 2).mean().item())
            train_corr = float((y_train * f_mean).mean().item())
            train_y2 = float((y_train * y_train).mean().item())
            train_f2 = float((f_mean * f_mean).mean().item())

            # susceptibility on modes (optional parity diag)
            if M_comp > 0:
                chi_AA = self.compute_susceptibility_AA(phi_train, c, D_or_corr, C_train, chunk_B=2048)
                chi_AA_list = chi_AA.detach().cpu().tolist()
                chi_AA_norm1 = float(chi_AA.abs().sum().item())
                chi_AA_norm2 = float((chi_AA * chi_AA).sum().sqrt().item())
            else:
                chi_AA_list, chi_AA_norm1, chi_AA_norm2 = [], 0.0, 0.0

            # eval
            with torch.no_grad():
                f_eval = self.predict_mean_f_on(X_eval, W, c, chunk_B=1024*1024)
                if M_comp > 0:
                    m_vec = (C_eval * f_eval.view(-1, 1)).mean(dim=0)
                    f_signal = (C_eval * m_vec.view(1, -1)).sum(dim=1)
                    r = f_eval - f_signal
                    mS_vec = m_vec.detach().cpu().tolist()
                    mS_norm2 = float((m_vec * m_vec).sum().item())
                    coeff_mse = float(((m_vec - true_coeff) ** 2).mean().item())
                    coeff_mae = float((m_vec - true_coeff).abs().mean().item())
                    coeff_sign_acc = float(((m_vec > 0).float().mean()).item())
                    noise_norm2 = float((r * r).mean().item())
                else:
                    mS_vec, mS_norm2 = [], 0.0
                    coeff_mse = coeff_mae = coeff_sign_acc = float('nan')
                    noise_norm2 = float('nan')

                mse_total = float(((f_eval - y_eval) ** 2).mean().item())
                var_y = float(((y_eval - y_eval.mean()) ** 2).mean().item())
                yf = float((y_eval * f_eval).mean().item())
                corr_yf = yf / max(var_y, 1e-12) if var_y > 0 else float('nan')
                R2 = 1.0 - mse_total / max(var_y, 1e-12) if var_y > 0 else float('nan')

            # 1/N diagnostics
            frac_le0 = float((Sigma_eff_now <= 0).to(torch.float32).mean().item())
            mean_R_divN = float(R_divN_new.mean().item())
            mean_R_over_Sigma = float((R_divN_new / (Sigma_raw + 1e-12)).mean().item())
            mean_Sigma_eff = float(Sigma_eff_now.mean().item())
            min_Sigma_eff = float(Sigma_eff_now.min().item())

            # log
            traj["iter"].append(it)
            traj["time_s"].append(time.time() - t0)
            traj["train_mse"].append(train_mse)
            traj["train_corr"].append(train_corr)
            traj["train_y2"].append(train_y2)
            traj["train_f2"].append(train_f2)
            traj["eval_mS_vec"].append(mS_vec)
            traj["eval_mS_norm2"].append(mS_norm2)
            traj["eval_coeff_mse"].append(coeff_mse)
            traj["eval_coeff_mae"].append(coeff_mae)
            traj["eval_coeff_sign_acc"].append(coeff_sign_acc)
            traj["eval_noise_norm2"].append(noise_norm2)
            traj["eval_mse_total"].append(mse_total)
            traj["eval_corr_yf"].append(corr_yf)
            traj["eval_R2"].append(R2)
            traj["mean_Sigma"].append(float(Sigma_raw.mean().item()))
            traj["mean_abs_Jr"].append(float(Jr.abs().mean().item()))
            traj["kappa2N2A"].append(float(kappa2N2A))
            traj["mean_c"].append(float(c.mean().item()))
            traj["chi_AA_vec"].append(chi_AA_list)
            traj["chi_AA_norm1"].append(chi_AA_norm1)
            traj["chi_AA_norm2"].append(chi_AA_norm2)
            traj["mean_R_divN"].append(mean_R_divN)
            traj["mean_R_over_Sigma"].append(mean_R_over_Sigma)
            traj["mean_Sigma_eff"].append(mean_Sigma_eff)
            traj["min_Sigma_eff"].append(min_Sigma_eff)
            traj["frac_Sigma_eff_le0"].append(frac_le0)
            traj["probe_rank_r"].append(int(self.react.r))
            traj["rho_clip"].append(float(self.react.rho_clip))
            traj["lambda_react"].append(float(self.react.lambda_react))
            traj["lambda_eff"].append(float(rstats["lambda_eff"]))
            traj["ema"].append(float(self.react.ema))
            traj["center_phi"].append(bool(self.react.center_phi))
            traj["use_corr"].append(bool(self.react.use_corr))
            traj["a2_mode"].append(self.react.a2_mode)
            traj["clamp_sigma_eff_min"].append(float(self.react.clamp_sigma_eff_min))
            traj["rsq_mean"].append(float(rstats["mean_rsq"]))
            traj["a2_react_mean"].append(float(rstats["mean_a2_react"]))
            traj["raw_mean_R_divN"].append(float(rstats["raw_mean_R_divN"]))
            traj["clipped_mean_R_divN"].append(float(rstats["clipped_mean_R_divN"]))

            if it % self.sol.print_every == 1 or it == outer_steps:
                m_preview = mS_vec[:8] if len(mS_vec) > 0 else []
                chi_preview = chi_AA_list[:8] if len(chi_AA_list) > 0 else []
                print(json.dumps({
                    "iter": it,
                    "train_mse": train_mse,
                    "train_corr": train_corr,
                    "eval_mse_total": mse_total,
                    "eval_R2": R2,
                    "eval_noise_norm2": noise_norm2,
                    "eval_mS_norm2": mS_norm2,
                    "eval_mS_head": m_preview,
                    "coeff_mse": coeff_mse,
                    "coeff_sign_acc": coeff_sign_acc,
                    "chi_AA_norm2": chi_AA_norm2,
                    "chi_AA_head": chi_preview,
                    "mean_Sigma": float(Sigma_raw.mean().item()),
                    "mean_abs_Jr": float(Jr.abs().mean().item()),
                    "kappa2N2A": float(kappa2N2A),
                    "mean_c": float(c.mean().item()),
                    "mean_R_divN": mean_R_divN,
                    "mean_R_over_Sigma": mean_R_over_Sigma,
                    "mean_Sigma_eff": mean_Sigma_eff,
                    "min_Sigma_eff": min_Sigma_eff,
                    "frac_Sigma_eff_le0": frac_le0,
                    "probe_rank_r": int(self.react.r),
                    "rho_clip": float(self.react.rho_clip),
                    "lambda_react": float(self.react.lambda_react),
                    "lambda_eff": float(rstats["lambda_eff"]),
                    "ema": float(self.react.ema),
                    "center_phi": bool(self.react.center_phi),
                    "use_corr": bool(self.react.use_corr),
                    "a2_mode": self.react.a2_mode,
                    "clamp_sigma_eff_min": float(self.react.clamp_sigma_eff_min),
                    "rsq_mean": float(rstats["mean_rsq"]),
                    "a2_react_mean": float(rstats["mean_a2_react"]),
                    "raw_mean_R_divN": float(rstats["raw_mean_R_divN"]),
                    "clipped_mean_R_divN": float(rstats["clipped_mean_R_divN"]),
                    "elapsed_s": round(traj["time_s"][-1], 2),
                    "B": self.mcmc.B, "P_train": P_train, "P_eval": X_eval.shape[0],
                    "num_components": M_comp
                }))

        out = {
            "summary": {
                "P_train": int(P_train),
                "P_eval": int(X_eval.shape[0]),
                "d": self.mdl.d,
                "N": self.mdl.N,
                "gamma": self.mdl.gamma,
                "kappa": self.kappa,
                "act": self.mdl.act,
                "sigma_a": self.mdl.sigma_a,
                "num_components": M_comp,
                "teacher_sets": [],  # omit big arrays
                "mS_last": traj["eval_mS_vec"][-1],
                "mS_norm2_last": traj["eval_mS_norm2"][-1],
                "chi_AA_last": traj["chi_AA_vec"][-1],
                "chi_AA_norm2_last": traj["chi_AA_norm2"][-1],
                "noise_norm2_last": traj["eval_noise_norm2"][-1],
                "mse_total_last": traj["eval_mse_total"][-1],
                "R2_last": traj["eval_R2"][-1],
                # 1/N snapshot
                "mean_R_divN_last": traj["mean_R_divN"][-1],
                "mean_R_over_Sigma_last": traj["mean_R_over_Sigma"][-1],
                "mean_Sigma_eff_last": traj["mean_Sigma_eff"][-1],
                "min_Sigma_eff_last": traj["min_Sigma_eff"][-1],
                "frac_Sigma_eff_le0_last": traj["frac_Sigma_eff_le0"][-1],
                "probe_rank_r": int(self.react.r),
                "rho_clip": float(self.react.rho_clip),
                "lambda_react": float(self.react.lambda_react),
                "ema": float(self.react.ema),
                "use_corr": bool(self.react.use_corr),
                "a2_mode": self.react.a2_mode,
                "clamp_sigma_eff_min": float(self.react.clamp_sigma_eff_min),
            },
            "traj": traj,
            "config": {
                "mdl": vars(self.mdl),
                "mcmc": vars(self.mcmc),
                "sol": vars(self.sol),
                "kappa": self.kappa,
                "reaction": vars(self.react),
            }
        }
        tag = run_tag or time.strftime("%Y%m%d_%H%M%S")
        gstr = f"{self.mdl.gamma:.6g}".replace('.', 'p').replace('-', 'm')
        path = os.path.join(
            log_dir,
            f"cavity_func_gamma_composite_{tag}"
            f"_P{P_train}_Neval{X_eval.shape[0]}"
            f"_kap{float(self.kappa):.3e}"
            f"_N{int(self.mdl.N)}_g{gstr}"
            f"_M{M_comp}.json"
        )
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {path}")
        return out

# ----------------------------- Data -----------------------------
def generate_composite_data(P: int, d: int, sets: List[torch.Tensor], device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(0)
    X = (torch.randint(0, 2, (P, d), generator=g, device=device, dtype=torch.int8).float() * 2.0 - 1.0)
    if len(sets) == 0:
        y = torch.zeros(P, device=device, dtype=torch.float32)
        C = torch.zeros(P, 0, device=device, dtype=torch.float32)
        return X, y, C
    C = compute_characters_matrix(X, sets)
    y = C.sum(dim=1)
    return X, y, C

# ----------------------------- Entrypoint -----------------------------
if __name__ == "__main__":
    set_seed(42)
    _ = check_gpu()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---- Teacher/data ----
    d = 35
    teacher_spec = "{0,1,2,3}"
    sets_idx_lists = parse_composite_spec(teacher_spec)
    teacher_sets = [torch.tensor(s, device=device, dtype=torch.long) for s in sets_idx_lists]

    # Eval set
    P_eval = 50000
    X_eval,  y_eval,  _ = generate_composite_data(P_eval, d, teacher_sets, device)

    # ---- Model/MCMC/Solver ----
    N = 512
    GAMMA_LIST = [0.5]

    mcmc = MCMCParams(
        B=1024*4, steps=600, step_size=5e-5, step_decay=0.999999,
        grad_clip=0.0, clamp_w=0.0, autocast=False
    )
    sol  = SolveParams(
        outer_steps=4000, saem_a0=0.2, saem_t0=80.0, saem_damping=0.5,
        print_every=10
    )

    react = ReactionParams(
        r=4096,
        resample_every=1,
        a2_mode="full",          # <— connected susceptibility by default
        use_corr=True,
        center_phi=True,
        lambda_react=0.02,      # small; annealed to this by 500 iters
        ema=0.99,
        anneal_iters=100,
        rho_clip=0.35,
        clamp_sigma_eff_min=1e-3
    )

    P_TRAIN_LIST = [10000]
    KAPPA_LIST   = [7.5e-3]  # start with the problematic small kappa

    results_dir = "./results_cavity_gamma"
    os.makedirs(results_dir, exist_ok=True)

    all_runs: List[Dict] = []
    for gamma in GAMMA_LIST:
        for P_train in P_TRAIN_LIST:
            # Generate train set
            X_train, y_train, _ = generate_composite_data(P_train, d, teacher_sets, device)
            for kappa in KAPPA_LIST:
                print(f"\n=== RUN: P_train={P_train}, kappa={kappa:.3e}, gamma={gamma} ===")
                mdl  = ModelParams(d=d, N=N, k=0, sigma_a=1.0, sigma_w=1.0, gamma=gamma, act="relu")

                solver = FunctionalCavitySolver(mdl, mcmc, sol, kappa=kappa, device=device,
                                                teacher_sets=teacher_sets, react=react)

                gstr = f"{gamma:.6g}".replace('.', 'p').replace('-', 'm')
                run_tag = f"P{P_train}_kap{float(kappa):.3e}_g{gstr}"

                out = solver.run(X_train, y_train, X_eval, y_eval,
                                 log_dir=results_dir, run_tag=run_tag)

                all_runs.append({
                    "P_train": P_train,
                    "kappa": float(kappa),
                    "gamma": float(gamma),
                    "path": os.path.join(
                        results_dir,
                        f"cavity_func_gamma_composite_{run_tag}"
                        f"_P{P_train}_Neval{P_eval}"
                        f"_kap{float(kappa):.3e}"
                        f"_N{int(N)}_g{gstr}"
                        f"_M{len(teacher_sets)}.json"
                    ),
                    "summary": out["summary"]
                })

    index_path = os.path.join(results_dir, "sweep_index.json")
    with open(index_path, "w") as f:
        json.dump({"runs": all_runs}, f, indent=2)
    print(f"\n[sweep saved] index: {index_path}")
