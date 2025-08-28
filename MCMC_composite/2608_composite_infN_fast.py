# cavity_functional_selfconsistency_gamma_composite.py
# Fully corrected implementation with:
#   • Gamma sweep, filenames include N and g{gamma}
#   • (New) MALA with dual-averaging warmup (no hand LR), then frozen step
#   • (New) Fixed-point drift metric + early stop on eval stability
#   • Linear warmup of SGLD step size (fallback path if use_mala=False)
#   • Chain-wise curvature preconditioning for SGLD fallback
#   • Vectorized/H100-friendly evaluation (bf16 autocast for GEMMs, float32 math)
#
# Core formulas:
#   ⟨f(x)⟩ = N * E_b[c(w_b) φ(w_b^T x)],   c = μ/N^γ = Jr / (κ^2 N^{2γ} α) = Jr / D_oracle
#   α(w) = 1/(2σ_a^2) + Σ/(κ^2 N^{2γ}),  β = Jr/(κ^2 N^γ),  μ = β/α,  σ^2 = 1/α
#   D_oracle = κ^2 N^{2γ} α = κ^2 N^{2γ} * 1/(2σ_a^2) + Σ
#   χ_AA = (N/κ^2) E_b[ E[a^2|w_b] * J_A(w_b)^2 ],  with E[a^2|w_b] = N^{2γ}(c_b^2 + κ^2/D_oracle_b)

import os, json, time, math, random, re
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import numpy as np
import torch
import torch._dynamo as dynamo

# ----------------------------- Performance defaults (H100) -----------------------------
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision("high")  # allow TF32 fast paths
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

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
    return S.sort().values

def parity_character(X_pm1: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """
    χ_S(x) = ∏_{i in S} x_i  for x_i in {±1}.
    """
    if S.numel() == 0:
        return torch.ones(X_pm1.shape[0], device=X_pm1.device, dtype=torch.float32)
    return X_pm1[:, S].prod(dim=1).to(torch.float32)

def parity_labels(X_pm1: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    return parity_character(X_pm1, S)

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

def compute_characters_matrix(X_pm1: torch.Tensor, sets: List[torch.Tensor]) -> torch.Tensor:
    P = X_pm1.shape[0]
    M = len(sets)
    if M == 0:
        return torch.zeros(P, 0, device=X_pm1.device, dtype=torch.float32)
    C = torch.empty(P, M, device=X_pm1.device, dtype=torch.float32)
    for j, Sj in enumerate(sets):
        C[:, j] = parity_character(X_pm1, Sj)
    return C

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
    steps: int = 200
    step_size: float = 1e-3          # initial ε for MALA; will adapt during warmup
    step_decay: float = 1.0          # unused by MALA (kept for SGLD fallback)
    grad_clip: float = 10.0
    clamp_w: Optional[float] = None  # avoid bias; MALA keeps things stable
    langevin_sqrt2: bool = True
    autocast: Optional[bool] = True  # bf16 matmuls on CUDA

    # --- MALA controls ---
    use_mala: bool = True
    warmup_steps: int = 100
    target_accept: float = 0.574      # optimal for MALA in many dims
    da_kappa: float = 0.75            # dual-averaging decay exponent
    da_t0: float = 10.0               # dual-averaging offset
    step_clip_min: float = 1e-6       # hard clamps for ε during warmup
    step_clip_max: float = 1e-1
    # (kept for SGLD fallback)
    warmup_start_frac: float = 0.01


@dataclass
class SolveParams:
    outer_steps: int = 500
    saem_a0: float = 1.0
    saem_t0: float = 20.0
    saem_damping: float = 1.0
    print_every: int = 10

# ----------------------------- SGLD/MALA over w (integrate out a) -----------------------------
@dynamo.disable
@dynamo.disable
def sgld_sample_w(
    w: torch.Tensor,        # (B,d)
    X: torch.Tensor,        # (P,d), entries ±1
    y: torch.Tensor,        # (P,)
    f_mean: torch.Tensor,   # (P,)
    kappa: float,
    mdl: ModelParams,
    mcmc: MCMCParams,
):
    """
    Preconditioned MALA (per-chain scalar metric C = 1/alpha) with dual-averaging warmup.
    Falls back to preconditioned SGLD if mcmc.use_mala=False.
    Energy:
        U(w) = (d/(2 σ_w^2)) ||w||^2 + 0.5 log α(w) - 0.5 * [Jr(w)^2] / [κ^4 N^{2γ} α(w)]
    α(w) = 1/(2σ_a^2) + Σ(w)/(κ^2 N^{2γ}),   Jr = E[φ y] - E[φ ⟨f⟩]
    D_oracle = κ^2 N^{2γ} α = κ^2 N^{2γ} * 1/(2σ_a^2) + Σ(w)
    """
    device = w.device
    P = X.shape[0]
    N = float(mdl.N)
    gamma = float(mdl.gamma)
    N2g = N ** (2.0 * gamma)

    var_w_per_coord = (mdl.sigma_w ** 2) / mdl.d         # prior variance per coord = σ_w^2/d
    Acoef = 1.0 / (2.0 * (mdl.sigma_a ** 2))
    d_dim = w.shape[1]

    autocast_enabled = (mcmc.autocast if (mcmc.autocast is not None) else w.is_cuda)
    y_f = y.view(-1).to(torch.float32)
    m_f = f_mean.view(-1).to(torch.float32)

    # ---- helper: U, grad, and cached stats (vectorized over chains) ----
    def energy_and_grad(w_in: torch.Tensor):
        w_in = w_in.detach().requires_grad_(True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            z   = X @ w_in.t().contiguous()          # (P,B)
            phi = activation(z, mdl.act)             # bf16
        g     = phi.to(torch.float32)                # (P,B)
        Sigma = (g * g).mean(dim=0)                  # (B,)
        JY    = (g.t() @ y_f) / float(P)             # (B,)
        Jm    = (g.t() @ m_f) / float(P)             # (B,)
        Jr    = JY - Jm                              # (B,)

        alpha = (Acoef + Sigma / (kappa * kappa * N2g)).clamp_min(1e-12)  # α(w): (B,)
        prior   = 0.5 * (w_in * w_in).sum(dim=1) / var_w_per_coord        # (B,)
        log_det = 0.5 * torch.log(alpha)                                  # (B,)
        data_q  = (-0.5) * (Jr * Jr) / ((kappa ** 4) * N2g * alpha + 1e-30)
        U = prior + log_det + data_q                                      # (B,)

        grad = torch.autograd.grad(U.sum(), w_in, retain_graph=False, create_graph=False)[0]
        grad = torch.where(torch.isfinite(grad), grad, torch.zeros_like(grad))
        if mcmc.grad_clip and mcmc.grad_clip > 0:
            gn = grad.norm(dim=1, keepdim=True).clamp_min(1e-12)
            grad = grad * (mcmc.grad_clip / gn).clamp(max=1.0)

        # per-chain scalar preconditioner
        C = (1.0 / (alpha.to(torch.float32) + 1e-8)).clamp_max(1e4)       # (B,)
        return (
            U.detach(), grad, C.detach(),
            phi.detach(), JY.detach(), Jm.detach(), Sigma.detach(), alpha.detach()
        )

    # ---------------- MALA path ----------------
    if mcmc.use_mala:
        log_step = math.log(max(mcmc.step_size, 1e-12))
        log_step_avg = log_step
        Hbar = 0.0

        U, grad, C, phi, JY, Jm, Sigma, alpha = energy_and_grad(w)

        for t in range(1, mcmc.steps + 1):
            # step with clamps during warmup
            eps = math.exp(log_step)
            if t <= mcmc.warmup_steps:
                eps = float(np.clip(eps, mcmc.step_clip_min, mcmc.step_clip_max))

            C_col   = C.view(-1, 1)                         # (B,1)
            sqrtC   = torch.sqrt(C_col)                     # (B,1)
            noise   = torch.randn_like(w)

            mean_prop = w - eps * (C_col * grad)            # (B,d)
            w_prop    = mean_prop + noise * math.sqrt(2.0 * eps) * sqrtC

            if mcmc.clamp_w is not None:
                w_prop = w_prop.clamp(-mcmc.clamp_w, mcmc.clamp_w)

            U_p, grad_p, C_p, phi_p, JY_p, Jm_p, Sigma_p, alpha_p = energy_and_grad(w_prop)

            # --- MH correction with metric C (scalar per chain) ---
            # q(w'|w) = N( w - eps C grad, 2 eps C I )
            # log q(w | w') - log q(w' | w) =
            #   -||w - w' + eps C'(grad')||^2 / (4 eps C') - (d/2)log C'
            #   +||w' - w + eps C (grad)||^2 / (4 eps C)   + (d/2)log C
            C_vec  = C
            C_pvec = C_p
            C_col  = C_vec.view(-1, 1)
            Cp_col = C_pvec.view(-1, 1)

            delta1 = (w - w_prop + eps * (Cp_col * grad_p))   # (B,d)
            delta2 = (w_prop - w + eps * (C_col  * grad  ))   # (B,d)

            term1 = - (delta1.pow(2).sum(dim=1)) / (4.0 * eps * C_pvec + 1e-30) - 0.5 * d_dim * torch.log(C_pvec + 1e-30)
            term2 = + (delta2.pow(2).sum(dim=1)) / (4.0 * eps * C_vec  + 1e-30) + 0.5 * d_dim * torch.log(C_vec  + 1e-30)
            log_q_ratio = term1 + term2                       # (B,)

            log_alpha = (-U_p + U) + log_q_ratio              # (B,)
            # accept with probability min(1, exp(log_alpha))
            u = torch.rand_like(log_alpha)
            accept_mask = (torch.log(u) < log_alpha).float()  # (B,)

            # correct broadcasting masks
            acc_vec = accept_mask.view(-1)      # (B,)
            acc_col = accept_mask.view(-1, 1)   # (B,1)
            acc_row = accept_mask.view(1, -1)   # (1,B)

            # mix back accepted proposals
            w       = acc_col * w_prop   + (1.0 - acc_col) * w
            U       = acc_vec * U_p      + (1.0 - acc_vec) * U
            grad    = acc_col * grad_p   + (1.0 - acc_col) * grad
            C       = acc_vec * C_p      + (1.0 - acc_vec) * C
            phi     = acc_row * phi_p    + (1.0 - acc_row) * phi
            JY      = acc_vec * JY_p     + (1.0 - acc_vec) * JY
            Jm      = acc_vec * Jm_p     + (1.0 - acc_vec) * Jm
            Sigma   = acc_vec * Sigma_p  + (1.0 - acc_vec) * Sigma
            alpha   = acc_vec * alpha_p  + (1.0 - acc_vec) * alpha

            # --- dual-averaging warmup using actual acceptance rate
            if t <= mcmc.warmup_steps:
                acc_rate = float(accept_mask.mean().item())
                Hbar = (1.0 - 1.0 / (t + mcmc.da_t0)) * Hbar + (mcmc.target_accept - acc_rate) / (t + mcmc.da_t0)
                eta_t = t ** (-mcmc.da_kappa)
                log_step = log_step - eta_t * Hbar
                log_step_avg = eta_t * log_step + (1.0 - eta_t) * log_step_avg
            elif t == mcmc.warmup_steps + 1:
                log_step = log_step_avg  # freeze

        # final outputs
        Jr = (JY - Jm).to(torch.float32)
        D_oracle = ((kappa ** 2) * N2g * Acoef + Sigma).clamp_min(1e-12)
        return (
            w.detach(),
            phi.detach(),
            D_oracle.detach(),
            JY.detach(),
            Jm.detach(),
            Sigma.detach()
        )

    # ---------------- SGLD fallback (preconditioned; if use_mala=False) ----------------
    stiff = (kappa ** 4) * (mdl.N ** (2.0 * mdl.gamma)) * P
    base_step = mcmc.step_size * max(stiff, 1e-12)

    for t in range(1, mcmc.steps + 1):
        w = w.detach().requires_grad_(True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            z = X @ w.t().contiguous()
            phi = activation(z, mdl.act)
        g = phi.to(torch.float32)
        Sigma = (g * g).mean(dim=0)
        JY    = (g.t() @ y_f) / float(P)
        Jm    = (g.t() @ m_f) / float(P)
        Jr    = JY - Jm

        alpha = (Acoef + Sigma / (kappa * kappa * N2g)).clamp_min(1e-12)
        prior   = 0.5 * (w * w).sum(dim=1) / var_w_per_coord
        log_det = 0.5 * torch.log(alpha)
        data_q  = (-0.5) * (Jr * Jr) / ((kappa ** 4) * N2g * alpha + 1e-30)
        U = prior + log_det + data_q

        grad = torch.autograd.grad(U.sum(), w, retain_graph=False, create_graph=False)[0]
        grad = torch.where(torch.isfinite(grad), grad, torch.zeros_like(grad))
        if mcmc.grad_clip and mcmc.grad_clip > 0:
            gn = grad.norm(dim=1, keepdim=True).clamp_min(1e-12)
            grad = grad * (mcmc.grad_clip / gn).clamp(max=1.0)

        # simple warmup+decay schedule (SGLD only)
        warmup_frac = min(t / float(max(mcmc.warmup_steps, 1)), 1.0)
        warm = mcmc.warmup_start_frac + (1.0 - mcmc.warmup_start_frac) * warmup_frac
        decay = mcmc.step_decay ** (t - 1)
        step = base_step * warm * decay

        C = (1.0 / (alpha.to(torch.float32) + 1e-8)).clamp_max(1e4)
        C_col = C.view(-1, 1)
        sqrtC = torch.sqrt(C_col)

        noise = torch.randn_like(w)
        if mcmc.langevin_sqrt2:
            w = w - step * (C_col * grad) + noise * math.sqrt(2.0 * step) * sqrtC
        else:
            w = w - 0.5 * step * (C_col * grad) + noise * math.sqrt(step) * sqrtC

        if mcmc.clamp_w is not None:
            w = torch.clamp(w, -mcmc.clamp_w, mcmc.clamp_w)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        z = X @ w.t().contiguous()
        phi = activation(z, mdl.act).detach()
    g = phi.to(torch.float32)
    Sigma = (g * g).mean(dim=0)
    JY    = (g.t() @ y_f) / float(P)
    Jm    = (g.t() @ m_f) / float(P)
    Jr    = JY - Jm
    D_oracle = ((kappa ** 2) * N2g * Acoef + Sigma).clamp_min(1e-12)

    return (
        w.detach(),
        phi.detach(),
        D_oracle.detach(),
        JY.detach(),
        Jm.detach(),
        Sigma.detach()
    )




# ----------------------------- Functional self-consistency loop -----------------------------
class FunctionalCavitySolver:
    def __init__(self, mdl: ModelParams, mcmc: MCMCParams, sol: SolveParams,
                 kappa: float, device: torch.device,
                 teacher_sets: Optional[List[torch.Tensor]] = None):
        self.mdl = mdl
        self.mcmc = mcmc
        self.sol = sol
        self.kappa = kappa
        self.device = device
        self.teacher_sets = teacher_sets or []

    @torch.no_grad()
    def predict_mean_f_on(self, X_eval: torch.Tensor, W: torch.Tensor, c: torch.Tensor,
                          chunk_P: int = 200_000, chunk_B: Optional[int] = None) -> torch.Tensor:
        """
        Vectorized prediction:
            ⟨f(x)⟩ = N * E_b[c_b φ(w_b^T x)] = N * φ(x) @ (c / B)
        """
        P_eval = X_eval.shape[0]
        B = W.shape[0]
        coeff = (c.to(torch.float32) / float(B)).view(-1, 1)  # (B,1)
        f = torch.empty(P_eval, device=X_eval.device, dtype=torch.float32)

        use_autocast = (self.mcmc.autocast if (self.mcmc.autocast is not None) else X_eval.is_cuda)

        if chunk_B is None:
            for start in range(0, P_eval, chunk_P):
                stop = min(start + chunk_P, P_eval)
                Xe = X_eval[start:stop, :]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
                    z = Xe @ W.t().contiguous()          # (p,B)
                    gb = activation(z, self.mdl.act)     # bf16
                f[start:stop] = (gb.to(torch.float32) @ coeff).view(-1)
        else:
            for start in range(0, P_eval, chunk_P):
                stop = min(start + chunk_P, P_eval)
                Xe = X_eval[start:stop, :]
                accum = torch.zeros(stop - start, device=X_eval.device, dtype=torch.float32)
                for sb in range(0, B, chunk_B):
                    eb = min(sb + chunk_B, B)
                    Wb = W[sb:eb, :]
                    cb = coeff[sb:eb, :]
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
                        z = Xe @ Wb.t().contiguous()
                        gb = activation(z, self.mdl.act)
                    accum += (gb.to(torch.float32) @ cb).view(-1)
                f[start:stop] = accum

        return self.mdl.N * f

    @torch.no_grad()
    def project_onto_teacher_span(self, X: torch.Tensor, f: torch.Tensor, sets: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(sets) == 0:
            m_vec = torch.zeros(0, device=X.device, dtype=torch.float32)
            return m_vec, torch.zeros_like(f), f
        C = compute_characters_matrix(X, sets)      # (P, M)
        m_vec = (C * f.view(-1, 1)).mean(dim=0)     # (M,)
        f_signal = (C * m_vec.view(1, -1)).sum(dim=1)
        r = f - f_signal
        return m_vec, f_signal, r

    @torch.no_grad()
    def compute_susceptibility_AA(
        self,
        phi_train: torch.Tensor,     # (P, B) bf16
        c: torch.Tensor,             # (B,)
        D_oracle: torch.Tensor,      # (B,)
        C_train: torch.Tensor,       # (P, M)
        chunk_B: int = 4096
    ) -> torch.Tensor:
        """
        χ_AA[A] = (N / κ^2) * E_b[ E[a^2 | w_b] * J_A(w_b)^2 ],  with
        E[a^2|w_b] = N^{2γ} * ( c_b^2 + κ^2 / D_oracle_b )
        J_A(w_b) ≈ E_x[ φ(w_b^T x) χ_A(x) ]
        """
        if C_train.numel() == 0:
            return torch.zeros(0, device=phi_train.device, dtype=torch.float32)

        P, B = phi_train.shape
        M = C_train.shape[1]
        device = phi_train.device

        N = float(self.mdl.N)
        gamma = float(self.mdl.gamma)
        N2g = N ** (2.0 * gamma)

        a2 = N2g * (c.to(torch.float32) ** 2 + (self.kappa ** 2) / D_oracle.to(torch.float32))  # (B,)

        accum = torch.zeros(M, device=device, dtype=torch.float32)
        Ct = C_train.to(torch.float32).t().contiguous()          # (M, P)

        for start in range(0, B, chunk_B):
            stop = min(start + chunk_B, B)
            g_chunk = phi_train[:, start:stop].to(torch.float32)    # (P, Bb)
            J_chunk = (Ct @ g_chunk) / float(P)                     # (M, Bb)
            term = (a2[start:stop].view(1, -1) * (J_chunk * J_chunk))  # (M, Bb)
            accum += term.sum(dim=1)

        chi_AA = (N / (self.kappa ** 2)) * (accum / float(B))
        return chi_AA

    def run(self, X_train: torch.Tensor, y_train: torch.Tensor,
            X_eval: torch.Tensor, y_eval: torch.Tensor,
            outer_steps: Optional[int] = None,
            log_dir: str = "./results_cavity_gamma", run_tag: str = "") -> Dict:

        os.makedirs(log_dir, exist_ok=True)
        if outer_steps is None: outer_steps = self.sol.outer_steps

        P_train, d = X_train.shape

        # init mean function on train set
        f_mean = torch.zeros(P_train, device=self.device, dtype=torch.float32)

        # init SGLD/MALA chains for w ~ N(0, σ_w^2 I / d)
        W = torch.randn(self.mcmc.B, d, device=self.device, dtype=torch.float32) * (self.mdl.sigma_w / math.sqrt(self.mdl.d))

        # Precompute teacher characters
        teacher_sets = self.teacher_sets
        M_comp = len(teacher_sets)
        C_train = compute_characters_matrix(X_train, teacher_sets) if M_comp > 0 else torch.zeros(P_train, 0, device=self.device, dtype=torch.float32)
        C_eval  = compute_characters_matrix(X_eval,  teacher_sets) if M_comp > 0 else torch.zeros(X_eval.shape[0], 0, device=self.device, dtype=torch.float32)
        true_coeff = torch.ones(M_comp, device=self.device, dtype=torch.float32)

        traj = {
            "iter": [], "time_s": [],
            "train_mse": [], "train_corr": [], "train_y2": [], "train_f2": [],
            "eval_mS_vec": [], "eval_mS_norm2": [],
            "eval_coeff_mse": [], "eval_coeff_mae": [], "eval_coeff_sign_acc": [],
            "eval_noise_norm2": [], "eval_mse_total": [], "eval_corr_yf": [], "eval_R2": [],
            "mean_Sigma": [], "mean_abs_Jr": [], "kappa2N2A": [], "mean_c": [],
            "chi_AA_vec": [], "chi_AA_norm1": [], "chi_AA_norm2": [],
            "fp_drift": []
        }
        t0 = time.time()

        Acoef = 1.0 / (2.0 * (self.mdl.sigma_a ** 2))
        kappa2N2A = (self.kappa ** 2) * (self.mdl.N ** (2.0 * self.mdl.gamma)) * Acoef

        for it in range(1, outer_steps + 1):
            # E-step: sample w given current f_mean
            W, phi_train, D_or, JY, Jm, Sigma = sgld_sample_w(
                W, X_train, y_train, f_mean, self.kappa, self.mdl, self.mcmc
            )
            Jr = (JY - Jm).to(torch.float32)

            # c_b = Jr / D_oracle
            c = (Jr / D_or).to(torch.float32)           # (B,)

            # New mean on train: f_new_train = N * (φ * c).mean_b
            coeff = (c / float(self.mcmc.B)).view(-1, 1)                   # (B,1)
            f_new_train = self.mdl.N * (phi_train.to(torch.float32) @ coeff).view(-1)  # (P,)

            # ---- Fixed-point drift (before damping) ----
            denom = max(float((y_train * y_train).mean().sqrt().item()), 1e-8)
            drift = float(((f_new_train - f_mean).pow(2).mean().sqrt() / denom).item())

            # SAEM damping
            a_t = self.sol.saem_a0 / (it + self.sol.saem_t0)
            damp = self.sol.saem_damping * a_t
            f_mean = (1 - damp) * f_mean + damp * f_new_train

            # training diagnostics
            train_mse = float(((y_train - f_mean) ** 2).mean().item())
            train_corr = float((y_train * f_mean).mean().item())
            train_y2 = float((y_train * y_train).mean().item())
            train_f2 = float((f_mean * f_mean).mean().item())

            # Per-mode susceptibility on TRAIN
            if M_comp > 0:
                chi_AA = self.compute_susceptibility_AA(phi_train, c, D_or, C_train, chunk_B=4096)
                chi_AA_list = chi_AA.detach().cpu().tolist()
                chi_AA_norm1 = float(chi_AA.abs().sum().item())
                chi_AA_norm2 = float((chi_AA * chi_AA).sum().sqrt().item())
            else:
                chi_AA_list, chi_AA_norm1, chi_AA_norm2 = [], 0.0, 0.0

            # Evaluation (vectorized)
            with torch.no_grad():
                f_eval = self.predict_mean_f_on(X_eval, W, c, chunk_P=200_000, chunk_B=None)

                if M_comp > 0:
                    m_vec = (C_eval * f_eval.view(-1, 1)).mean(dim=0)  # (M,)
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

            # Logging
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
            traj["mean_Sigma"].append(float(Sigma.mean().item()))
            traj["mean_abs_Jr"].append(float(Jr.abs().mean().item()))
            traj["kappa2N2A"].append(float(kappa2N2A))
            traj["mean_c"].append(float(c.mean().item()))
            traj["chi_AA_vec"].append(chi_AA_list)
            traj["chi_AA_norm1"].append(chi_AA_norm1)
            traj["chi_AA_norm2"].append(chi_AA_norm2)
            traj["fp_drift"].append(drift)

            if it % self.sol.print_every == 1 or it == outer_steps:
                m_preview = mS_vec[:8] if len(mS_vec) > 0 else []
                chi_preview = chi_AA_list[:8] if len(chi_AA_list) > 0 else []
                print(json.dumps({
                    "iter": it,
                    "fp_drift": drift,
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
                    "mean_Sigma": float(Sigma.mean().item()),
                    "mean_abs_Jr": float(Jr.abs().mean().item()),
                    "kappa2N2A": float(kappa2N2A),
                    "mean_c": float(c.mean().item()),
                    "elapsed_s": round(traj["time_s"][-1], 2),
                    "B": self.mcmc.B, "P_train": P_train, "P_eval": X_eval.shape[0],
                    "num_components": M_comp
                }))

            # ---- Early stopping: small drift + stable eval ----
            if it > 50:
                recent = traj["eval_mse_total"][-20:]
                if len(recent) >= 5:
                    rel_band = (max(recent) - min(recent)) / max(1e-8, float(np.mean(recent)))
                    if drift < 1e-3 and rel_band < 1e-3:
                        break

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
                "teacher_sets": [s.cpu().tolist() for s in teacher_sets],
                "mS_last": traj["eval_mS_vec"][-1],
                "mS_norm2_last": traj["eval_mS_norm2"][-1],
                "chi_AA_last": traj["chi_AA_vec"][-1],
                "chi_AA_norm2_last": traj["chi_AA_norm2"][-1],
                "noise_norm2_last": traj["eval_noise_norm2"][-1],
                "mse_total_last": traj["eval_mse_total"][-1],
                "R2_last": traj["eval_R2"][-1],
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

# ----------------------------- Data generation -----------------------------
def generate_dense_parity_data(P: int, d: int, S_idx: torch.Tensor, device) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(0)
    X = (torch.randint(0, 2, (P, d), generator=g, device=device, dtype=torch.int8).float() * 2.0 - 1.0)
    y = parity_labels(X, S_idx)
    return X, y

def generate_composite_data(P: int, d: int, sets: List[torch.Tensor], device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(0)
    X = (torch.randint(0, 2, (P, d), generator=g, device=device, dtype=torch.int8).float() * 2.0 - 1.0)
    if len(sets) == 0:
        y = torch.zeros(P, device=device, dtype=torch.float32)
        C = torch.zeros(P, 0, device=device, dtype=torch.float32)
        return X, y, C
    C = compute_characters_matrix(X, sets)  # (P, M)
    y = C.sum(dim=1)
    return X, y, C

# ----------------------------- Entrypoint -----------------------------
if __name__ == "__main__":
    set_seed(42)
    _ = check_gpu()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---- Teacher/data ----
    d = 35  # (picked to match your directory label d35; change if desired)
    teacher_spec = "{0,1,2,3}"
    sets_idx_lists = parse_composite_spec(teacher_spec)
    teacher_sets = [torch.tensor(s, device=device, dtype=torch.long) for s in sets_idx_lists]
    M_components = len(teacher_sets)

    # Fixed eval set
    P_eval = 50_000
    X_eval,  y_eval,  _ = generate_composite_data(P_eval, d, teacher_sets, device)

    # ---- Model/MCMC/Solver knobs ----
    N = 512
    GAMMA_LIST = [0.5, 1.0]  # extend as needed

    mcmc = MCMCParams(
        B=16_384, steps=400, step_size=5e4, step_decay=0.999,
        grad_clip=10.0, clamp_w=None, autocast=True,
        warmup_steps=100, warmup_start_frac=0.01,
        use_mala=False, target_accept=0.574, da_t0=10.0, da_kappa=0.75
    )
    sol  = SolveParams(
        outer_steps=2000, saem_a0=0.2, saem_t0=100.0, saem_damping=3.0,
        print_every=10
    )

    # ---- Sweep settings ----
    P_TRAIN_LIST = [5000]#[10, 100, 500, 750, 1000, 2500, 5000, 7500, 10000, 20000]
    # NOTE: fixed obvious typo '7-5e-2' -> '7.5e-2'
    KAPPA_LIST   = [0.05]#[5e-4, 7.5e-3, 2.5e-2, 7.5e-2, 1e-2, 1e-1, 1e-3, 5e-3, 7.5e-2, 2.5e-2, 5e-2]

    results_dir = "/home/goring/mean_field_langevin/MCMC_finiteP/resutls_mf1/d35_k4_biggrid"
    os.makedirs(results_dir, exist_ok=True)

    all_runs: List[Dict] = []
    for gamma in GAMMA_LIST:
        for P_train in P_TRAIN_LIST:
            # Fresh train set
            X_train, y_train, _ = generate_composite_data(P_train, d, teacher_sets, device)

            for kappa in KAPPA_LIST:
                print(f"\n=== RUN: P_train={P_train}, kappa={kappa:.3e}, gamma={gamma} ===")

                mdl  = ModelParams(d=d, N=N, k=0, sigma_a=1.0, sigma_w=1.0, gamma=gamma, act="relu")
                solver = FunctionalCavitySolver(mdl, mcmc, sol, kappa=kappa, device=device, teacher_sets=teacher_sets)

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
                        f"_M{M_components}.json"
                    ),
                    "summary": out["summary"]
                })

    index_path = os.path.join(results_dir, "sweep_index.json")
    with open(index_path, "w") as f:
        json.dump({"runs": all_runs}, f, indent=2)
    print(f"\n[sweep saved] index: {index_path}")
