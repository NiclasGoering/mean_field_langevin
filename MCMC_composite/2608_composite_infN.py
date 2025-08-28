# cavity_functional_selfconsistency_gamma_composite.py
# Fully corrected implementation matching the final formulas you provided.
# Updates in this version:
#   • Gamma sweep: iterate over GAMMA_LIST (nested with P_TRAIN_LIST, KAPPA_LIST).
#   • Each run’s JSON and filename include gamma (N and gamma are in the name).
#   • Run tag includes P, kappa, and gamma for uniqueness and clarity.
#
# Key fixes retained from your previous version:
#   (1) Mean prediction ⟨f(x)⟩ = N * E_b[c(w_b) φ(w_b^T x)]   [c already includes N^{-γ}]
#   (2) Susceptibility scale: χ_AA = (N/κ^2) E_b[ E[a^2|w_b] * J_A(w_b)^2 ]
#   (3) All "1/(2 σ_a^2)" factors implemented in α, and consistent β, μ, σ^2
#   (4) Prior over w uses variance σ_w^2 / d  ⇒ S_prior(w) = d ||w||^2 / (2 σ_w^2)
#   (5) Comments at every formula implementation

import os, json, time, math, random, re
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import numpy as np
import torch
import torch._dynamo as dynamo
#torch.set_default_dtype(torch.float64)

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

def parity_character(X_pm1: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """
    χ_S(x) = ∏_{i in S} x_i  for x_i in {±1}.
    X_pm1: (P, d) with entries ±1
    S:     (k,)
    Returns: (P,) with entries ±1
    """
    if S.numel() == 0:
        # Empty set character is constant 1
        return torch.ones(X_pm1.shape[0], device=X_pm1.device, dtype=torch.float32)
    feats = X_pm1[:, S]             # (P, k)
    y = feats.prod(dim=1)           # (P,)
    return y.to(torch.float32)

def parity_labels(X_pm1: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    # legacy single parity label (±1)
    return parity_character(X_pm1, S)

def parse_composite_spec(spec: str) -> List[List[int]]:
    """
    Parse strings like:
      "{0,1,2,3}+{0,1,2,3,4,5,6,7}"
      " { 1 , 3 } + { 2 , 4 ,5 } "
    into a list of integer lists [[0,1,2,3], [0,1,2,3,4,5,6,7]]
    """
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
    """
    Return C of shape (P, M) where C[:, j] = χ_{S_j}(x)
    """
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
    sigma_a: float = 1.0            # INTERPRETATION: standard deviation of a (σ_a)
    sigma_w: float = 1.0            # INTERPRETATION: standard deviation of ||w|| scaled by sqrt(d) (σ_w)
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
    autocast: Optional[bool] = False  # None -> auto (enabled on CUDA)

@dataclass
class SolveParams:
    outer_steps: int = 500
    saem_a0: float = 1.0
    saem_t0: float = 20.0
    saem_damping: float = 1.0
    print_every: int = 10

# ----------------------------- SGLD over w (integrate out a) -----------------------------
@dynamo.disable
def sgld_sample_w(
    w: torch.Tensor,        # (B,d)
    X: torch.Tensor,        # (P,d), entries ±1
    y: torch.Tensor,        # (P,), real-valued (composite OK)
    f_mean: torch.Tensor,   # (P,), current mean function on train set
    kappa: float,
    mdl: ModelParams,
    mcmc: MCMCParams,
):
    """
    Single-neuron posterior with 'a' analytically integrated for:
        f(x) = N^{-γ} * Σ_i a_i * φ(w_i^T x),  with prior a_i ~ N(0, σ_a^2).
    We use the population/empirical approximations:
        Σ(w)   := E_x[ φ(w^T x)^2 ]           (empirical mean on train)
        JY(w)  := E_x[ φ(w^T x) * y(x) ]
        Jm(w)  := E_x[ φ(w^T x) * ⟨f⟩(x) ]
        Jr     := JY - Jm

    FORMULAS (your final corrected ones):
      α(w)  = 1/(2 σ_a^2) + Σ(w) / (κ^2 N^{2γ})                  [Eq. α]
      β(w)  = Jr / (κ^2 N^{γ})                                   [Eq. β]
      μ(w)  = β/α                                                [posterior mean of a | w]
      σ^2(w)= 1/α                                                [posterior var of a | w]

      S_eff(w) = (d / (2 σ_w^2)) ||w||^2 + 0.5 * log α - 0.5 * β^2 / α   [+ const]
      c(w)  = μ / N^{γ} = Jr / (κ^2 N^{2γ} α) = Jr / D_oracle
      D_oracle(w) = κ^2 N^{2γ} α = κ^2 N^{2γ} * (1/(2 σ_a^2)) + Σ(w)
    """
    device = w.device
    P = X.shape[0]
    N = float(mdl.N)
    gamma = float(mdl.gamma)
    N2g = N ** (2.0 * gamma)

    # Prior over w: each coordinate has variance σ_w^2 / d  ⇒
    # S_prior(w) = d ||w||^2 / (2 σ_w^2)
    var_w_per_coord = (mdl.sigma_w ** 2) / mdl.d

    # Acoef implements 1/(2 σ_a^2)  [matches α's first term]
    Acoef = 1.0 / (2.0 * (mdl.sigma_a ** 2))

    stiff = (kappa**4) * (mdl.N ** (2.0*mdl.gamma))
    step = mcmc.step_size * max(stiff, 1e-12)
    #step = mcmc.step_size
    autocast_enabled = mcmc.autocast if (mcmc.autocast is not None) else w.is_cuda

    # pre-cast once
    y_f = y.view(-1).to(torch.float32)
    m_f = f_mean.view(-1).to(torch.float32)

    for _ in range(mcmc.steps):
        w = w.detach().requires_grad_(True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            z = X @ w.t().contiguous()         # (P,B)
            phi = activation(z, mdl.act)

        g = phi.to(torch.float32)
        Sigma = (g * g).mean(dim=0)            # Σ(w): (B,)
        JY    = (g.t() @ y_f) / float(P)       # (B,)
        Jm    = (g.t() @ m_f) / float(P)       # (B,)
        Jr    = JY - Jm                         # Jr(w) = JY - Jm

        # α(w) = 1/(2 σ_a^2) + Σ / (κ^2 N^{2γ})
        D_internal = (Acoef + Sigma / (kappa * kappa * N2g)).clamp_min(1e-12)   # α(w)

        # Prior term: d ||w||^2 / (2 σ_w^2)  == 0.5 * ||w||^2 / (σ_w^2 / d)
        prior   = 0.5 * (w * w).sum(dim=1) / var_w_per_coord

        # 0.5 * log α
        log_det = 0.5 * torch.log(D_internal)

        # - 0.5 * β^2 / α, where β^2 = Jr^2 / (κ^4 N^{2γ})
        data_quad = (-0.5) * (Jr * Jr) / ((kappa ** 4) * N2g * D_internal + 1e-30)

        # Total energy U(w) ≡ S_eff(w) up to additive const
        U = prior + log_det + data_quad

        # Langevin step on w
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

    # final recompute for outputs at the sampled w's
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        z = X @ w.t().contiguous()
        phi = activation(z, mdl.act).detach()
    g = phi.to(torch.float32)
    Sigma = (g * g).mean(dim=0)
    JY    = (g.t() @ y_f) / float(P)
    Jm    = (g.t() @ m_f) / float(P)
    Jr    = JY - Jm

    # D_oracle = κ^2 N^{2γ} α  = κ^2 N^{2γ} * (1/(2 σ_a^2)) + Σ
    D_oracle = ((kappa ** 2) * N2g * 1.0 / (2.0 * (mdl.sigma_a ** 2)) + Sigma).clamp_min(1e-12)

    return (
        w.detach(),            # sampled weights
        phi.detach(),          # φ on train for each chain
        D_oracle.detach(),     # D_oracle(w_b) for each chain b
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
                          chunk_B: int = 1024) -> torch.Tensor:
        """
        Compute ⟨f(x)⟩ using:
            ⟨f(x)⟩ = N * E_b[ c_b * φ(w_b^T x) ]
        where c_b = μ(w_b) / N^{γ} and μ(w_b) = E[a | w_b].
        NOTE: The factor N here is CRUCIAL (the previous bug was missing this).
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
            # E_b[c_b φ_b(x)] accumulated chunkwise
            f += (gb * cb).mean(dim=1) * (stop - start) / B
        # Multiply by N (because c includes N^{-γ})  ⇒ ⟨f⟩ = N * E_b[c φ]
        return self.mdl.N * f

    @torch.no_grad()
    def project_onto_teacher_span(self, X: torch.Tensor, f: torch.Tensor, sets: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given f(x) on dataset X and teacher character sets S_j, compute:
          - m_vec: (M,) with m_j = E[f * χ_{S_j}]
          - f_signal: (P,) the projection ∑_j m_j χ_{S_j}(x)
          - r: residual f - f_signal
        """
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
        phi_train: torch.Tensor,     # (P, B)   φ on train for each chain
        c: torch.Tensor,             # (B,)     c_b = μ_b / N^{γ}
        D_oracle: torch.Tensor,      # (B,)     D_oracle_b = κ^2 N^{2γ} α_b
        C_train: torch.Tensor,       # (P, M)   characters χ_S on train
        chunk_B: int = 2048
    ) -> torch.Tensor:
        """
        Compute per-mode susceptibility vector χ_AA of length M using your formula:
            χ_AA[A] = (N / κ^2) * E_b[ E[a^2 | w_b] * J_A(w_b)^2 ]
        with:
            μ_b     = N^{γ} * c_b                                        [since c_b = μ_b / N^{γ}]
            σ_b^2   = 1/α_b = (κ^2 N^{2γ}) / D_oracle_b
            E[a^2]  = μ_b^2 + σ_b^2 = N^{2γ} * ( c_b^2 + κ^2 / D_oracle_b )

        J_A(w_b) is estimated empirically on train:
            J_A(w_b) = E_x[ φ(w_b^T x) χ_A(x) ]
        """
        if C_train.numel() == 0:
            return torch.zeros(0, device=phi_train.device, dtype=torch.float32)

        P, B = phi_train.shape
        M = C_train.shape[1]
        device = phi_train.device

        N = float(self.mdl.N)
        gamma = float(self.mdl.gamma)
        N2g = N ** (2.0 * gamma)

        # E[a^2 | w_b] = N^{2γ} * ( c_b^2 + κ^2 / D_oracle_b )
        a2 = N2g * (c.to(torch.float32) ** 2 + (self.kappa ** 2) / D_oracle.to(torch.float32))  # (B,)

        accum = torch.zeros(M, device=device, dtype=torch.float32)
        Ct = C_train.to(torch.float32).t()          # (M, P)

        for start in range(0, B, chunk_B):
            stop = min(start + chunk_B, B)
            g_chunk = phi_train[:, start:stop].to(torch.float32)    # (P, Bb)
            # J_A chunk: (M, Bb), empirical J_A(w_b) = E_x[φ χ_A]
            J_chunk = (Ct @ g_chunk) / float(P)
            term = (a2[start:stop].view(1, -1) * (J_chunk * J_chunk))  # (M, Bb)
            accum += term.sum(dim=1)

        # χ_AA = (N / κ^2) * mean_b[ E[a^2] * J_A^2 ]
        chi_AA = (N / (self.kappa ** 2)) * (accum / float(B))
        return chi_AA

    def run(self, X_train: torch.Tensor, y_train: torch.Tensor,
            X_eval: torch.Tensor, y_eval: torch.Tensor,
            outer_steps: Optional[int] = None,
            log_dir: str = "./results_cavity_gamma", run_tag: str = "") -> Dict:

        os.makedirs(log_dir, exist_ok=True)
        if outer_steps is None: outer_steps = self.sol.outer_steps

        P_train, d = X_train.shape

        # init mean function on train set: ⟨f⟩(x) ≈ 0 initially
        f_mean = torch.zeros(P_train, device=self.device, dtype=torch.float32)

        # init SGLD chains for w ~ N(0, σ_w^2 I / d) ⇒ std = σ_w / sqrt(d)
        W = torch.randn(self.mcmc.B, d, device=self.device) * (self.mdl.sigma_w / math.sqrt(self.mdl.d))

        # Precompute teacher characters
        teacher_sets = self.teacher_sets
        M_comp = len(teacher_sets)
        C_train = compute_characters_matrix(X_train, teacher_sets) if M_comp > 0 else torch.zeros(P_train, 0, device=self.device)
        C_eval  = compute_characters_matrix(X_eval,  teacher_sets) if M_comp > 0 else torch.zeros(X_eval.shape[0], 0, device=self.device)
        true_coeff = torch.ones(M_comp, device=self.device, dtype=torch.float32)  # y = Σ χ_S ⇒ coeffs = 1

        traj = {
            "iter": [], "time_s": [],
            "train_mse": [], "train_corr": [], "train_y2": [], "train_f2": [],
            "eval_mS_vec": [], "eval_mS_norm2": [],
            "eval_coeff_mse": [], "eval_coeff_mae": [], "eval_coeff_sign_acc": [],
            "eval_noise_norm2": [], "eval_mse_total": [], "eval_corr_yf": [], "eval_R2": [],
            "mean_Sigma": [], "mean_abs_Jr": [], "kappa2N2A": [], "mean_c": [],
            "chi_AA_vec": [], "chi_AA_norm1": [], "chi_AA_norm2": []
        }
        t0 = time.time()

        # For logging: κ^2 N^{2γ} * (1/(2 σ_a^2))
        Acoef = 1.0 / (2.0 * (self.mdl.sigma_a ** 2))
        kappa2N2A = (self.kappa ** 2) * (self.mdl.N ** (2.0 * self.mdl.gamma)) * Acoef

        for it in range(1, outer_steps + 1):
            # --- E-step: sample w from single-neuron posterior (given current f_mean) ---
            W, phi_train, D_or, JY, Jm, Sigma = sgld_sample_w(
                W, X_train, y_train, f_mean, self.kappa, self.mdl, self.mcmc
            )
            Jr = (JY - Jm).to(torch.float32)

            # c_b = μ / N^{γ} = Jr / (κ^2 N^{2γ} α) = Jr / D_oracle
            c = (Jr / D_or).to(torch.float32)           # (B,)

            # --- Build new mean function on training set ---
            # E_b[c_b φ_b(x)] over chains, then multiply by N   [⟨f⟩ = N * E_b[c φ]]
            f_chunk_mean = (phi_train.to(torch.float32) * c.view(1, -1)).mean(dim=1)
            f_new_train = self.mdl.N * f_chunk_mean

            # SAEM-style damping for stability
            a_t = self.sol.saem_a0 / (it + self.sol.saem_t0)
            f_mean = (1 - self.sol.saem_damping * a_t) * f_mean + self.sol.saem_damping * a_t * f_new_train

            # training diagnostics
            train_mse = float(((y_train - f_mean) ** 2).mean().item())
            train_corr = float((y_train * f_mean).mean().item())
            train_y2 = float((y_train * y_train).mean().item())
            train_f2 = float((f_mean * f_mean).mean().item())

            # --- Per-mode susceptibility on TRAIN (empirical) ---
            if M_comp > 0:
                chi_AA = self.compute_susceptibility_AA(phi_train, c, D_or, C_train, chunk_B=2048)  # (M,)
                chi_AA_list = chi_AA.detach().cpu().tolist()
                chi_AA_norm1 = float(chi_AA.abs().sum().item())
                chi_AA_norm2 = float((chi_AA * chi_AA).sum().sqrt().item())
            else:
                chi_AA_list, chi_AA_norm1, chi_AA_norm2 = [], 0.0, 0.0

            # --- Unbiased evaluation on large held-out set ---
            with torch.no_grad():
                # Predict mean on eval with the CORRECT N multiplier
                f_eval = self.predict_mean_f_on(X_eval, W, c, chunk_B=1024*1024)

                if M_comp > 0:
                    # m_j = E[f * χ_j] computed empirically on eval set
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

            # --- Logging ---
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
                    "mean_Sigma": float(Sigma.mean().item()),
                    "mean_abs_Jr": float(Jr.abs().mean().item()),
                    "kappa2N2A": float(kappa2N2A),
                    "mean_c": float(c.mean().item()),
                    "elapsed_s": round(traj["time_s"][-1], 2),
                    "B": self.mcmc.B, "P_train": P_train, "P_eval": X_eval.shape[0],
                    "num_components": M_comp
                }))

        # final snapshot
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
        # Filename-safe gamma string (e.g., g0p5 for 0.5, gm0p1 for -0.1)
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
    """
    Legacy: single parity dataset (labels ±1).
    """
    g = torch.Generator(device=device).manual_seed(0)
    X = (torch.randint(0, 2, (P, d), generator=g, device=device, dtype=torch.int8).float() * 2.0 - 1.0)
    y = parity_labels(X, S_idx)
    return X, y

def generate_composite_data(P: int, d: int, sets: List[torch.Tensor], device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Composite teacher: y(x) = sum_j χ_{S_j}(x)
    Returns:
      X: (P, d) in {±1}
      y: (P,) real-valued
      C: (P, M) characters matrix with C[:, j] = χ_{S_j}(x)
    """
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
    d = 35
    # Example composite spec: A={0,1}, B={0,1,2,3}
    teacher_spec = "{0,1,2,3}"
    sets_idx_lists = parse_composite_spec(teacher_spec)
    teacher_sets = [torch.tensor(s, device=device, dtype=torch.long) for s in sets_idx_lists]
    M_components = len(teacher_sets)

    # Fixed eval set (reused for all runs)
    P_eval = 20000
    X_eval,  y_eval,  _ = generate_composite_data(P_eval, d, teacher_sets, device)

    # ---- Model/MCMC/Solver knobs ----
    N = 512

    # Make gamma iterable — sweep over these values
    GAMMA_LIST = [0.5,1.0]  # e.g., [0.0, 0.25, 0.5, 0.75, 1.0]

    mcmc = MCMCParams(
        B=8192, steps=400, step_size=800, step_decay=0.999,
        grad_clip=1e8, clamp_w=20.0, autocast=True
    )
    sol  = SolveParams(
        outer_steps=1000, saem_a0=0.2, saem_t0=100.0, saem_damping=3.0,
        print_every=10
    )

    P_TRAIN_LIST = [5000] #[10, 100, 500, 750, 1000, 2500, 5000, 7500, 10000, 20000]
    # NOTE: fixed obvious typo '7-5e-2' -> '7.5e-2'
    KAPPA_LIST   = [5e-4, 7.5e-3, 2.5e-2, 7.5e-2, 1e-2, 1e-1, 1e-3, 5e-3, 7.5e-2, 2.5e-2, 5e-2]

    results_dir = "/home/goring/mean_field_langevin/MCMC_finiteP/resutls_mf1/d35_k4_biggrid"
    os.makedirs(results_dir, exist_ok=True)

    all_runs: List[Dict] = []
    for gamma in GAMMA_LIST:
        for P_train in P_TRAIN_LIST:
            # Generate a fresh train set for this P_train
            X_train, y_train, _ = generate_composite_data(P_train, d, teacher_sets, device)

            for kappa in KAPPA_LIST:
                print(f"\n=== RUN: P_train={P_train}, kappa={kappa:.3e}, gamma={gamma} ===")

                # Build model params PER gamma (important when sweeping)
                mdl  = ModelParams(d=d, N=N, k=0, sigma_a=1.0, sigma_w=1.0, gamma=gamma, act="relu")

                solver = FunctionalCavitySolver(mdl, mcmc, sol, kappa=kappa, device=device, teacher_sets=teacher_sets)

                # filename/run tag: include P, kappa, gamma (gamma sanitized for readability)
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

    # Optional: write an index file summarizing the sweep
    index_path = os.path.join(results_dir, "sweep_index.json")
    with open(index_path, "w") as f:
        json.dump({"runs": all_runs}, f, indent=2)
    print(f"\n[sweep saved] index: {index_path}")
