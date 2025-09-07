# basis_free_cavity_cg_fixedpoint_eval.py
# Basis-free functional MF with exact (I+K)^{-1} fixed-point solve for ⟨f⟩,
# optional LOO Onsager reaction, and large held-out evaluation of m_S and
# half-MSE decomposition (modes + noise) computed every print iteration.
#
# Conventions:
#   alpha(w) = 1/(2 σ_a^2) + Σ_eff(w)/(2 κ^2 N^{2γ})
#   beta(w)  = Jr(w)/(κ^2 N^γ),  Jr = E_x[ φ(w^T x) y ] - E_x[ φ(w^T x) ⟨f(x)⟩ ]
#   mu(w)    = beta(w) / (2 alpha(w))           (posterior mean of a | w)
#   Var[a|w] = 1 / (2 alpha(w))                 (posterior variance)
#
# Potential (integrated out a):
#   U(w) = (d/(2 σ_w^2))||w||^2 + (1/2) ln alpha(w) - beta(w)^2 / (4 alpha(w))
#
# MF fixed point (basis-free on points):
#   A_b = 1/(2 α_b κ^2 N^γ),  K = (N^{1-γ}/(B P)) * Φ diag(A) Φ^T
#   (I + K) f = K y     (solved via CG each outer step)
#
# Reaction (optional, quenched, LOO, connected):
#   C = (1/P) Φ^T Φ,  a2 = μ^2 + Var[a|w]
#   R_b/N = (1/κ^2) * [ (1/(B-1)) * sum_{j≠b} a2_j C_{bj}^2
#                        - ( (1/(B-1)) * sum_{j≠b} μ_j C_{bj} )^2 ]
#   Σ_eff,b = Σ_b - (R_b/N)
#
# Held-out evaluation (every print):
#   Generate X_eval ∈ {±1}^d of size P_eval, y_eval = Σ_S χ_S(x)
#   f_eval(x) = N^{1-γ} * (1/B) Σ_b μ_b φ(w_b^T x)
#   m_S = E_eval[ f_eval χ_S ]
#   noise = E_eval[ f_eval^2 ] - 2 m^T m + m^T G m,  G = E_eval[ χ χ^T ]
#   half_mse_from_modes = 0.5 * Σ_S (1 - m_S)^2
#   half_noise          = 0.5 * noise
#   half_mse_total_ms   = half_mse_from_modes + half_noise
#   half_mse_empirical  = 0.5 * ( E[f^2] - 2 Σ_S m_S + 1^T G 1 )

import os, json, time, math, random
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import numpy as np
import torch

# ----------------------------- utils -----------------------------

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
    import re
    sets = re.findall(r"\{([^}]*)\}", spec)
    out: List[List[int]] = []
    for s in sets:
        elems = [int(tok.strip()) for tok in s.split(",") if tok.strip() != ""]
        out.append(sorted(elems))
    if len(out) == 0:
        raise ValueError(f"Failed to parse composite spec: {spec!r}")
    return out

def parity_character(X_pm1: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    # χ_S(x) = ∏_{i in S} x_i, with x_i ∈ {±1}
    if S.numel() == 0:
        return torch.ones(X_pm1.shape[0], device=X_pm1.device, dtype=torch.float32)
    return X_pm1[:, S].prod(dim=1).to(torch.float32)

def generate_parity_data(P: int, d: int, sets: List[torch.Tensor], device) -> Tuple[torch.Tensor, torch.Tensor]:
    # X ∈ {±1}^d, y = ∑_S χ_S(x) for requested teacher sets (e.g., {0,1,2,3})
    g = torch.Generator(device=device).manual_seed(0)
    X = (torch.randint(0, 2, (P, d), generator=g, device=device, dtype=torch.int8).float() * 2.0 - 1.0)
    if len(sets) == 0:
        y = torch.zeros(P, device=device, dtype=torch.float32)
    else:
        C = torch.stack([parity_character(X, S) for S in sets], dim=1)  # (P, M)
        y = C.sum(dim=1)
    return X, y

# ----------------------------- config -----------------------------

@dataclass
class ModelParams:
    d: int = 25
    N: int = 1024
    gamma: float = 0.5
    sigma_a: float = 1.0
    sigma_w: float = 1.0
    act: str = "relu"

@dataclass
class MCMCParams:
    B: int = 8192
    step_size: float = 2e-5
    grad_clip: float = 5.0
    clamp_w: float = 0.0
    langevin_sqrt2: bool = True

@dataclass
class SolveParams:
    outer_steps: int = 500
    print_every: int = 10
    use_reaction: bool = True
    # CG solve for (I+K) f = K y
    cg_tol: float = 1e-6
    cg_maxit: int = 200
    ridge: float = 1e-12
    # Held-out evaluation
    P_eval: int = 200_000          # big held-out set for H100
    eval_chunk: int = 8192         # stream over eval points in chunks

# ------------------------- core helpers ---------------------------

def compute_phi(X: torch.Tensor, W: torch.Tensor, act: str) -> torch.Tensor:
    # Φ_{μb} = φ(w_b^T x_μ)
    z = X @ W.t().contiguous()
    return activation(z, act).to(torch.float32)

def train_stats(phi: torch.Tensor, y: torch.Tensor, f_mean_detached: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Σ_b = E_x[φ_b^2], JY_b = E_x[φ_b y], Jm_b = E_x[φ_b ⟨f⟩], Jr = JY - Jm
    P = phi.shape[0]
    g = phi
    Sigma = (g * g).mean(dim=0)                                       # (B,)
    JY    = (g.t() @ y.view(-1)) / float(P)                           # (B,)
    Jm    = (g.t() @ f_mean_detached.view(-1)) / float(P)             # (B,)
    Jr    = JY - Jm                                                    # (B,)
    return Sigma, JY, Jm, Jr

def alpha_beta_mu_vara(Sigma_eff: torch.Tensor, Jr: torch.Tensor, mdl: ModelParams, kappa: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    alpha = 1/(2 σ_a^2) + Σ_eff/(2 κ^2 N^{2γ})
    beta  = Jr/(κ^2 N^γ)
    mu    = beta/(2 alpha)
    var_a = 1/(2 alpha)
    """
    N2g = (float(mdl.N) ** (2.0 * float(mdl.gamma)))
    alpha = (1.0 / (2.0 * (mdl.sigma_a ** 2))) + (Sigma_eff / (2.0 * (kappa ** 2) * N2g))
    beta  = Jr / ( (kappa ** 2) * (float(mdl.N) ** float(mdl.gamma)) )
    mu    = beta / (2.0 * alpha)
    var_a = 1.0 / (2.0 * alpha)
    return alpha.clamp_min(1e-30), beta, mu, var_a

def potential_U(W: torch.Tensor, Sigma_eff: torch.Tensor, Jr: torch.Tensor, mdl: ModelParams, kappa: float) -> torch.Tensor:
    """
    U(w) = (d/(2 σ_w^2))||w||^2 + (1/2) ln alpha - beta^2/(4 alpha)
    """
    var_w_per_coord = (mdl.sigma_w ** 2) / mdl.d
    prior = 0.5 * (W * W).sum(dim=1) / var_w_per_coord
    alpha, beta, _, _ = alpha_beta_mu_vara(Sigma_eff, Jr, mdl, kappa)
    log_det = 0.5 * torch.log(alpha)
    data_quad = - (beta * beta) / (4.0 * alpha)
    return prior + log_det + data_quad

@torch.no_grad()
def onsager_reaction_sameP_loo(phi: torch.Tensor, mu: torch.Tensor, var_a: torch.Tensor, kappa: float) -> torch.Tensor:
    """
    LOO reaction (quenched, connected):
      C = (1/P) Φ^T Φ,  a2 = μ^2 + var_a.
      R_b/N = (1/κ^2) * [ (1/(B-1)) Σ_{j≠b} a2_j C_{bj}^2 - ( (1/(B-1)) Σ_{j≠b} μ_j C_{bj} )^2 ].
    """
    P, B = phi.shape
    C = (phi.t().matmul(phi)) / float(P)          # (B,B)
    a2 = (mu * mu) + var_a                        # (B,)

    sum_a2C2 = (C * C) @ a2                       # (B,)
    sum_muC  = C @ mu                             # (B,)

    diagC    = C.diag()                           # (B,)
    self_a2C2 = a2 * (diagC * diagC)              # a2_b * C_{bb}^2
    self_muC  = mu * diagC                        # μ_b * C_{bb}

    denom = max(B - 1, 1)
    term_var = (sum_a2C2 - self_a2C2) / denom
    mean_vec = (sum_muC  - self_muC ) / denom

    R_divN = (term_var - mean_vec * mean_vec) / (kappa * kappa)
    return torch.clamp(R_divN, min=0.0)

# ----- CG solver for (I + K) f = rhs with K as an implicit operator -----

class KOperator:
    """
    K v = (N^{1-γ}/(B P)) * Φ * ( A ⊙ (Φ^T v) )
    Shapes: Φ: (P,B), A: (B,)
    """
    def __init__(self, Phi: torch.Tensor, A: torch.Tensor, N: int, gamma: float):
        self.Phi = Phi.to(torch.float32)     # (P,B)
        self.A   = A.to(torch.float32)       # (B,)
        self.P   = float(Phi.shape[0])
        self.B   = float(Phi.shape[1])
        self.scale = (float(N) ** (1.0 - float(gamma))) / (self.B * self.P)

    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        # t = Φ^T v  (B,),  t = A ⊙ t,  u = Φ t  (P,),  scale at the end
        t = self.Phi.t().matmul(v)            # (B,)
        t = t * self.A                         # (B,)
        u = self.Phi.matmul(t)                 # (P,)
        return self.scale * u

def cg_solve_I_plus_K(Kop: KOperator, rhs: torch.Tensor, tol: float = 1e-6, maxit: int = 200, ridge: float = 0.0) -> torch.Tensor:
    """
    Solve (I + K) f = rhs with CG using only K.matvec.
    Optionally add a tiny 'ridge' on the diagonal: (I + ridge*I + K).
    """
    x = torch.zeros_like(rhs)
    def apply_M(v):  # (I + K + ridge I) v
        out = v + Kop.matvec(v)
        if ridge > 0:
            out = out + ridge * v
        return out

    r = rhs - apply_M(x)
    p = r.clone()
    rsold = (r * r).sum()
    eps = torch.tensor(tol, device=rhs.device, dtype=rhs.dtype)
    for _ in range(maxit):
        Ap = apply_M(p)
        denom = (p * Ap).sum().clamp_min(1e-30)
        alpha = rsold / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = (r * r).sum()
        if rsnew.sqrt() < eps:
            break
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew
    return x

# ----------------------------- solver -----------------------------

class BasisFreeCavitySolver:
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
    def _eval_big_heldout(self, W: torch.Tensor, mu: torch.Tensor) -> Dict[str, object]:
        """
        Batched evaluation on a large held-out set (size P_eval),
        computing m_S, noise, and half-MSE decomposition without storing f_eval.
        """
        d = self.mdl.d
        P_eval = self.sol.P_eval
        bs = self.sol.eval_chunk
        device = self.device
        M = len(self.teacher_sets)

        # Build held-out X_eval (±1) once per call; compute sufficient statistics in a stream.
        g = torch.Generator(device=device).manual_seed(12345)
        # chunked loop: generate chunk of X_eval on-the-fly to avoid a huge full matrix in memory
        sum_f2 = torch.zeros(1, device=device, dtype=torch.float32)
        sum_Ct_f = torch.zeros(M, device=device, dtype=torch.float32) if M > 0 \
                   else torch.zeros(0, device=device, dtype=torch.float32)
        sum_G = torch.zeros(M, M, device=device, dtype=torch.float32) if M > 0 \
                else torch.zeros(0, 0, device=device, dtype=torch.float32)

        N1mg = float(self.mdl.N) ** (1.0 - float(self.mdl.gamma))
        B = W.shape[0]
        mu_row = mu.view(1, -1)  # (1,B)

        for start in range(0, P_eval, bs):
            n = min(bs, P_eval - start)
            # sample X_chunk ∈ {±1}^{n×d}
            Xc = (torch.randint(0, 2, (n, d), generator=g, device=device, dtype=torch.int8).float() * 2.0 - 1.0)

            # f_chunk = N^{1-γ} * mean_b [ μ_b φ(w_b^T x) ]
            z = Xc @ W.t().contiguous()                        # (n,B)
            phi_c = activation(z, self.mdl.act).to(torch.float32)
            f_chunk = N1mg * (phi_c * mu_row).mean(dim=1)      # (n,)

            # parity characters on this chunk (if any)
            if M > 0:
                C_cols = []
                for S in self.teacher_sets:
                    C_cols.append(parity_character(Xc, S))
                Cc = torch.stack(C_cols, dim=1)                # (n,M)

                # accumulate sufficient statistics
                sum_Ct_f += Cc.t().matmul(f_chunk)             # (M,)
                sum_G    += Cc.t().matmul(Cc)                  # (M,M)

            sum_f2 += (f_chunk * f_chunk).sum()

            # free chunk tensors
            del Xc, z, phi_c, f_chunk
            if M > 0:
                del Cc

        inv_P = 1.0 / float(P_eval)
        f2_bar = (sum_f2 * inv_P).item()

        if M == 0:
            return {
                "eval_m_S": [],
                "half_mse_modes": 0.0,
                "half_noise": 0.0,
                "half_mse_total_ms": 0.5 * f2_bar,   # y=0 case
                "half_mse_empirical": 0.5 * f2_bar
            }

        v = sum_Ct_f * inv_P                     # (M,)  ~ m_S (IP coefficients)
        G = sum_G * inv_P                         # (M,M)
        ones = torch.ones(M, device=device, dtype=torch.float32)

        # m_S (inner-product coefficients as requested)
        m_S = v

        # noise power using sample Gram:
        # noise = E[f^2] - 2 m^T m + m^T G m
        mTm = float((m_S * m_S).sum().item())
        mTGm = float(m_S.view(1, -1).matmul(G).matmul(m_S.view(-1, 1)).item())
        noise = f2_bar - 2.0 * mTm + mTGm

        # half-MSE via requested decomposition:
        half_mse_modes = 0.5 * float(((1.0 - m_S) ** 2).sum().item())
        half_noise = 0.5 * float(noise)
        half_mse_total_ms = half_mse_modes + half_noise

        # empirical half-MSE (cross-check) without forming y explicitly:
        # 0.5 * (E[f^2] - 2 E[f y] + E[y^2]), with E[f y] = 1^T v, E[y^2] = 1^T G 1
        Ef_y = float(ones.dot(v).item())
        Ey2 = float(ones.view(1, -1).matmul(G).matmul(ones.view(-1, 1)).item())
        half_mse_empirical = 0.5 * (f2_bar - 2.0 * Ef_y + Ey2)

        return {
            "eval_m_S": m_S.detach().cpu().tolist(),
            "half_mse_modes": half_mse_modes,
            "half_noise": half_noise,
            "half_mse_total_ms": half_mse_total_ms,
            "half_mse_empirical": half_mse_empirical
        }

    def run(self, X: torch.Tensor, y: torch.Tensor,
            log_dir: str = "./results_basis_free",
            run_tag: str = "") -> Dict:
        os.makedirs(log_dir, exist_ok=True)
        device = self.device

        P, d = X.shape
        B = self.mcmc.B

        # Initialize chains: w ~ N(0, σ_w^2/d I)
        W = torch.randn(B, d, device=device) * (self.mdl.sigma_w / math.sqrt(self.mdl.d))

        # Initialize mean field on points: ⟨f(x_μ)⟩
        f_mean = torch.zeros(P, device=device, dtype=torch.float32)

        # Initialize reaction memory (R/N per chain)
        R_divN_prev = torch.zeros(B, device=device, dtype=torch.float32)

        traj = {
            "iter": [], "train_mse": [], "mean_Sigma": [], "mean_R_divN": [],
            "mean_Sigma_eff": [], "min_Sigma_eff": [], "chi_normF": [],
            # held-out diagnostics
            "eval_m_S": [], "half_mse_modes": [], "half_noise": [],
            "half_mse_total_ms": [], "half_mse_empirical": []
        }
        t0 = time.time()

        for it in range(1, self.sol.outer_steps + 1):
            # --- make W a fresh leaf each outer iter ---
            W = W.detach().requires_grad_(True)

            # ---------- forward (current graph) ----------
            phi = compute_phi(X, W, self.mdl.act)                         # (P,B)
            Sigma, JY, Jm, Jr = train_stats(phi, y, f_mean.detach())      # f_mean detached!

            # Σ_eff = Σ - (R/N) if using reaction, else Σ
            Sigma_eff = Sigma - (R_divN_prev if self.sol.use_reaction else 0.0)
            #Sigma_eff = torch.clamp(Sigma_eff, min=1e-18) #1e-12

            # α, β, μ, Var[a] for CURRENT graph
            alpha, beta, mu, var_a = alpha_beta_mu_vara(Sigma_eff, Jr, self.mdl, self.kappa)

            # ---------- one SGLD step on U(w) ----------
            U = potential_U(W, Sigma_eff, Jr, self.mdl, self.kappa)
            loss = U.sum()
            loss.backward()
            with torch.no_grad():
                grad = W.grad
                if self.mcmc.grad_clip and self.mcmc.grad_clip > 0:
                    gn = grad.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    grad = grad * (self.mcmc.grad_clip / gn).clamp(max=1.0)
                step = self.mcmc.step_size
                noise = torch.randn_like(W)
                if self.mcmc.langevin_sqrt2:
                    W += (- step * grad) + noise * math.sqrt(2.0 * step)
                else:
                    W += (- 0.5 * step * grad) + noise * math.sqrt(step)
                if self.mcmc.clamp_w and self.mcmc.clamp_w > 0:
                    W.clamp_(-self.mcmc.clamp_w, self.mcmc.clamp_w)
                W.grad.zero_()

            # ---------- recompute environment & SOLVE f via CG ----------
            with torch.no_grad():
                phi_env = compute_phi(X, W, self.mdl.act)                 # (P,B)
                Sigma_env = (phi_env * phi_env).mean(dim=0)               # (B,)
                Sigma_eff_env = Sigma_env - (R_divN_prev if self.sol.use_reaction else 0.0)
                #Sigma_eff_env = torch.clamp(Sigma_eff_env, min=1e-12)

                # Jr under cavity (fixed ⟨f⟩)
                _, _, _, Jr_env = train_stats(phi_env, y, f_mean.detach())

                # α, μ, Var for environment (used for reaction & A)
                alpha_env, beta_env, mu_env, var_a_env = alpha_beta_mu_vara(Sigma_eff_env, Jr_env, self.mdl, self.kappa)

                # Build A_b = 1/(2 α_b κ^2 N^γ)
                A_env = (1.0 / (2.0 * alpha_env * (self.kappa ** 2) * (float(self.mdl.N) ** float(self.mdl.gamma)))).to(torch.float32)

                # Solve (I + K) f = K y   with K v = (N^{1-γ}/(B P)) Φ (A ⊙ (Φ^T v))
                Kop = KOperator(phi_env, A_env, self.mdl.N, self.mdl.gamma)
                rhs = Kop.matvec(y)                  # K y
                f_new = cg_solve_I_plus_K(Kop, rhs, tol=self.sol.cg_tol, maxit=self.sol.cg_maxit, ridge=self.sol.ridge)

                f_mean = f_new  # exact fixed point

                # Optional χ diagnostic (connected)
                a2_env = (mu_env * mu_env) + (1.0 / (2.0 * alpha_env))    # (B,)
                u = (phi_env * mu_env.view(1, -1)).mean(dim=1, keepdim=True)  # (P,1)
                term = (phi_env * a2_env.view(1, -1)) @ phi_env.t() / float(B)
                chi_mat = ((float(self.mdl.N) ** (2.0 - 2.0 * float(self.mdl.gamma))) / (self.kappa * self.kappa)) * (term - u @ u.t())
                chi_normF = float((chi_mat * chi_mat).sum().sqrt().item())

                # Onsager reaction (LOO) using env μ & Var
                if self.sol.use_reaction:
                    R_divN = onsager_reaction_sameP_loo(phi_env, mu_env, (1.0 / (2.0 * alpha_env)), self.kappa)
                    R_divN_prev = R_divN.detach()
                else:
                    R_divN_prev = torch.zeros_like(R_divN_prev)

            # ---------- diagnostics ----------
            with torch.no_grad():
                train_mse = float(((y - f_mean) ** 2).mean().item())

                # Held-out evaluation every print interval
                if (it % self.sol.print_every == 1) or (it == self.sol.outer_steps):
                    eval_stats = self._eval_big_heldout(W.detach(), mu_env.detach())
                    traj["eval_m_S"].append(eval_stats["eval_m_S"])
                    traj["half_mse_modes"].append(eval_stats["half_mse_modes"])
                    traj["half_noise"].append(eval_stats["half_noise"])
                    traj["half_mse_total_ms"].append(eval_stats["half_mse_total_ms"])
                    traj["half_mse_empirical"].append(eval_stats["half_mse_empirical"])

                    print(json.dumps({
                        "iter": it,
                        "train_mse": train_mse,
                        "mean_Sigma": float(Sigma_env.mean().item()),
                        "mean_R_divN": float(R_divN_prev.mean().item()),
                        "mean_Sigma_eff": float(Sigma_eff_env.mean().item()),
                        "min_Sigma_eff": float(Sigma_eff_env.min().item()),
                        "chi_normF": chi_normF,
                        "half_mse_modes": eval_stats["half_mse_modes"],
                        "half_noise": eval_stats["half_noise"],
                         "m_S": eval_stats["eval_m_S"],
                        "half_mse_total_ms": eval_stats["half_mse_total_ms"],
                        "half_mse_empirical": eval_stats["half_mse_empirical"],
                        "B": B, "P_train": P, "P_eval": self.sol.P_eval,
                        "elapsed_s": round(time.time() - t0, 2)
                    }))

                traj["iter"].append(it)
                traj["train_mse"].append(train_mse)
                traj["mean_Sigma"].append(float(Sigma_env.mean().item()))
                traj["mean_R_divN"].append(float(R_divN_prev.mean().item()))
                traj["mean_Sigma_eff"].append(float(Sigma_eff_env.mean().item()))
                traj["min_Sigma_eff"].append(float(Sigma_eff_env.min().item()))
                traj["chi_normF"].append(chi_normF)

        out = {
            "summary": {
                "P_train": int(P), "d": self.mdl.d, "N": self.mdl.N, "gamma": self.mdl.gamma,
                "kappa": self.kappa, "act": self.mdl.act, "sigma_a": self.mdl.sigma_a,
                "train_mse_last": traj["train_mse"][-1],
                "mean_R_divN_last": traj["mean_R_divN"][-1],
                "mean_Sigma_eff_last": traj["mean_Sigma_eff"][-1],
                "chi_normF_last": traj["chi_normF"][-1],
                "P_eval": int(self.sol.P_eval)
            },
            "traj": traj,
        }
        tag = run_tag or time.strftime("%Y%m%d_%H%M%S")
        gstr = f"{self.mdl.gamma:.6g}".replace('.', 'p').replace('-', 'm')
        path = os.path.join(
            log_dir,
            f"basis_free_cavity_cg_eval_{tag}_Ptr{P}_Peval{int(self.sol.P_eval)}"
            f"_kap{float(self.kappa):.3e}_N{int(self.mdl.N)}_g{gstr}.json"
        )
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {path}")
        return out

# ----------------------------- main -----------------------------

if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Teacher parity sets define y(x); training remains basis-free.
    d = 35
    teacher_spec = "{0,1,2,3}"   # example
    sets = [torch.tensor(s, device=device, dtype=torch.long)
            for s in parse_composite_spec(teacher_spec)]

    # Training set
    P_train = 10000
    X, y = generate_parity_data(P_train, d, sets, device)

    # Model / sampler / solver
    N = 512
    gamma = 0.5
    kappa = 7.5e-3

    mdl  = ModelParams(d=d, N=N, gamma=gamma, sigma_a=1.0, sigma_w=1.0, act="relu")
    mcmc = MCMCParams(B=512, step_size=1e-3, grad_clip=0.0, clamp_w=0.0, langevin_sqrt2=True)
    sol  = SolveParams(
        outer_steps=250000, print_every=250, use_reaction=False,
        cg_tol=1e-7, cg_maxit=200, ridge=0.0,
        P_eval=50_000, eval_chunk=8192*4
    )

    solver = BasisFreeCavitySolver(mdl, mcmc, sol, kappa=kappa, device=device, teacher_sets=sets)
    _ = solver.run(X, y, log_dir="./results_basis_free",
                   run_tag=f"Ptr{P_train}_Peval{sol.P_eval}_kap{kappa:.3e}_g{gamma}")
