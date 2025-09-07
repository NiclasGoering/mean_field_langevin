# Basis-free functional MF with exact (I+K)^{-1} fixed-point solve for ⟨f⟩,
# OPTIONAL resolvent-based connected susceptibility χ self-consistency,
# and optional LOO Onsager reaction. Includes held-out diagnostics.
#
# Key corrections vs. earlier version:
#   • Factor-of-two in the 1/N mass shift:  Σ_eff = Σ - 2*(R/N).
#   • χ computed self-consistently from the resolvent:
#       χ = (N/κ^2) * (I + K)^(-1) K,  with  K = (N^{1-γ}/(B P)) Φ diag(A) Φ^T
#     Implemented efficiently in the B×B feature space, avoiding P×P matrices.
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
# Reaction (two options; both are CONNECTED):
#   Option "resolvent" (recommended; χ self-consistent):
#     C = (1/P) Φ^T Φ,  c = N^{1-γ}/B,  M = I + c * diag(A) * C
#     Z = M^{-1} ( c * diag(A) * C ),  R/N = (1/(κ^2 P)) * diag(C Z)
#   Option "loo" (MC, leave-one-out):
#     a2 = μ^2 + Var[a|w],  C = (1/P) Φ^T Φ
#     R_b/N = (1/κ^2) * [ (1/(B-1)) * sum_{j≠b} a2_j C_{bj}^2
#                         - ( (1/(B-1)) * sum_{j≠b} μ_j C_{bj} )^2 ]
#
# Held-out evaluation (every print):
#   Generate X_eval ∈ {±1}^d of size P_eval, y_eval = Σ_S χ_S(x)
#   f_eval(x) = N^{1-γ} * (1/B) Σ_b μ_b φ(w_b^T x)
#   m_S = E_eval[ f_eval χ_S ]; noise/mode MSEs as in prior script.

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
    # X ∈ {±1}^d, y = ∑_S χ_S(x)
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
    step_size: float = 2e-5     # MALA stepsize (fixed)
    grad_clip: float = 0.0      # unused in MALA to avoid bias
    clamp_w: float = 0.0        # unused in MALA to avoid bias
    langevin_sqrt2: bool = True # kept for API compatibility (not used in MALA)

@dataclass
class SolveParams:
    outer_steps: int = 500
    print_every: int = 10
    reaction_mode: str = "resolvent"  # "resolvent" (self-consistent χ) or "loo"
    reaction_every: int = 1           # compute R every k steps (resolvent can be heavy)
    use_reaction: bool = True
    # CG solve for (I+K) f = K y
    cg_tol: float = 1e-6
    cg_maxit: int = 200
    ridge: float = 1e-12
    # Held-out evaluation
    P_eval: int = 200_000
    eval_chunk: int = 8192

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

@torch.no_grad()
def reaction_via_resolvent(phi: torch.Tensor, A: torch.Tensor, N: int, gamma: float, kappa: float) -> torch.Tensor:
    """
    Self-consistent (connected) susceptibility via resolvent in feature space.
      C = (1/P) Φ^T Φ,  c = N^{1-γ}/B,
      M = I + c * diag(A) * C,
      Z = M^{-1} (c * diag(A) * C),
      R_b/N = (1/(κ^2 P)) * [C Z]_{bb}.
    """
    P, B = phi.shape
    device = phi.device
    dtype = phi.dtype

    C = (phi.t() @ phi) / float(P)                # (B,B)
    c = (float(N) ** (1.0 - float(gamma))) / float(B)

    # M and R := c * diag(A) * C
    M = torch.eye(B, device=device, dtype=dtype) + (c * (A.view(-1, 1) * C))
    R = c * (A.view(-1, 1) * C)

    # Solve M Z = R for multiple RHS (B×B); SPD so solve is stable
    Z = torch.linalg.solve(M, R)                  # (B,B)
    CZ = C @ Z                                    # (B,B)
    diag_CZ = torch.diag(CZ)                      # (B,)

    R_divN = (diag_CZ / (float(P) * (kappa ** 2))).clamp_min(0.0)  # (B,)
    return R_divN

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

        g = torch.Generator(device=device).manual_seed(12345)
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
            Xc = (torch.randint(0, 2, (n, d), generator=g, device=device, dtype=torch.int8).float() * 2.0 - 1.0)
            z = Xc @ W.t().contiguous()                        # (n,B)
            phi_c = activation(z, self.mdl.act).to(torch.float32)
            f_chunk = N1mg * (phi_c * mu_row).mean(dim=1)      # (n,)

            if M > 0:
                C_cols = [parity_character(Xc, S) for S in self.teacher_sets]
                Cc = torch.stack(C_cols, dim=1)                # (n,M)
                sum_Ct_f += Cc.t().matmul(f_chunk)             # (M,)
                sum_G    += Cc.t().matmul(Cc)                  # (M,M)

            sum_f2 += (f_chunk * f_chunk).sum()

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

        v = sum_Ct_f * inv_P                     # (M,)  ~ m_S
        G = sum_G * inv_P                         # (M,M)
        ones = torch.ones(M, device=device, dtype=torch.float32)

        m_S = v
        mTm = float((m_S * m_S).sum().item())
        mTGm = float(m_S.view(1, -1).matmul(G).matmul(m_S.view(-1, 1)).item())
        noise = f2_bar - 2.0 * mTm + mTGm

        half_mse_modes = 0.5 * float(((1.0 - m_S) ** 2).sum().item())
        half_noise = 0.5 * float(noise)
        half_mse_total_ms = half_mse_modes + half_noise

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
            "iter": [], "train_mse": [], "mean_Sigma": [], "mean_R_divN": [], "mean_Sigma_eff": [],
            "min_Sigma_eff": [], "chi_diag_est": [],
            # held-out diagnostics
            "eval_m_S": [], "half_mse_modes": [], "half_noise": [],
            "half_mse_total_ms": [], "half_mse_empirical": [],
            "mala_accept_rate": []
        }
        t0 = time.time()
        accept_window = 0

        for it in range(1, self.sol.outer_steps + 1):
            # --- make W a fresh leaf each outer iter ---
            W = W.detach().requires_grad_(True)

            # ---------- forward (current graph) ----------
            phi = compute_phi(X, W, self.mdl.act)                         # (P,B)
            Sigma, JY, Jm, Jr = train_stats(phi, y, f_mean.detach())      # f_mean detached!

            # Σ_eff = Σ - 2 * (R/N)  [factor-of-two correction]
            Sigma_eff = Sigma - (2.0 * R_divN_prev if self.sol.use_reaction else 0.0)

            # α, β, μ, Var[a] for CURRENT graph
            alpha, beta, mu, var_a = alpha_beta_mu_vara(Sigma_eff, Jr, self.mdl, self.kappa)

            # ---------- one MALA step on U(w) (Metropolis-Adjusted Langevin) ----------
            U_vec = potential_U(W, Sigma_eff, Jr, self.mdl, self.kappa)   # (B,)
            loss = U_vec.sum()
            loss.backward()

            with torch.no_grad():
                U_curr = loss.detach()
                grad_curr = W.grad.detach().clone()

                eta = self.mcmc.step_size
                xi = torch.randn_like(W)
                W_prop = (W.detach() - eta * grad_curr + math.sqrt(2.0 * eta) * xi).requires_grad_(True)

            # Compute U and grad at proposal
            phi_prop = compute_phi(X, W_prop, self.mdl.act)
            Sigma_prop, JY_prop, Jm_prop, Jr_prop = train_stats(phi_prop, y, f_mean.detach())
            Sigma_eff_prop = Sigma_prop - (2.0 * R_divN_prev if self.sol.use_reaction else 0.0)
            U_vec_prop = potential_U(W_prop, Sigma_eff_prop, Jr_prop, self.mdl, self.kappa)
            U_prop = U_vec_prop.sum()
            U_prop.backward()
            with torch.no_grad():
                grad_prop = W_prop.grad.detach().clone()

                def sqnorm(A):  # ||A||^2
                    return (A * A).sum()

                m_curr = W.detach() - eta * grad_curr
                m_prop = W_prop.detach() - eta * grad_prop

                log_q_prop_given_curr = - sqnorm(W_prop.detach() - m_curr) / (4.0 * eta)
                log_q_curr_given_prop = - sqnorm(W.detach()      - m_prop) / (4.0 * eta)

                log_acc = (-U_prop + U_curr) + (log_q_curr_given_prop - log_q_prop_given_curr)

                u = torch.rand((), device=W.device).log()
                accept = (u < log_acc)

                if bool(accept.item()):
                    W = W_prop.detach()
                    accept_window += 1
                else:
                    W = W.detach()

            W.requires_grad_(True)

            # ---------- recompute environment & SOLVE f via CG ----------
            with torch.no_grad():
                phi_env = compute_phi(X, W, self.mdl.act)                 # (P,B)
                Sigma_env = (phi_env * phi_env).mean(dim=0)               # (B,)

                # Jr under cavity (fixed ⟨f⟩)
                _, _, _, Jr_env = train_stats(phi_env, y, f_mean.detach())

                # Σ_eff uses PREVIOUS reaction (outer fixed point)
                Sigma_eff_env = Sigma_env - (2.0 * R_divN_prev if self.sol.use_reaction else 0.0)

                # α, μ, Var for environment (used for reaction & A)
                alpha_env, beta_env, mu_env, var_a_env = alpha_beta_mu_vara(Sigma_eff_env, Jr_env, self.mdl, self.kappa)

                # Build A_b = 1/(2 α_b κ^2 N^γ)
                A_env = (1.0 / (2.0 * alpha_env * (self.kappa ** 2) * (float(self.mdl.N) ** float(self.mdl.gamma)))).to(torch.float32)

                # Solve (I + K) f = K y   with K v = (N^{1-γ}/(B P)) Φ (A ⊙ (Φ^T v))
                Kop = KOperator(phi_env, A_env, self.mdl.N, self.mdl.gamma)
                rhs = Kop.matvec(y)                  # K y
                f_new = cg_solve_I_plus_K(Kop, rhs, tol=self.sol.cg_tol, maxit=self.sol.cg_maxit, ridge=self.sol.ridge)
                f_mean = f_new  # exact fixed point on training points

                # --- Reaction update (either resolvent-χ or LOO), not necessarily every step ---
                recompute_reaction = (it % max(1, self.sol.reaction_every) == 0)
                if self.sol.use_reaction and recompute_reaction:
                    if self.sol.reaction_mode.lower() == "resolvent":
                        # Self-consistent χ via resolvent in feature space
                        R_divN = reaction_via_resolvent(phi_env, A_env, self.mdl.N, self.mdl.gamma, self.kappa)
                    elif self.sol.reaction_mode.lower() == "loo":
                        R_divN = onsager_reaction_sameP_loo(phi_env, mu_env, (1.0 / (2.0 * alpha_env)), self.kappa)
                    else:
                        raise ValueError(f"Unknown reaction_mode: {self.sol.reaction_mode}")
                    R_divN_prev = R_divN.detach()

            # ---------- diagnostics ----------
            with torch.no_grad():
                train_mse = float(((y - f_mean) ** 2).mean().item())

                # Held-out evaluation & logs every print interval
                if (it % self.sol.print_every == 1) or (it == self.sol.outer_steps):
                    window = min(it, self.sol.print_every)
                    mala_acc_rate = accept_window / float(window)
                    accept_window = 0

                    eval_stats = self._eval_big_heldout(W.detach(), mu_env.detach())

                    traj["iter"].append(it)
                    traj["train_mse"].append(train_mse)
                    traj["mean_Sigma"].append(float(Sigma_env.mean().item()))
                    traj["mean_R_divN"].append(float(R_divN_prev.mean().item()))
                    Sigma_eff_log = Sigma_env - (2.0 * R_divN_prev if self.sol.use_reaction else 0.0)
                    traj["mean_Sigma_eff"].append(float(Sigma_eff_log.mean().item()))
                    traj["min_Sigma_eff"].append(float(Sigma_eff_log.min().item()))
                    traj["chi_diag_est"].append([])  # placeholder; χ can be large to log
                    traj["eval_m_S"].append(eval_stats["eval_m_S"])
                    traj["half_mse_modes"].append(eval_stats["half_mse_modes"])
                    traj["half_noise"].append(eval_stats["half_noise"])
                    traj["half_mse_total_ms"].append(eval_stats["half_mse_total_ms"])
                    traj["half_mse_empirical"].append(eval_stats["half_mse_empirical"])
                    traj["mala_accept_rate"].append(mala_acc_rate)

                    print(json.dumps({
                        "iter": it,
                        "train_mse": train_mse,
                        "mean_Sigma": float(Sigma_env.mean().item()),
                        "mean_R_divN": float(R_divN_prev.mean().item()),
                        "mean_Sigma_eff": float(Sigma_eff_log.mean().item()),
                        "min_Sigma_eff": float(Sigma_eff_log.min().item()),
                        "half_mse_modes": eval_stats["half_mse_modes"],
                        "half_noise": eval_stats["half_noise"],
                        "m_S": eval_stats["eval_m_S"],
                        "half_mse_total_ms": eval_stats["half_mse_total_ms"],
                        "half_mse_empirical": eval_stats["half_mse_empirical"],
                        "mala_acc_rate": mala_acc_rate,
                        "B": B, "P_train": P, "P_eval": self.sol.P_eval,
                        "elapsed_s": round(time.time() - t0, 2),
                        "reaction_mode": self.sol.reaction_mode
                    }))

                else:
                    traj["iter"].append(it)
                    traj["train_mse"].append(train_mse)
                    traj["mean_Sigma"].append(float(Sigma_env.mean().item()))
                    traj["mean_R_divN"].append(float(R_divN_prev.mean().item()))
                    Sigma_eff_log = Sigma_env - (2.0 * R_divN_prev if self.sol.use_reaction else 0.0)
                    traj["mean_Sigma_eff"].append(float(Sigma_eff_log.mean().item()))
                    traj["min_Sigma_eff"].append(float(Sigma_eff_log.min().item()))
                    traj["chi_diag_est"].append([])

        out = {
            "summary": {
                "P_train": int(P), "d": self.mdl.d, "N": self.mdl.N, "gamma": self.mdl.gamma,
                "kappa": self.kappa, "act": self.mdl.act, "sigma_a": self.mdl.sigma_a,
                "train_mse_last": traj["train_mse"][-1],
                "mean_R_divN_last": traj["mean_R_divN"][-1],
                "mean_Sigma_eff_last": traj["mean_Sigma_eff"][-1],
                "P_eval": int(self.sol.P_eval),
                "reaction_mode": self.sol.reaction_mode
            },
            "traj": traj,
        }
        tag = run_tag or time.strftime("%Y%m%d_%H%M%S")
        gstr = f"{self.mdl.gamma:.6g}".replace('.', 'p').replace('-', 'm')
        path = os.path.join(
            log_dir,
            f"basis_free_cavity_cg_eval_{tag}_Ptr{P}_Peval{int(self.sol.P_eval)}"
            f"_kap{float(self.kappa):.3e}_N{int(self.mdl.N)}_g{gstr}_{self.sol.reaction_mode}.json"
        )
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {path}")
        return out

# ----------------------------- main ------------------------------

if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Teacher parity sets define y(x); training remains basis-free.
    d = 35
    teacher_spec = "{0,1,2,3}"   # example
    sets = [torch.tensor(s, device=device, dtype=torch.long)
            for s in parse_composite_spec(teacher_spec)]

    # Training set
    P_train = 2500
    X, y = generate_parity_data(P_train, d, sets, device)

    # Model / sampler / solver
    N = 512
    gamma = 0.5
    kappa = 7.5e-3

    mdl  = ModelParams(d=d, N=N, gamma=gamma, sigma_a=1.0, sigma_w=1.0, act="relu")
    mcmc = MCMCParams(B=512, step_size=5e-8, grad_clip=0.0, clamp_w=0.0, langevin_sqrt2=True)

    # NOTE: resolvent χ involves a B×B solve; you can set reaction_every > 1 if B is large.
    sol  = SolveParams(
        outer_steps=250000, print_every=50, use_reaction=True,
        reaction_mode="resolvent", reaction_every=1,
        cg_tol=1e-6, cg_maxit=200, ridge=0.0,
        P_eval=50_000, eval_chunk=8192*4
    )

    solver = BasisFreeCavitySolver(mdl, mcmc, sol, kappa=kappa, device=device, teacher_sets=sets)
    _ = solver.run(X, y, log_dir="./results_basis_free",
                   run_tag=f"Ptr{P_train}_Peval{sol.P_eval}_kap{kappa:.3e}_g{gamma}")
