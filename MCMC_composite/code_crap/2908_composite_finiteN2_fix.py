# basis_free_cavity_fixed.py
# Basis-free functional MF with 1/N Onsager reaction on the SAME P (quenched).
# Conventions used EVERYWHERE:
#   alpha(w) = 1/(2 σ_a^2) + Σ_eff(w)/(2 κ^2 N^{2γ})
#   beta(w)  = Jr(w)/(κ^2 N^γ),  where Jr = E_x[ φ(w^T x) y ] - E_x[ φ(w^T x) ⟨f(x)⟩ ]
#   mu(w)    = beta(w) / (2 alpha(w))           (posterior mean of a | w)
#   Var[a|w] = 1 / (2 alpha(w))                 (posterior variance)
#
# Integrated-out-a single-chain potential (used for SGLD):
#   U(w) = (d/(2 σ_w^2))||w||^2 + (1/2) ln alpha(w) - beta(w)^2 / (4 alpha(w))
#
# Onsager reaction on the SAME training set (quenched):
#   Φ_{μb} = φ(w_b^T x_μ),  C = (1/P) Φ^T Φ  (B×B Gram over points).
#   a_b^2 = mu_b^2 + Var[a|w_b].
#   R_b/N = (1/κ^2) * [ (C^2 @ (a^2/B))_b - ( (C @ (mu/B))_b )^2 ]        (connected)
#   Σ_eff,b = Σ_b - 2 * (R_b/N)                                            <-- factor 2 here
#
# Self-consistency each outer iter (basis-free, on the dataset):
#   Mean field:      ⟨f(x_μ)⟩ = N^{1-γ} * (1/B) Σ_b μ_b Φ_{μb}
#   Susceptibility:  χ_{μν}   = (N^{2-2γ}/κ^2) * [ (1/B) Σ_b a_b^2 Φ_{μβ}Φ_{νβ} - u_μ u_ν ],
#                    u_μ = (1/B) Σ_b μ_b Φ_{μβ}.
#
# Implementation notes for autograd stability:
#   • At the START of each outer iteration:      W = W.detach().requires_grad_(True)
#   • When computing Jm (→ Jr):                  use f_mean.detach()
#   • After the SGD/Langevin step:               recompute phi and (μ, Var) under no_grad
#   • Pass mu.detach(), var_a.detach() into χ and reaction; store R_divN_prev.detach()

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
    B: int = 4096
    steps: int = 1200
    step_size: float = 2e-6
    step_decay: float = 1.0
    grad_clip: float = 0.0
    clamp_w: float = 0.0
    langevin_sqrt2: bool = True
    autocast: Optional[bool] = False

@dataclass
class SolveParams:
    outer_steps: int = 1500
    print_every: int = 10
    # Optional damping on ⟨f⟩ (helps stability if needed)
    eta_f: float = 1.0     # EMA towards new ⟨f⟩; 1.0 = overwrite

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
def onsager_reaction_sameP(phi: torch.Tensor, mu: torch.Tensor, var_a: torch.Tensor, kappa: float) -> torch.Tensor:
    """
    Reaction on the SAME training set (quenched), connected:
      C = (1/P) Φ^T Φ  (B×B)
      a2 = mu^2 + var_a
      R_b/N = (1/κ^2) * [ (C^2 @ (a2/B))_b - ( (C @ (mu/B))_b )^2 ].
    Returns: R_divN (B,)
    """
    P, B = phi.shape
    C = (phi.t().matmul(phi)) / float(P)                  # (B,B)
    a2 = (mu * mu) + var_a                                # (B,)
    term_var = (C * C) @ (a2 / float(B))                  # (B,)
    mean_vec = C @ (mu / float(B))                        # (B,)
    R_divN = (term_var - mean_vec * mean_vec) / (kappa * kappa)
    return torch.clamp(R_divN, min=0.0)

@torch.no_grad()
def susceptibility_matrix(phi: torch.Tensor, mu: torch.Tensor, var_a: torch.Tensor, mdl: ModelParams, kappa: float) -> torch.Tensor:
    """
    Basis-free χ on SAME P (P×P, connected):
      χ_{μν} = (N^{2-2γ}/κ^2) * [ (1/B) Σ_b a_b^2 Φ_{μβ}Φ_{νβ} - u_μ u_ν ],
      u_μ = (1/B) Σ_b μ_b Φ_{μβ},  a_b^2 = μ_b^2 + Var[a|w_b].
    """
    P, B = phi.shape
    a2 = (mu * mu) + var_a                                  # (B,)
    term = (phi * a2.view(1, -1)) @ phi.t() / float(B)      # (P,P)
    u = (phi * mu.view(1, -1)).mean(dim=1, keepdim=True)    # (P,1)
    chi = ((float(mdl.N) ** (2.0 - 2.0 * float(mdl.gamma))) / (kappa * kappa)) * (term - u @ u.t())
    return chi

# ----------------------------- solver -----------------------------

class BasisFreeCavitySolver:
    def __init__(self, mdl: ModelParams, mcmc: MCMCParams, sol: SolveParams,
                 kappa: float, device: torch.device):
        self.mdl = mdl
        self.mcmc = mcmc
        self.sol = sol
        self.kappa = kappa
        self.device = device

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

        traj = {"iter": [], "train_mse": [], "mean_Sigma": [], "mean_R_divN": [],
                "mean_Sigma_eff": [], "min_Sigma_eff": [], "chi_normF": []}
        t0 = time.time()

        for it in range(1, self.sol.outer_steps + 1):
            # --- make W a fresh leaf each outer iter ---
            W = W.detach().requires_grad_(True)

            # ---------- forward (current graph) ----------
            phi = compute_phi(X, W, self.mdl.act)                         # (P,B)
            Sigma, JY, Jm, Jr = train_stats(phi, y, f_mean.detach())      # f_mean detached!

            # Σ_eff = Σ - 2 * (R/N)
            Sigma_eff = torch.clamp(Sigma - 0*2.0 * R_divN_prev, min=1e-12)

            # α, β, μ, Var[a] for CURRENT graph
            alpha, beta, mu, var_a = alpha_beta_mu_vara(Sigma_eff, Jr, self.mdl, self.kappa)

            # ---------- SGLD on U(w) ----------
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

            if self.mcmc.step_decay != 1.0:
                self.mcmc.step_size *= self.mcmc.step_decay

            # ---------- recompute (environment) under no_grad ----------
            with torch.no_grad():
                # Use UPDATED W, recompute φ and stats to update environment fields
                phi_env = compute_phi(X, W, self.mdl.act)
                Sigma_env = (phi_env * phi_env).mean(dim=0)                        # (B,)
                Sigma_eff_env = torch.clamp(Sigma_env - 0* 2.0 * R_divN_prev, min=1e-12)
                # Jr uses detached f_mean by definition of cavity
                _, _, _, Jr_env = train_stats(phi_env, y, f_mean.detach())
                alpha_env, beta_env, mu_env, var_a_env = alpha_beta_mu_vara(Sigma_eff_env, Jr_env, self.mdl, self.kappa)

                # SELF-CONSISTENCY #1: ⟨f(x)⟩ = N^{1-γ} E_b[ μ_b φ_b(x) ]
                f_new = (phi_env * mu_env.view(1, -1)).mean(dim=1)
                f_new = (float(self.mdl.N) ** (1.0 - float(self.mdl.gamma))) * f_new
                f_mean = (1.0 - self.sol.eta_f) * f_mean + self.sol.eta_f * f_new

                # SELF-CONSISTENCY #2 (optional diagnostic): χ on SAME P
                chi_mat = susceptibility_matrix(phi_env, mu_env, var_a_env, self.mdl, self.kappa)
                chi_normF = float((chi_mat * chi_mat).sum().sqrt().item())

                # Onsager reaction on SAME P (uses μ, Var from env)
                R_divN = onsager_reaction_sameP(phi_env, mu_env, var_a_env, self.kappa)
                R_divN_prev = R_divN.detach()

            # ---------- diagnostics ----------
            with torch.no_grad():
                train_mse = float(((y - f_mean) ** 2).mean().item())
                traj["iter"].append(it)
                traj["train_mse"].append(train_mse)
                traj["mean_Sigma"].append(float(Sigma_env.mean().item()))
                traj["mean_R_divN"].append(float(R_divN_prev.mean().item()))
                traj["mean_Sigma_eff"].append(float(Sigma_eff_env.mean().item()))
                traj["min_Sigma_eff"].append(float(Sigma_eff_env.min().item()))
                traj["chi_normF"].append(chi_normF)

                if (it % self.sol.print_every == 1) or (it == self.sol.outer_steps):
                    print(json.dumps({
                        "iter": it,
                        "train_mse": train_mse,
                        "mean_Sigma": traj["mean_Sigma"][-1],
                        "mean_R_divN": traj["mean_R_divN"][-1],
                        "mean_Sigma_eff": traj["mean_Sigma_eff"][-1],
                        "min_Sigma_eff": traj["min_Sigma_eff"][-1],
                        "chi_normF": chi_normF,
                        "elapsed_s": round(time.time() - t0, 2),
                        "B": B, "P": P
                    }))

        out = {
            "summary": {
                "P": int(P), "d": self.mdl.d, "N": self.mdl.N, "gamma": self.mdl.gamma,
                "kappa": self.kappa, "act": self.mdl.act, "sigma_a": self.mdl.sigma_a,
                "train_mse_last": traj["train_mse"][-1],
                "mean_R_divN_last": traj["mean_R_divN"][-1],
                "mean_Sigma_eff_last": traj["mean_Sigma_eff"][-1],
                "chi_normF_last": traj["chi_normF"][-1],
            },
            "traj": traj,
        }
        tag = run_tag or time.strftime("%Y%m%d_%H%M%S")
        gstr = f"{self.mdl.gamma:.6g}".replace('.', 'p').replace('-', 'm')
        path = os.path.join(
            log_dir,
            f"basis_free_cavity_{tag}_P{P}_kap{float(self.kappa):.3e}"
            f"_N{int(self.mdl.N)}_g{gstr}.json"
        )
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {path}")
        return out

# ----------------------------- main -----------------------------

if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Teacher parity sets define y(x) (basis-free learning; no projection of m)
    d = 35
    teacher_spec = "{0,1,2,3}"      # e.g., parity over first 4 bits
    sets = [torch.tensor(s, device=device, dtype=torch.long)
            for s in parse_composite_spec(teacher_spec)]

    # SAME P used for training, χ, and reaction (quenched)
    P = 2500
    X, y = generate_parity_data(P, d, sets, device)

    # Model / sampler / solver
    N = 512
    gamma = 0.5
    kappa = 7.5e-2

    mdl  = ModelParams(d=d, N=N, gamma=gamma, sigma_a=1.0, sigma_w=1.0, act="relu")
    mcmc = MCMCParams(B=4096*8, steps=12000, step_size=8e-5, step_decay=1.0,
                      grad_clip=10.0, clamp_w=0.0, autocast=False)
    
    sol  = SolveParams(outer_steps=1500, print_every=10, eta_f=0.07)

    solver = BasisFreeCavitySolver(mdl, mcmc, sol, kappa=kappa, device=device)
    _ = solver.run(X, y, log_dir="./results_basis_free",
                   run_tag=f"P{P}_kap{kappa:.3e}_g{gamma}")
