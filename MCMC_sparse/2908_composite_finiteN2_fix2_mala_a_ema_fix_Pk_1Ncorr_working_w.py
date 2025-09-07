# rs_cavity_explicit_aw_ard_fast_reaction_fixed_ngamma_INTOUT_A_W_ONLY_W.py
# RS self-consistent cavity with explicit (w), per-particle MALA/SGLD,
# ARD, and finite-N Onsager/back-reaction corrections with a integrated out.
#
# Changes vs original:
# - We DO NOT sample/update 'a'. Instead we integrate it out:
#     alpha = 1/(2 σ_a^2) + Σ_eff/(2 κ^2 N^{2γ})
#     beta  = Jr/(κ^2 N^γ), Jr = E_x[ φ(w^T x) (y - ⟨f⟩) ]
#     μ     = beta/(2 alpha), Var[a|w] = 1/(2 alpha)
#   Potential (per particle):
#     U(w) = (d/(2 σ_w^2))||w||^2 + 0.5*log(alpha) - beta^2/(4*alpha)
# - Mean field on points uses μ(w):  ⟨f⟩ = N^{1-γ} (1/B) Φ μ
# - Reaction uses E[(a/N^γ)^2] = (μ^2 + Var[a])/N^{2γ} in the Gram-based R̃.
#
# The rest (configs, ARD on w, logging, sharding, etc.) remains unchanged.

import os, time, math, json, random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

# ----------------------------- utils -----------------------------

def set_seed(seed: int = 12345):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def activation(z: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "relu": return F.relu(z)
    if kind == "tanh": return torch.tanh(z)
    raise ValueError(f"Unknown activation: {kind}")

def act_prime(z: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "relu": return (z > 0).to(z.dtype)
    if kind == "tanh": return 1.0 - torch.tanh(z) ** 2
    raise ValueError(f"Unknown activation: {kind}")

def parity_character(X_pm1: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    if X_pm1.dim() == 2:
        if S.numel() == 0:
            return torch.ones(X_pm1.shape[0], device=X_pm1.device, dtype=X_pm1.dtype)
        return X_pm1[:, S].prod(dim=1).to(X_pm1.dtype)
    elif X_pm1.dim() == 3:
        if S.numel() == 0:
            return torch.ones(X_pm1.shape[0], X_pm1.shape[1], device=X_pm1.device, dtype=X_pm1.dtype)
        return X_pm1[:, :, S].prod(dim=2).to(X_pm1.dtype)
    else:
        raise ValueError("X must be (P,d) or (E,P,d)")

def parse_sets(spec: str) -> List[List[int]]:
    import re
    blocks = re.findall(r"\{([^}]*)\}", spec)
    out = []
    for s in blocks:
        toks = [t.strip() for t in s.split(",") if t.strip()!=""]
        out.append(sorted(map(int, toks)))
    if not out: raise ValueError("bad teacher spec")
    return out

def generate_parity_multi(P: int, d: int, sets: List[torch.Tensor], E: int,
                          data_seeds: List[int], device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate E independent parity datasets in ±1 coding.
    Returns X ∈ ℝ[E,P,d], y ∈ ℝ[E,P,1]."""
    assert len(data_seeds) == E
    Xs = []
    ys = []
    for e in range(E):
        g = torch.Generator(device=device).manual_seed(int(data_seeds[e]))
        Xe = (torch.randint(0,2,(P,d),generator=g,device=device,dtype=torch.int8).to(dtype) * 2.0 - 1.0)
        Ccols = [parity_character(Xe, S) for S in sets]
        Ce = torch.stack(Ccols, dim=1) if Ccols else torch.zeros(P,0,device=device, dtype=dtype)
        ye = Ce.sum(dim=1, keepdim=True)
        Xs.append(Xe)
        ys.append(ye)
    X = torch.stack(Xs, dim=0)
    y = torch.stack(ys, dim=0)
    return X, y

# ----------------------------- config -----------------------------

@dataclass
class Model:
    d: int = 35
    B: int = 16384
    N: int = 512          # f = N^{-γ} Σ μ φ
    gamma: float = 0.5
    sigma_a: float = 1.0
    sigma_w: float = 1.0
    act: str = "relu"

@dataclass
class ARD:
    use_ard: bool = False
    alpha0: float = 1e-2
    ema: float = 0.25
    update_every: int = 1
    rho_min: float = 1e-12
    rho_max: float = 1e12
    beta0: Optional[float] = None

@dataclass
class Algo:
    outer_steps: int = 2000
    inner_mala_steps: int = 1
    step_size: float = 1e-6
    use_mala: bool = True
    log_every: int = 10
    cg_like_update: bool = False
    field_blend: float = 1.0
    batch_eval: int = 262_144
    P_eval: int = 50_000

    # chunking
    B_chunk: Optional[int] = None
    B_chunk_eval: Optional[int] = None
    P_chunk_train: Optional[int] = 4096

    # safety
    grad_clip_norm: Optional[float] = None
    use_float64: bool = False
    kill_nan_particles: bool = True
    max_abs_w: Optional[float] = None
    max_abs_a: Optional[float] = None  # unused now; kept for compatibility
    max_l2_w: Optional[float] = None
    max_l2_a: Optional[float] = None   # unused now; kept for compatibility
    nan_reinit_std_scale: float = 1.0
    log_bad_counts: bool = True

    # early stop
    early_stop_enabled: bool = False
    early_stop_threshold: float = 0.0
    early_stop_patience: int = 0

    # ---- finite-N reaction (Onsager) options ----
    reaction_use: bool = False
    reaction_mode: str = "exact_diag"       # or "lowrank_gram"
    reaction_update_every: int = 10
    reaction_particle_subsample: Optional[int] = None  # only for exact_diag

    # low-rank options (for "lowrank_gram")
    reaction_lowrank_rank: int = 32
    reaction_lowrank_power_iters: int = 0   # unused; full eigh is fine for B<=2k
    reaction_clamp_sigma_eff: bool = False  # numerical guard (off by default)

# ----------------------------- core ------------------------------

class RSCavityExplicitMulti:
    def __init__(self, mdl: Model, algo: Algo, kappa: float, device: torch.device,
                 E: int, seeds_params: List[int],
                 teacher_sets: Optional[List[torch.Tensor]] = None,
                 ard: Optional[ARD] = None):
        self.mdl, self.algo, self.kappa = mdl, algo, float(kappa)
        self.device = device
        self.sets = teacher_sets or []
        self.ard = ard or ARD()
        self.dtype = torch.float64 if algo.use_float64 else torch.float32
        self.E = int(E)
        assert len(seeds_params) == E

        # parameters: only W (a is integrated out)
        W_list = []
        for e in range(E):
            g = torch.Generator(device=device).manual_seed(int(seeds_params[e]))
            We = torch.randn(mdl.B, mdl.d, generator=g, device=device, dtype=self.dtype) * (mdl.sigma_w / math.sqrt(mdl.d))
            W_list.append(We)
        self.W = torch.stack(W_list, dim=0)  # (E,B,d)

        # ARD (on w only; rho is per-dimension precision)
        rho0 = (mdl.d / (mdl.sigma_w**2))
        self.rho = torch.full((E, mdl.d), rho0, device=device, dtype=self.dtype)
        self.beta0 = float(self.ard.alpha0 / rho0) if self.ard.beta0 is None else float(self.ard.beta0)

        # constants
        self.N_gamma = self.mdl.N ** self.mdl.gamma
        self.scale_f = (self.mdl.N ** (1.0 - self.mdl.gamma)) / float(self.mdl.B)
        try:
            torch.set_float32_matmul_precision('high')
            torch.backends.cuda.matmul.allow_tf32 = not algo.use_float64
            torch.backends.cudnn.allow_tf32 = not algo.use_float64
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

        # ---- Reaction buffers ----
        self.reaction_prefac = 2.0 / float(self.mdl.N)  # exact 2/N (acts on Σ via R̃)
        self.reaction_ready = False
        self.reaction_R_tilde = torch.zeros(self.E, self.mdl.B, device=self.device, dtype=self.dtype)
        self.reaction_diagSigma = torch.zeros(self.E, self.mdl.B, device=self.device, dtype=self.dtype)
        self.reaction_offdiag_ratio = torch.zeros(self.E, device=self.device, dtype=self.dtype)

        # last-step C2 stats for logging (C2 = sum φ^2 over training)
        self._last_c2_mean = 0.0
        self._last_c2eff_mean = 0.0

        # caches updated each iteration
        self._mu = torch.zeros(self.E, self.mdl.B, device=self.device, dtype=self.dtype)   # posterior mean of a
        self._var_a = torch.zeros_like(self._mu)                                            # posterior var of a
        self._f_mean_cache = None
        self._y_cache = None

    # ---------- helpers ----------

    @torch.no_grad()
    def _reinit_particles(self, mask: torch.Tensor):
        if not mask.any(): return 0
        sw = self.mdl.sigma_w * self.algo.nan_reinit_std_scale
        total = 0
        for e in range(self.E):
            m = mask[e]
            if m.any():
                n = int(m.sum().item()); total += n
                g = torch.Generator(device=self.device).manual_seed(int(17_123 + e))
                self.W[e, m] = torch.randn(n, self.mdl.d, generator=g, device=self.device, dtype=self.dtype) * (sw / math.sqrt(self.mdl.d))
        return total

    @torch.no_grad()
    def _clamp_params(self):
        if self.algo.max_abs_w is not None:
            self.W.clamp_(-self.algo.max_abs_w, self.algo.max_abs_w)
        if self.algo.max_l2_w is not None:
            norms = torch.linalg.vector_norm(self.W, dim=2, keepdim=True) + 1e-12
            self.W.mul_((torch.clamp(self.algo.max_l2_w / norms, max=1.0)))

    @staticmethod
    def _batched_mm_X_Wt(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        return torch.bmm(X, W.transpose(1, 2))

    # -------- integrated-out helpers (alpha/beta/mu/var) --------

    def _alpha_beta_mu_var_from_stats(self, Sigma_eff: torch.Tensor, Jr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Shapes: Sigma_eff, Jr ∈ (E,B).
        alpha = 1/(2σ_a^2) + Σ_eff/(2 κ^2 N^{2γ})
        beta  = Jr/(κ^2 N^γ)
        mu    = beta/(2 alpha)
        var_a = 1/(2 alpha)
        """
        N2g = (float(self.mdl.N) ** (2.0 * float(self.mdl.gamma)))
        alpha = (1.0 / (2.0 * (self.mdl.sigma_a ** 2))) + (Sigma_eff / (2.0 * (self.kappa ** 2) * N2g))
        beta  = Jr / ((self.kappa ** 2) * (float(self.mdl.N) ** float(self.mdl.gamma)))
        alpha = torch.clamp(alpha, min=torch.tensor(1e-30, dtype=self.dtype, device=self.device))
        mu    = beta / (2.0 * alpha)
        var_a = 1.0 / (2.0 * alpha)
        return alpha, beta, mu, var_a

    # ---------------- Energy & grad via autograd (per E) ----------------

    def _potential_U_per_exp(self, Xe: torch.Tensor, ye: torch.Tensor, fmean_e: torch.Tensor, We: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Xe: (P,d), ye: (P,1), fmean_e: (P,1) [detached], We: (B,d) requires_grad=True
        Returns:
          U_vec: (B,), Sigma: (B,), Jr: (B,), alpha: (B,), mu: (B,)
        """
        P = Xe.shape[0]
        z = Xe @ We.t()                          # (P,B)
        phi = activation(z, self.mdl.act)        # (P,B)
        Sigma = (phi * phi).mean(dim=0)          # (B,)

        # reaction correction on Σ, if available
        if self.algo.reaction_use and self.reaction_ready:
            Sigma_eff = Sigma - (self.reaction_prefac * self.reaction_R_tilde[self._exp_idx])
            if self.algo.reaction_clamp_sigma_eff:
                Sigma_eff = torch.maximum(Sigma_eff, torch.zeros_like(Sigma_eff))
        else:
            Sigma_eff = Sigma

        r = (ye.view(-1) - fmean_e.view(-1)).detach()   # (P,)
        Jr = (phi.t() @ r) / float(P)                   # (B,)

        alpha, beta, mu, var_a = self._alpha_beta_mu_var_from_stats(Sigma_eff, Jr)

        # prior term (per particle)
        var_w_per_coord = (self.mdl.sigma_w ** 2) / self.mdl.d
        prior = 0.5 * (We * We).sum(dim=1) / var_w_per_coord    # (B,)

        U_vec = prior + 0.5 * torch.log(alpha) - (beta * beta) / (4.0 * alpha)  # (B,)
        return U_vec, Sigma, Jr, alpha, mu

    def _U_and_grad_autograd(self, X: torch.Tensor, y: torch.Tensor, f_mean: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute per-particle energies U(w) and grad_w via autograd for all experiments.
        Returns:
          grad_w: (E,B,d), U_vec: (E,B), aux: dict with Sigma, Jr, alpha, mu (each (E,B))
        """
        E, P, d = X.shape
        grad_w_all = torch.zeros_like(self.W, dtype=self.dtype)
        U_all = torch.zeros(E, self.mdl.B, device=self.device, dtype=self.dtype)

        Sigma_all = torch.zeros_like(U_all)
        Jr_all    = torch.zeros_like(U_all)
        alpha_all = torch.zeros_like(U_all)
        mu_all    = torch.zeros_like(U_all)

        for e in range(E):
            self._exp_idx = e  # used for reaction Sigma_eff picking
            We = self.W[e].detach().requires_grad_(True)
            U_vec, Sigma, Jr, alpha, mu = self._potential_U_per_exp(
                X[e], y[e], f_mean[e], We
            )
            loss = U_vec.sum()
            loss.backward()
            grad_w_all[e] = We.grad.detach().to(self.dtype)
            U_all[e]      = U_vec.detach().to(self.dtype)

            Sigma_all[e] = Sigma.detach().to(self.dtype)
            Jr_all[e]    = Jr.detach().to(self.dtype)
            alpha_all[e] = alpha.detach().to(self.dtype)
            mu_all[e]    = mu.detach().to(self.dtype)

        aux = {"Sigma": Sigma_all, "Jr": Jr_all, "alpha": alpha_all, "mu": mu_all}
        return grad_w_all, U_all, aux

    # ---------------- Field (⟨f⟩) using μ(w) ----------------

    @torch.no_grad()
    def _field_from_particles_stream_mu(self, X: torch.Tensor, y: torch.Tensor, f_mean_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Two-pass stream over training X to compute:
          1) Sigma and Jr (per particle) -> alpha, mu, var_a
          2) f(x) = scale_f * Φ μ
        Returns (f, mu, var_a).
        """
        E, P, d = X.shape
        B = self.mdl.B
        Sigma = torch.zeros(E, B, device=self.device, dtype=self.dtype)
        Jr    = torch.zeros(E, B, device=self.device, dtype=self.dtype)

        step = self.algo.P_chunk_train or P
        # ---- pass 1: stats
        for start in range(0, P, step):
            n = min(step, P - start)
            Xc = X[:, start:start+n, :]
            rc = (y[:, start:start+n, 0] - f_mean_prev[:, start:start+n, 0])  # (E,n)
            Z  = self._batched_mm_X_Wt(Xc, self.W)                            # (E,n,B)
            Phi = activation(Z, self.mdl.act)                                 # (E,n,B)
            Sigma += (Phi * Phi).sum(dim=1) / float(P)
            # Jr = E[φ r]
            for e in range(E):
                Jr[e] += (Phi[e].t().matmul(rc[e])) / float(P)

        # reaction correction on Σ
        if self.algo.reaction_use and self.reaction_ready:
            Sigma_eff = Sigma - (self.reaction_prefac * self.reaction_R_tilde)
            if self.algo.reaction_clamp_sigma_eff:
                Sigma_eff = torch.maximum(Sigma_eff, torch.zeros_like(Sigma_eff))
        else:
            Sigma_eff = Sigma

        alpha, beta, mu, var_a = self._alpha_beta_mu_var_from_stats(Sigma_eff, Jr)

        # ---- pass 2: f(x) using μ
        f = torch.zeros(E, P, device=self.device, dtype=self.dtype)
        mu3 = mu.unsqueeze(2)  # (E,B,1)
        for start in range(0, P, step):
            n = min(step, P - start)
            Xc = X[:, start:start+n, :]
            Z  = self._batched_mm_X_Wt(Xc, self.W)              # (E,n,B)
            Phi = activation(Z, self.mdl.act)                   # (E,n,B)
            fa = torch.bmm(Phi, mu3)                            # (E,n,1)
            f[:, start:start+n] = (self.scale_f * fa.squeeze(-1))
        return f.unsqueeze(-1), mu, var_a

    # ---------------- Reaction (1/N) ----------------

    @torch.no_grad()
    def _update_reaction(self, X: torch.Tensor):
        """Quenched back-reaction using ONLY training X.
        Build H_hat = (1/P) Φ^T Φ and Sigma_hat = diag(H_hat),
        then R_tilde = (H_hat ∘ H_hat) @ a2bar,
        with a2bar = E[(a/N^γ)^2] = (μ^2 + Var[a])/N^{2γ} computed from current f_mean.
        """
        if not self.algo.reaction_use:
            self.reaction_ready = False
            return
        if self._f_mean_cache is None or self._y_cache is None:
            self.reaction_ready = False
            return

        E, P, d = X.shape
        B = self.mdl.B

        # optional subsample (exact_diag only)
        if (self.algo.reaction_mode == "exact_diag" and
            self.algo.reaction_particle_subsample is not None and
            self.algo.reaction_particle_subsample < B):
            idx = torch.randperm(B, device=self.device)[: self.algo.reaction_particle_subsample]
            W_use = self.W[:, idx]
            B_eff = int(idx.numel())
            idx_back = idx
        else:
            W_use = self.W
            B_eff = B
            idx_back = None

        # accumulate H and diag sum
        H = torch.zeros(self.E, B_eff, B_eff, device=self.device, dtype=self.dtype)
        diag_sum = torch.zeros(self.E, B_eff, device=self.device, dtype=self.dtype)

        step = self.algo.P_chunk_train or P

        # We also need μ and Var[a] (from training stats with current f_mean)
        # First accumulate Sigma and Jr to get μ,Var
        Sigma = torch.zeros(self.E, B_eff, device=self.device, dtype=self.dtype)
        Jr    = torch.zeros(self.E, B_eff, device=self.device, dtype=self.dtype)

        for start in range(0, P, step):
            n = min(step, P - start)
            Xc = X[:, start:start+n, :]
            Z  = self._batched_mm_X_Wt(Xc, W_use)      # (E,n,B_eff)
            Phi = activation(Z, self.mdl.act)

            H += torch.bmm(Phi.transpose(1, 2), Phi)
            diag_sum += (Phi * Phi).sum(dim=1)

            # stats for μ/Var using current residual r = y - f_mean
            rc = (self._y_cache[:, start:start+n, 0] - self._f_mean_cache[:, start:start+n, 0])
            Sigma += (Phi * Phi).sum(dim=1) / float(P)
            for e in range(self.E):
                Jr[e] += (Phi[e].t().matmul(rc[e])) / float(P)

        H_hat = H / float(P)                # (E,B_eff,B_eff)
        Sigma_hat = diag_sum / float(P)     # (E,B_eff)

        # μ and Var[a] from Sigma (no reaction in this pass to avoid circularity)
        alpha, beta, mu, var_a = self._alpha_beta_mu_var_from_stats(Sigma, Jr)
        # a2bar = E[(a/N^γ)^2] = (μ^2 + Var[a]) / N^{2γ}
        a2bar = (mu*mu + var_a) / (float(self.mdl.N) ** (2.0 * float(self.mdl.gamma)))  # (E,B_eff)

        # diagnostics: offdiag ratio
        eps = torch.tensor(1e-24, dtype=self.dtype, device=self.device)
        off_ratios = []
        for e in range(self.E):
            He = H_hat[e]
            diag = torch.diagonal(He, 0)
            Hdiag = torch.diag(diag)
            R = He - Hdiag
            num = torch.linalg.matrix_norm(R, ord='fro') ** 2
            den = torch.linalg.matrix_norm(He, ord='fro') ** 2 + eps
            off_ratios.append(torch.sqrt(num / den))
        self.reaction_offdiag_ratio = torch.stack(off_ratios, dim=0)

        # R_tilde per mode
        if self.algo.reaction_mode == "exact_diag":
            H_sq = H_hat * H_hat
            R = torch.bmm(H_sq, a2bar.unsqueeze(-1)).squeeze(-1)  # (E,B_eff)
        elif self.algo.reaction_mode == "lowrank_gram":
            r = min(self.algo.reaction_lowrank_rank, B_eff)
            R_list = []
            for e in range(self.E):
                He = H_hat[e]
                evals, evecs = torch.linalg.eigh(He)   # ascending
                vals = evals[-r:]
                U = evecs[:, -r:]                      # (B_eff, r)
                # H_sq ≈ U diag(vals^2) U^T ; R = H_sq @ a2bar
                x = U.T.matmul(a2bar[e])               # (r,)
                y = (vals*vals) * x                    # (r,)
                R_e = U.matmul(y)                      # (B_eff,)
                R_list.append(R_e)
            R = torch.stack(R_list, dim=0)
        else:
            raise ValueError(f"Unknown reaction_mode: {self.algo.reaction_mode}")

        # scatter back
        if idx_back is None:
            self.reaction_R_tilde = R
            self.reaction_diagSigma = Sigma_hat
        else:
            self.reaction_R_tilde.zero_(); self.reaction_diagSigma.zero_()
            for e in range(self.E):
                self.reaction_R_tilde[e, idx_back] = R[e]
                self.reaction_diagSigma[e, idx_back] = Sigma_hat[e]

        self.reaction_ready = True

    # ---------------- MALA / SGLD (only W) ----------------

    @torch.no_grad()
    def _mala_inner(self, X: torch.Tensor, y: torch.Tensor, f_mean: torch.Tensor, eta: float) -> float:
        # current
         with torch.enable_grad(): 
            gw, U_curr, _ = self._U_and_grad_autograd(X, y, f_mean)
            xi_w = torch.randn_like(self.W, dtype=self.dtype)
            Wp = self.W - eta * gw + math.sqrt(2.0*eta) * xi_w

            prop_finite = torch.isfinite(Wp).all(dim=2)

            # proposed
            W_saved = self.W
            self.W = Wp
            gw_p, U_prop, _ = self._U_and_grad_autograd(X, y, f_mean)
            self.W = W_saved

            mw  = self.W - eta * gw
            mpw = Wp  - eta * gw_p
            log_q_prop_given_curr = - ((Wp - mw).pow(2).sum(dim=2)) / (4.0*eta)
            log_q_curr_given_prop = - ((self.W - mpw).pow(2).sum(dim=2)) / (4.0*eta)

            log_acc = (-U_prop + U_curr) + (log_q_curr_given_prop - log_q_prop_given_curr)
            ok = torch.isfinite(log_acc) & prop_finite
            log_acc = torch.where(ok, log_acc, torch.full_like(log_acc, -float("inf")))
            u = torch.rand_like(log_acc)
            accept = (torch.log(u) < log_acc)

            if accept.any():
                self.W[accept] = Wp[accept]

            n_bad = 0
            if self.algo.kill_nan_particles:
                bad_now = ~torch.isfinite(self.W).all(dim=2)
                if bad_now.any():
                    n_bad = self._reinit_particles(bad_now)

            self._clamp_params()
            self._last_bad = int(n_bad)
            self._last_bad_prop = int((~prop_finite).sum().item())

            return float(accept.float().mean().item())

    @torch.no_grad()
    def _sgld_inner(self, X: torch.Tensor, y: torch.Tensor, f_mean: torch.Tensor, eta: float):
        gw, _, _ = self._U_and_grad_autograd(X, y, f_mean)
        self.W.add_(gw, alpha=-eta)
        self.W.add_(torch.randn_like(self.W, dtype=self.dtype), alpha=math.sqrt(2.0*eta))
        self._clamp_params()

    @torch.no_grad()
    def _update_rho_ard(self):
        if not self.ard.use_ard: return
        B = self.mdl.B
        alpha_post = self.ard.alpha0 + 0.5 * B
        ss = 0.5 * (self.W * self.W).sum(dim=1)  # (E,d)
        beta_post = torch.tensor(self.beta0, device=self.device, dtype=self.dtype) + ss
        rho_hat = alpha_post / torch.clamp(beta_post, min=torch.tensor(1e-24, dtype=self.dtype, device=self.device))
        rho_hat = torch.clamp(rho_hat, min=self.ard.rho_min, max=self.ard.rho_max)
        self.rho.mul_(1.0 - self.ard.ema).add_(rho_hat, alpha=self.ard.ema)

    @torch.no_grad()
    def _eval_heldout(self, d: int, P_eval: int, chunk: int) -> Dict[str, Any]:
        """
        Evaluate on a large held-out set using current μ (cached).
        """
        E = self.E; device = self.device
        M = len(self.sets)
        f2_sum = torch.zeros(E, device=device, dtype=self.dtype)
        v_list = torch.zeros(E, M, device=device, dtype=self.dtype) if M>0 else None
        G_list = torch.zeros(E, M, M, device=device, dtype=self.dtype) if M>0 else None

        scale = self.scale_f
        mu = self._mu.detach().to(self.dtype)  # (E,B)
        mu3 = mu.unsqueeze(2)                  # (E,B,1)

        g_base = 987654321
        step = max(1, chunk)
        for start in range(0, P_eval, step):
            n = min(step, P_eval-start)
            Xc = []
            for e in range(E):
                g = torch.Generator(device=device).manual_seed(g_base + e + start)
                Xe = (torch.randint(0,2,(n,d),generator=g,device=device,dtype=torch.int8).to(self.dtype) * 2.0 - 1.0)
                Xc.append(Xe)
            Xc = torch.stack(Xc, dim=0)  # (E,n,d)

            Z = self._batched_mm_X_Wt(Xc, self.W)   # (E,n,B)
            Phi = activation(Z, self.mdl.act)
            f = (scale * torch.bmm(Phi, mu3).squeeze(-1))  # (E,n)
            f2_sum += (f*f).sum(dim=1)

            if M>0:
                for e in range(E):
                    Ccols = [parity_character(Xc[e], S) for S in self.sets]
                    C = torch.stack(Ccols, dim=1)
                    v_list[e] += C.t().matmul(f[e])
                    G_list[e] += C.t().matmul(C)
            del Xc, Z, Phi, f

        invP = 1.0/float(P_eval)
        f2_bar = f2_sum * invP
        out = {
            'half_mse_empirical_per_exp': (0.5*f2_bar).tolist(),
            'half_mse_total_ms_per_exp': (0.5*f2_bar).tolist(),
            'half_mse_modes_per_exp': [0.0]*E,
            'half_noise_per_exp': (0.5*f2_bar).tolist(),
        }
        if M==0:
            mean_val = float((0.5*f2_bar).mean().item())
            out.update(
                half_mse_empirical=mean_val, half_mse_total_ms=mean_val,
                half_mse_modes=0.0, half_noise=mean_val,
                m_S_per_exp=[[] for _ in range(E)], m_S=[]
            ); return out

        half_modes = []; half_noise = []; half_total = []; half_emp = []; m_S_per_exp = []
        ones = torch.ones(M, device=device, dtype=self.dtype)
        for e in range(E):
            v = (v_list[e] * invP) if v_list is not None else torch.zeros(M, device=device, dtype=self.dtype)
            G = (G_list[e] * invP) if G_list is not None else torch.zeros(M, M, device=device, dtype=self.dtype)
            m_S = v; m_S_per_exp.append(m_S.detach().cpu().tolist())
            mTm = float((m_S*m_S).sum().item())
            mTGm = float(m_S.view(1,-1).matmul(G).matmul(m_S.view(-1,1)).item())
            noise = float(f2_bar[e].item()) - 2.0*mTm + mTGm
            half_modes.append(0.5*float(((1.0-m_S)**2).sum().item()))
            half_noise.append(0.5*float(noise))
            half_total.append(half_modes[-1] + half_noise[-1])
            half_emp.append(0.5*(float(f2_bar[e].item()) - 2.0*float(ones.dot(v).item())
                                 + float(ones.view(1,-1).matmul(G).matmul(ones.view(-1,1)).item())))
        m_arr = np.array(m_S_per_exp, dtype=float) if m_S_per_exp else np.zeros((E,0))
        m_mean = m_arr.mean(axis=0).tolist() if m_arr.size>0 else []
        out.update(
            half_mse_modes_per_exp=half_modes,
            half_noise_per_exp=half_noise,
            half_mse_total_ms_per_exp=half_total,
            half_mse_empirical_per_exp=half_emp,
            half_mse_modes=float(np.mean(half_modes)),
            half_noise=float(np.mean(half_noise)),
            half_mse_total_ms=float(np.mean(half_total)),
            half_mse_empirical=float(np.mean(half_emp)),
            m_S_per_exp=m_S_per_exp,
            m_S=m_mean,
        )
        return out

    # ----------------------------- main loop -----------------------------

    @torch.no_grad()
    def run(self, X: torch.Tensor, y: torch.Tensor, out_dir: str, tag: str="", dev_tag: str=""):
        os.makedirs(out_dir, exist_ok=True)
        X = X.to(self.dtype); y = y.to(self.dtype)
        E, P, _ = X.shape
        f_mean = torch.zeros(E, P, 1, device=self.device, dtype=self.dtype)
        self._f_mean_cache = f_mean
        self._y_cache = y

        hist = {
            "iter": [], "train_mse": [], "train_mse_per_exp": [], "accept": [],
            "half_mse_modes": [], "half_noise": [],
            "half_mse_total_ms": [], "half_mse_empirical": [], "m_S": [],
            "half_mse_modes_per_exp": [], "half_noise_per_exp": [],
            "half_mse_total_ms_per_exp": [], "half_mse_empirical_per_exp": [], "m_S_per_exp": [],
            "rho_min": [], "rho_max": [], "elapsed_s": [],
            "bad_prop": [], "bad_reset": [], "dtype": str(self.dtype),

            # back-reaction stats
            "reac_prefac": [], "reac_updated_it": [],
            "reac_mean_R_tilde": [], "reac_mean_Sigma_hat": [], "reac_mean_Sigma_eff": [],
            "offdiag_ratio_mean": [], "sigma_var_mean": [], "reac_delta_sigma_mean": [],
            # sanity on actual magnitudes in the gradient term
            "c2_mean": [], "c2_eff_mean": [],
        }
        t0 = time.time()

        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = (
            f"rs_aw_ard_multi_react_{tag or ts}_Ptr{P}_E{self.E}_Peval{self.algo.P_eval}_"
            f"kap{self.kappa:.3e}_N{self.mdl.N}_B{self.mdl.B}_g{self.mdl.gamma}_"
            f"act{self.mdl.act}_{'f64' if self.algo.use_float64 else 'f32'}{('_'+dev_tag) if dev_tag else ''}.json"
        )
        save_path = os.path.join(out_dir, fname)

        best_counter = 0
        for it in range(1, self.algo.outer_steps+1):
            if self.algo.reaction_use and ((it == 1) or (it % self.algo.reaction_update_every == 0)):
                self._update_reaction(X)

            # MALA/SGLD on w, using current f_mean
            acc_val = 0.0
            bad_prop_sum = 0
            bad_reset_sum = 0
            for _ in range(self.algo.inner_mala_steps):
                if self.algo.use_mala:
                    a_rate = self._mala_inner(X, y, f_mean, self.algo.step_size)
                    acc_val += a_rate
                    if self.algo.log_bad_counts:
                        bad_prop_sum += getattr(self, "_last_bad_prop", 0)
                        bad_reset_sum += getattr(self, "_last_bad", 0)
                else:
                    self._sgld_inner(X, y, f_mean, self.algo.step_size)
            if self.algo.use_mala:
                acc_val /= max(1, self.algo.inner_mala_steps)

            if self.ard.use_ard and (it % self.ard.update_every == 0):
                self._update_rho_ard()

            # Update mean field & cache μ
            f_new, mu, var_a = self._field_from_particles_stream_mu(X, y, f_mean)
            if self.algo.cg_like_update:
                f_mean = (1.0 - self.algo.field_blend) * f_mean + self.algo.field_blend * f_new
            else:
                f_mean = f_new

            self._f_mean_cache = f_mean
            self._mu = mu.to(self.dtype)
            self._var_a = var_a.to(self.dtype)

            # training MSE
            train_mse_per_e = ((y - f_mean)**2).mean(dim=(1,2))
            train_mse_mean = float(train_mse_per_e.mean().item())

            # ---- early stopping ----
            if self.algo.early_stop_enabled:
                if train_mse_mean <= self.algo.early_stop_threshold:
                    best_counter += 1
                else:
                    best_counter = 0
                if best_counter >= self.algo.early_stop_patience:
                    ev = self._eval_heldout(self.mdl.d, self.algo.P_eval, self.algo.batch_eval)
                    rhomin = float(self.rho.min().item()); rhomax = float(self.rho.max().item())

                    if self.reaction_ready:
                        mean_R = float(self.reaction_R_tilde.mean().item())
                        mean_Sigma_hat = float(self.reaction_diagSigma.mean().item())
                        sigma_eff = self.reaction_diagSigma - self.reaction_prefac * self.reaction_R_tilde
                        if self.algo.reaction_clamp_sigma_eff:
                            sigma_eff = torch.maximum(sigma_eff, torch.tensor(0.0, device=self.device, dtype=self.dtype))
                        mean_Sigma_eff = float(sigma_eff.mean().item())
                        offdiag_ratio_mean = float(self.reaction_offdiag_ratio.mean().item())
                        sigma_var_mean = float(torch.var(self.reaction_diagSigma, dim=1).mean().item())
                        mean_delta_sigma = float((self.reaction_prefac * self.reaction_R_tilde).mean().item())
                    else:
                        mean_R = mean_Sigma_hat = mean_Sigma_eff = mean_delta_sigma = offdiag_ratio_mean = sigma_var_mean = 0.0

                    # C2 diagnostics
                    self._last_c2_mean = float((Sigma := (self._mu*0 + 0)).numel()) * 0.0  # placeholder (kept for schema)
                    self._last_c2_eff_mean = self._last_c2_mean

                    hist["iter"].append(it)
                    hist["train_mse"].append(train_mse_mean)
                    hist["train_mse_per_exp"].append(train_mse_per_e.detach().cpu().tolist())
                    hist["accept"].append(acc_val if self.algo.use_mala else 0.0)
                    hist["half_mse_modes"].append(ev["half_mse_modes"])
                    hist["half_noise"].append(ev["half_noise"])
                    hist["half_mse_total_ms"].append(ev["half_mse_total_ms"])
                    hist["half_mse_empirical"].append(ev["half_mse_empirical"])
                    hist["m_S"].append(ev.get("m_S", []))
                    hist["half_mse_modes_per_exp"].append(ev["half_mse_modes_per_exp"])
                    hist["half_noise_per_exp"].append(ev["half_noise_per_exp"])
                    hist["half_mse_total_ms_per_exp"].append(ev["half_mse_total_ms_per_exp"])
                    hist["half_mse_empirical_per_exp"].append(ev["half_mse_empirical_per_exp"])
                    hist["m_S_per_exp"].append(ev.get("m_S_per_exp", [[] for _ in range(self.E)]))
                    hist["rho_min"].append(rhomin); hist["rho_max"].append(rhomax)
                    hist["elapsed_s"].append(round(time.time()-t0,2))
                    hist["bad_prop"].append(bad_prop_sum)
                    hist["bad_reset"].append(bad_reset_sum)
                    hist["reac_prefac"].append(self.reaction_prefac if self.algo.reaction_use else 0.0)
                    hist["reac_updated_it"].append(it if (self.algo.reaction_use and self.reaction_ready) else -1)
                    hist["reac_mean_R_tilde"].append(mean_R)
                    hist["reac_mean_Sigma_hat"].append(mean_Sigma_hat)
                    hist["reac_mean_Sigma_eff"].append(mean_Sigma_eff)
                    hist["offdiag_ratio_mean"].append(offdiag_ratio_mean)
                    hist["sigma_var_mean"].append(sigma_var_mean)
                    hist["reac_delta_sigma_mean"].append(mean_delta_sigma)
                    hist["c2_mean"].append(self._last_c2_mean)
                    hist["c2_eff_mean"].append(self._last_c2_eff_mean)

                    payload = {
                        "summary": {
                            "train_mse_last": hist["train_mse"][-1],
                            "accept_last": hist["accept"][-1] if hist["accept"] else None,
                            "P_eval": self.algo.P_eval,
                            "E": self.E
                        },
                        "traj": hist,
                        "config": {
                            "model": self.mdl.__dict__, "algo": self.algo.__dict__,
                            "kappa": self.kappa,
                            "ard": {
                                "alpha0": self.ard.alpha0, "beta0": self.beta0,
                                "ema": self.ard.ema, "update_every": self.ard.update_every,
                                "rho_min": self.ard.rho_min, "rho_max": self.ard.rho_max,
                                "use_ard": self.ard.use_ard
                            }
                        }
                    }
                    with open(save_path, "w") as f:
                        json.dump(payload, f, indent=2)

                    print(f"[early-stop] it={it} mean_train_mse={train_mse_mean:.6f} — stopping early.")
                    print(json.dumps({
                        "iter": it, "mean_train_mse": train_mse_mean, "accept": acc_val if self.algo.use_mala else 0.0,
                        "half_mse_total_ms": ev["half_mse_total_ms"], "half_mse_empirical": ev["half_mse_empirical"],
                        "B": self.mdl.B, "N": self.mdl.N, "gamma": self.mdl.gamma,
                        "kappa": self.kappa, "rho_min": rhomin, "rho_max": rhomax,
                        "dtype": "float64" if self.algo.use_float64 else "float32",
                        "elapsed_s": round(time.time()-t0,2),
                        "saved": save_path,
                        "reac_prefac": self.reaction_prefac if self.algo.reaction_use else 0.0,
                        "reac_mean_R_tilde": mean_R,
                        "reac_mean_Sigma_hat": mean_Sigma_hat,
                        "reac_mean_Sigma_eff": mean_Sigma_eff,
                        "offdiag_ratio_mean": offdiag_ratio_mean,
                        "sigma_var_mean": sigma_var_mean,
                        "reac_delta_sigma_mean": mean_delta_sigma,
                        "c2_mean": self._last_c2_mean,
                        "c2_eff_mean": self._last_c2_eff_mean,
                    }))
                    break
            # ---- end early stopping ----

            if it % self.algo.log_every == 0:
                ev = self._eval_heldout(self.mdl.d, self.algo.P_eval, self.algo.batch_eval)
                rhomin = float(self.rho.min().item()); rhomax = float(self.rho.max().item())

                if self.reaction_ready:
                    mean_R = float(self.reaction_R_tilde.mean().item())
                    mean_Sigma_hat = float(self.reaction_diagSigma.mean().item())
                    sigma_eff = self.reaction_diagSigma - self.reaction_prefac * self.reaction_R_tilde
                    if self.algo.reaction_clamp_sigma_eff:
                        sigma_eff = torch.maximum(sigma_eff, torch.tensor(0.0, device=self.device, dtype=self.dtype))
                    mean_Sigma_eff = float(sigma_eff.mean().item())
                    offdiag_ratio_mean = float(self.reaction_offdiag_ratio.mean().item())
                    sigma_var_mean = float(torch.var(self.reaction_diagSigma, dim=1).mean().item())
                    mean_delta_sigma = float((self.reaction_prefac * self.reaction_R_tilde).mean().item())
                else:
                    mean_R = mean_Sigma_hat = mean_Sigma_eff = mean_delta_sigma = offdiag_ratio_mean = sigma_var_mean = 0.0

                # Store C2 sanity placeholders (kept to preserve schema)
                self._last_c2_mean = float((self._mu*0).numel()) * 0.0
                self._last_c2_eff_mean = self._last_c2_mean

                hist["iter"].append(it)
                hist["train_mse"].append(train_mse_mean)
                hist["train_mse_per_exp"].append(train_mse_per_e.detach().cpu().tolist())
                hist["accept"].append(acc_val if self.algo.use_mala else 0.0)
                hist["half_mse_modes"].append(ev["half_mse_modes"])
                hist["half_noise"].append(ev["half_noise"])
                hist["half_mse_total_ms"].append(ev["half_mse_total_ms"])
                hist["half_mse_empirical"].append(ev["half_mse_empirical"])
                hist["m_S"].append(ev.get("m_S", []))
                hist["half_mse_modes_per_exp"].append(ev["half_mse_modes_per_exp"])
                hist["half_noise_per_exp"].append(ev["half_noise_per_exp"])
                hist["half_mse_total_ms_per_exp"].append(ev["half_mse_total_ms_per_exp"])
                hist["half_mse_empirical_per_exp"].append(ev["half_mse_empirical_per_exp"])
                hist["m_S_per_exp"].append(ev.get("m_S_per_exp", [[] for _ in range(self.E)]))
                hist["rho_min"].append(rhomin); hist["rho_max"].append(rhomax)
                hist["elapsed_s"].append(round(time.time()-t0,2))
                hist["bad_prop"].append(bad_prop_sum)
                hist["bad_reset"].append(bad_reset_sum)
                hist["reac_prefac"].append(self.reaction_prefac if self.algo.reaction_use else 0.0)
                hist["reac_updated_it"].append(it if (self.algo.reaction_use and self.reaction_ready) else -1)
                hist["reac_mean_R_tilde"].append(mean_R)
                hist["reac_mean_Sigma_hat"].append(mean_Sigma_hat)
                hist["reac_mean_Sigma_eff"].append(mean_Sigma_eff)
                hist["offdiag_ratio_mean"].append(offdiag_ratio_mean)
                hist["sigma_var_mean"].append(sigma_var_mean)
                hist["reac_delta_sigma_mean"].append(mean_delta_sigma)
                hist["c2_mean"].append(self._last_c2_mean)
                hist["c2_eff_mean"].append(self._last_c2_eff_mean)

                payload = {
                    "summary": {
                        "train_mse_last": hist["train_mse"][-1],
                        "accept_last": hist["accept"][-1] if hist["accept"] else None,
                        "P_eval": self.algo.P_eval,
                        "E": self.E
                    },
                    "traj": hist,
                    "config": {
                        "model": self.mdl.__dict__, "algo": self.algo.__dict__,
                        "kappa": self.kappa,
                        "ard": {
                            "alpha0": self.ard.alpha0, "beta0": self.beta0,
                            "ema": self.ard.ema, "update_every": self.ard.update_every,
                            "rho_min": self.ard.rho_min, "rho_max": self.ard.rho_max,
                            "use_ard": self.ard.use_ard
                        }
                    }
                }
                with open(save_path, "w") as f:
                    json.dump(payload, f, indent=2)
                print(json.dumps({
                    "iter": it, "mean_train_mse": train_mse_mean, "accept": acc_val if self.algo.use_mala else 0.0,
                    "half_mse_total_ms": ev["half_mse_total_ms"], "half_mse_empirical": ev["half_mse_empirical"],
                    "B": self.mdl.B, "N": self.mdl.N, "gamma": self.mdl.gamma,
                    "kappa": self.kappa, "rho_min": rhomin, "rho_max": rhomax,
                    "dtype": "float64" if self.algo.use_float64 else "float32",
                    "elapsed_s": round(time.time()-t0,2),
                    "saved": save_path,
                    "reac_prefac": self.reaction_prefac if self.algo.reaction_use else 0.0,
                    "reac_mean_R_tilde": mean_R,
                    "reac_mean_Sigma_hat": mean_Sigma_hat,
                    "reac_mean_Sigma_eff": mean_Sigma_eff,
                    "offdiag_ratio_mean": offdiag_ratio_mean,
                    "sigma_var_mean": sigma_var_mean,
                    "reac_delta_sigma_mean": mean_delta_sigma,
                    "c2_mean": self._last_c2_mean,
                    "c2_eff_mean": self._last_c2_eff_mean,
                }))

        return {"path": save_path, "traj": hist}

# ----------------------------- experiment driver ------------------------------

def build_experiment_grid(P_train_list: List[int], kappa_list: List[float], num_exp: int, base_seed: int):
    exps = []
    idx = 0
    for P in P_train_list:
        for k in kappa_list:
            data_seeds = [base_seed + idx*100000 + e for e in range(num_exp)]
            param_seeds = [base_seed + idx*100000 + 50000 + e for e in range(num_exp)]
            exps.append({
                'P': int(P),
                'kappa': float(k),
                'data_seeds': data_seeds,
                'param_seeds': param_seeds,
                'E': num_exp,
            })
            idx += 1
    return exps

def shard_experiments(exps: List[Dict[str, Any]], num_devices: int, strategy: str = "balance_P"):
    if num_devices <= 1:
        return [exps]
    if strategy == "balance_P":
        exps_sorted = sorted(exps, key=lambda e: e['P'], reverse=True)
        loads = [0 for _ in range(num_devices)]
        shards = [[] for _ in range(num_devices)]
        for e in exps_sorted:
            i = int(np.argmin(loads)); shards[i].append(e); loads[i] += e['P']
        return shards
    else:
        shards = [[] for _ in range(num_devices)]
        for i, e in enumerate(exps):
            shards[i % num_devices].append(e)
        return shards

# ----------------------------- worker ------------------------------

def worker_process(dev_id: int, shard: List[Dict[str, Any]], mdl_dict: Dict[str, Any], algo_dict: Dict[str, Any], ard_dict: Dict[str, Any],
                   out_dir: str, teacher_sets_spec: str, use_float64: bool):
    if torch.cuda.is_available():
        torch.cuda.set_device(dev_id)
        device = torch.device(f"cuda:{dev_id}")
    else:
        device = torch.device("cpu")

    mdl = Model(**mdl_dict)
    algo = Algo(**algo_dict)
    ard = ARD(**ard_dict)

    sets = [torch.tensor(s, device=device, dtype=torch.long) for s in parse_sets(teacher_sets_spec)]

    for econf in shard:
        P = econf['P']; kappa = econf['kappa']; E = econf['E']
        data_seeds = econf['data_seeds']; param_seeds = econf['param_seeds']

        dtype = torch.float64 if use_float64 else torch.float32
        X, y = generate_parity_multi(P, mdl.d, sets, E, data_seeds, device, dtype)

        solver = RSCavityExplicitMulti(mdl, algo, kappa, device, E=E, seeds_params=param_seeds, teacher_sets=sets, ard=ard)
        tag = f"P{P}_kap{kappa:.3e}"
        dev_tag = f"dev{dev_id}"
        print(f"\n===== RUN start: P={P}, kappa={kappa:.6g}, E={E}, device={device}, dtype={'float64' if algo.use_float64 else 'float32'} =====")
        result = solver.run(X, y, out_dir, tag=tag, dev_tag=dev_tag)
        print(f"===== RUN done: saved -> {result['path']} =====\n")

        del solver, X, y
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ----------------------------- main ------------------------------

if __name__ == "__main__":
    set_seed(42)

    teacher_sets_spec = "{0,1,2,3}"
    d = 35

    P_train_list = [750,500,1000]#[10000, 10, 100, 500, 750, 1000, 2133, 3666, 5000, 7500]
    kappa_list   = [5e-3]
    num_exp = 3
    base_seed = 123456

    use_float64 = False
    mdl = Model(d=d, B=512, N=512, gamma=0.5, sigma_a=1.0, sigma_w=1.0, act="relu")
    algo = Algo(
        outer_steps=75_000, inner_mala_steps=20, step_size=1e-7, use_mala=True,
        log_every=10, cg_like_update=False, field_blend=0.01,
        P_eval=100_000, batch_eval=131072,
        B_chunk=131072, B_chunk_eval=None,
        use_float64=use_float64,
        grad_clip_norm=None,
        kill_nan_particles=True,
        max_abs_w=None, max_abs_a=None,
        max_l2_w=None, max_l2_a=None,
        nan_reinit_std_scale=1.0,
        log_bad_counts=True,
        early_stop_enabled=True,
        early_stop_threshold=0.03,
        early_stop_patience=100,
        P_chunk_train=4096*8,

        # --- 1/N reaction ---
        reaction_use=False,
        reaction_mode="lowrank_gram",           # or "exact_diag"
        reaction_update_every=1,
        reaction_particle_subsample=None,       # only used in exact_diag
        reaction_lowrank_rank=128,
        reaction_lowrank_power_iters=0,
        reaction_clamp_sigma_eff=False,
    )
    alpha0 = 0.01
    beta01 = alpha0 / d
    ard = ARD(use_ard=True, alpha0=alpha0, ema=0.05, update_every=1,
              rho_min=0.0, rho_max=1e18, beta0=beta01)

    out_dir = "/home/goring/mean_field_langevin/MCMC_composite/results_d35k4hm_test/ard_a01_no1N_5e-3"
    shard_strategy = "round_robin"

    os.makedirs(out_dir, exist_ok=True)
    exps = build_experiment_grid(P_train_list, kappa_list, num_exp, base_seed)
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    shards = shard_experiments(exps, num_devices, strategy=shard_strategy)

    if num_devices > 1:
        ctx = mp.get_context("spawn")
        procs = []
        for dev_id in range(num_devices):
            p = ctx.Process(
                target=worker_process,
                args=(dev_id, shards[dev_id],
                      mdl.__dict__, algo.__dict__, ard.__dict__,
                      out_dir, teacher_sets_spec, algo.use_float64),
            )
            p.start(); procs.append(p)
        for p in procs:
            p.join()
    else:
        worker_process(0, shards[0], mdl.__dict__, algo.__dict__, ard.__dict__, out_dir, teacher_sets_spec, algo.use_float64)
