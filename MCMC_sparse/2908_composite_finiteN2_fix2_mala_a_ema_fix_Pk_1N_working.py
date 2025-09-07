# rs_cavity_explicit_aw_ard_fast_with_reaction.py
# RS self-consistent cavity with explicit (a,w), per-particle MALA/SGLD,
# and Automatic Relevance Determination (ARD) prior on w (diagonal precisions ρ_j).
#
# This version ADDS optional finite-N (1/N) reaction/Onsager corrections with a flag.
# It keeps the quenched-disorder setting: all statistics (Σ, χ, reaction) are
# estimated ONLY from the P training samples (no extra randomness).
#
# Key additions vs your rs_cavity_explicit_aw_ard_fast.py:
# - Algo has new flags: reaction_use (bool), reaction_lambda (float),
#   reaction_update_every (int), reaction_particle_subsample (Optional[int]).
# - We compute the feature Gram Ĥ = (1/P) Φ^T Φ for the CURRENT particles (per replica),
#   using streaming over P, then build the reaction vector R̃_j = Σ_b a_b^2 * (Ĥ_{jb})^2.
#   The effective self-energy uses Σ̂_eff = Σ̂ - prefac * R̃, where prefac = (2/B) * reaction_lambda.
#   (This is the discrete, quenched analogue; it matches dimensions of C2 in the code.)
# - We thread Σ̂_eff into the data term by replacing C2 with C2_eff := C2 - P * prefac * R̃
#   in both the energy and gradient-on-a. We treat R̃ as a fixed environment term during
#   the inner sampler steps (no backprop through R̃), and refresh it every
#   `reaction_update_every` outer steps.
# - We save detailed back-reaction stats into the training trajectory.
#
# NOTE on scales: This implementation uses a practical, dimensionally consistent
# discrete correction that matches the C2 term used in your energy. If you want
# to experiment with alternative scalings (e.g., include an explicit N/B factor),
# adjust `reaction_lambda` accordingly. Setting reaction_lambda=0 reproduces the
# baseline (no reaction). The default prefactor here is prefac = (2/B) * reaction_lambda.

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
    """Parity character; supports X of shape (P,d) or (E,P,d)."""
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
    Returns X ∈ ℝ[E,P,d], y ∈ ℝ[E,P,1].
    """
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
    B: int = 16384       # number of particles
    N: int = 512         # network N in f = N^{-γ} Σ a φ
    gamma: float = 0.5
    sigma_a: float = 1.0
    sigma_w: float = 1.0
    act: str = "relu"

@dataclass
class ARD:
    use_ard: bool = True
    alpha0: float = 1e-2        # weak Gamma prior shape
    ema: float = 0.25           # ρ ← (1-ema) ρ + ema * ρ_post
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

    # optional chunking hooks
    B_chunk: Optional[int] = None
    B_chunk_eval: Optional[int] = None
    P_chunk_train: Optional[int] = 4096  # chunk over P during training

    # safety knobs
    grad_clip_norm: Optional[float] = None
    use_float64: bool = False
    kill_nan_particles: bool = True
    max_abs_w: Optional[float] = None
    max_abs_a: Optional[float] = None
    max_l2_w: Optional[float] = None
    max_l2_a: Optional[float] = None
    nan_reinit_std_scale: float = 1.0
    log_bad_counts: bool = True

    # early stop (mean across replicas)
    early_stop_enabled: bool = False
    early_stop_threshold: float = 0.0
    early_stop_patience: int = 0

    # ---- NEW: finite-N reaction (Onsager) options ----
    reaction_use: bool = False              # turn 1/N correction on/off
    reaction_lambda: float = 1.0            # multiplicative strength (prefac = (2/B)*lambda)
    reaction_update_every: int = 10         # recompute reaction terms every K outer steps
    reaction_particle_subsample: Optional[int] = None  # if set, use a subset of particles when building Ĥ

# ----------------------------- core (multi-replica E) ------------------------------

class RSCavityExplicitMulti:
    def __init__(self, mdl: Model, algo: Algo, kappa: float, device: torch.device,
                 E: int,
                 seeds_params: List[int],
                 teacher_sets: Optional[List[torch.Tensor]] = None,
                 ard: Optional[ARD] = None):
        self.mdl, self.algo, self.kappa = mdl, algo, float(kappa)
        self.device = device
        self.sets = teacher_sets or []
        self.ard = ard or ARD()
        self.dtype = torch.float64 if algo.use_float64 else torch.float32
        self.E = int(E)
        assert len(seeds_params) == E

        # particles (E,B,d) and (E,B,1) with independent seeds per replica
        W_list, a_list = [], []
        for e in range(E):
            g = torch.Generator(device=device).manual_seed(int(seeds_params[e]))
            We = torch.randn(mdl.B, mdl.d, generator=g, device=device, dtype=self.dtype) * (mdl.sigma_w / math.sqrt(mdl.d))
            ae = torch.randn(mdl.B, 1, generator=g, device=device, dtype=self.dtype) * mdl.sigma_a
            W_list.append(We)
            a_list.append(ae)
        self.W = torch.stack(W_list, dim=0)  # (E,B,d)
        self.a = torch.stack(a_list, dim=0)  # (E,B,1)

        # ARD precisions ρ_j per replica: start isotropic to match old prior mass d/σ_w^2
        rho0 = (mdl.d / (mdl.sigma_w**2))
        self.rho = torch.full((E, mdl.d), rho0, device=device, dtype=self.dtype)
        self.beta0 = float(self.ard.alpha0 / rho0) if self.ard.beta0 is None else float(self.ard.beta0)

        # precompute constants
        self.N_gamma = self.mdl.N ** self.mdl.gamma
        self.scale_f = (self.mdl.N ** (1.0 - self.mdl.gamma)) / float(self.mdl.B)
        try:
            torch.set_float32_matmul_precision('high')
            torch.backends.cuda.matmul.allow_tf32 = not algo.use_float64
            torch.backends.cudnn.allow_tf32 = not algo.use_float64
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

        # Reaction storage (per replica): updated every K outer steps
        self.reaction_ready = False
        self.reaction_prefac = (2.0 / float(self.mdl.B)) * float(self.algo.reaction_lambda)
        self.reaction_R = torch.zeros(self.E, self.mdl.B, device=self.device, dtype=self.dtype)  # R̃_j per particle
        self.reaction_diagSigma = torch.zeros(self.E, self.mdl.B, device=self.device, dtype=self.dtype)  # Σ̂_j = (1/P)Σ φ^2

        # Optional compile hooks — leave disabled by default (torch.compile can be finicky)
        # if hasattr(torch, "compile"):
        #     try:
        #         self._stats_and_grads_stream = torch.compile(self._stats_and_grads_stream, dynamic=False, fullgraph=False)
        #         self._field_from_particles_stream = torch.compile(self._field_from_particles_stream, dynamic=False, fullgraph=False)
        #     except Exception:
        #         pass

    # ---------- helpers: sanitize / reinit / clipping ----------

    @torch.no_grad()
    def _reinit_particles(self, mask: torch.Tensor):
        """Reinitialize selected (E,B) particles from the prior (same variance)."""
        if not mask.any():
            return 0
        sw = self.mdl.sigma_w * self.algo.nan_reinit_std_scale
        sa = self.mdl.sigma_a * self.algo.nan_reinit_std_scale
        total = 0
        for e in range(self.E):
            m = mask[e]
            if m.any():
                n = int(m.sum().item()); total += n
                g = torch.Generator(device=self.device).manual_seed(int(17_123 + e))
                self.W[e, m] = torch.randn(n, self.mdl.d, generator=g, device=self.device, dtype=self.dtype) * (sw / math.sqrt(self.mdl.d))
                self.a[e, m] = torch.randn(n, 1, generator=g, device=self.device, dtype=self.dtype) * sa
        return total

    @torch.no_grad()
    def _clamp_params(self):
        if self.algo.max_abs_w is not None:
            self.W.clamp_(-self.algo.max_abs_w, self.algo.max_abs_w)
        if self.algo.max_abs_a is not None:
            self.a.clamp_(-self.algo.max_abs_a, self.algo.max_abs_a)
        if self.algo.max_l2_w is not None:
            norms = torch.linalg.vector_norm(self.W, dim=2, keepdim=True) + 1e-12  # (E,B,1)
            scale = torch.clamp(self.algo.max_l2_w / norms, max=1.0)
            self.W.mul_(scale)
        if self.algo.max_l2_a is not None:
            norms = torch.sqrt((self.a*self.a).sum(dim=2, keepdim=True)) + 1e-12   # (E,B,1)
            scale = torch.clamp(self.algo.max_l2_a / norms, max=1.0)
            self.a.mul_(scale)

    # ---------- low-level ops ----------

    @staticmethod
    def _batched_mm_X_Wt(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        # X: (E,P,d), W: (E,B,d) -> (E,P,B)
        return torch.bmm(X, W.transpose(1, 2))

    # ---------------- streaming kernels (fast path) ----------------

    @torch.no_grad()
    def _stats_and_grads_stream(self, X: torch.Tensor, r: torch.Tensor,
                                W: torch.Tensor, a: torch.Tensor,
                                return_field: bool = False):
        """
        Streaming over P:
          - Computes reductions for grad_a (C1, C2) and grad_w (G)
          - Returns (grad_w, grad_a, energy_per_particle, optional f)
        Shapes:
          X: (E,P,d), r: (E,P,1), W: (E,B,d), a: (E,B,1)
        """
        E, P, d = X.shape
        B = W.shape[1]
        inv_kappa2P = 1.0 / (self.kappa**2 * float(P))
        a_flat = a[:, :, 0]                 # (E,B)
        a_over = a / self.N_gamma           # (E,B,1)
        a_over_T = a_over.transpose(1, 2)   # (E,1,B)

        C1 = torch.zeros(E, B, device=self.device, dtype=self.dtype)   # Σ_p Phi * r
        C2 = torch.zeros(E, B, device=self.device, dtype=self.dtype)   # Σ_p Phi^2
        G  = torch.zeros(E, B, d, device=self.device, dtype=self.dtype)
        f_acc = None
        if return_field:
            f_acc = torch.zeros(E, P, device=self.device, dtype=self.dtype)

        step = self.algo.P_chunk_train or P
        for start in range(0, P, step):
            n = min(step, P - start)
            Xc = X[:, start:start+n, :]           # (E,n,d)
            rc = r[:, start:start+n, :]           # (E,n,1)

            Z   = self._batched_mm_X_Wt(Xc, W)    # (E,n,B)
            Phi = activation(Z, self.mdl.act)     # (E,n,B)

            if self.mdl.act == "relu":
                dPhi = (Z > 0).to(self.dtype)
            elif self.mdl.act == "tanh":
                dPhi = 1.0 - torch.tanh(Z) ** 2
            else:
                dPhi = act_prime(Z, self.mdl.act)

            # C1 += sum_p Phi * r
            Cr = torch.bmm(Phi.transpose(1, 2), rc).squeeze(-1)  # (E,B,1)->(E,B)
            C1 += Cr
            # C2 += sum_p Phi^2
            C2 += (Phi * Phi).sum(dim=1)

            # M = (r - a/N * Phi) * dPhi * a/N   -> (E,n,B)
            M = (rc - a_over_T * Phi) * dPhi * a_over_T
            # G += sum_p M * X -> (E,B,d)
            G += torch.bmm(M.transpose(1, 2), Xc) * (-inv_kappa2P)

            if return_field:
                # f_chunk = scale * Phi @ a  -> (E,n,1) -> store into f_acc
                fa = torch.bmm(Phi, a)  # (E,n,1)
                f_acc[:, start:start+n] = (self.scale_f * fa.squeeze(-1))

        # ---- 1/N reaction: adjust C2 with precomputed R̃ (kept fixed during inner steps) ----
        if self.algo.reaction_use and self.reaction_ready:
            # C2_eff = C2 - P * prefac * R̃
            C2_eff = C2 - (float(P) * self.reaction_prefac) * self.reaction_R
        else:
            C2_eff = C2

        # a-grad terms (using C2_eff)
        term1_a = (1.0 / (self.mdl.sigma_a**2)) * a_flat
        term2_a = - C1 * (1.0 / (self.kappa**2 * float(P) * self.N_gamma))
        term3_a = (C2_eff * (1.0 / (self.kappa**2 * float(P) * (self.N_gamma**2)))) * a_flat
        grad_a  = (term1_a + term2_a + term3_a).unsqueeze(2)  # (E,B,1)

        # w-grad (already scaled with -1/(kappa^2 P)). We DO NOT backprop through reaction_R.
        grad_w = G + W * self.rho.unsqueeze(1)

        # Grad clipping
        if self.algo.grad_clip_norm is not None:
            gw2 = (grad_w * grad_w).sum(dim=2, keepdim=True)
            ga2 = (grad_a * grad_a).sum(dim=2, keepdim=True)
            gn = torch.sqrt(gw2 + ga2) + 1e-12
            scale = torch.clamp(self.algo.grad_clip_norm / gn, max=1.0)
            grad_w = grad_w * scale
            grad_a = grad_a * scale

        # Energies per particle (E,B) using the same reductions (C2_eff)
        prior_w = 0.5 * (self.rho.unsqueeze(1) * (W * W)).sum(dim=2)       # (E,B)
        prior_a = 0.5 * (1.0 / (self.mdl.sigma_a**2)) * (a_flat ** 2)      # (E,B)
        data = ( - (a_flat / self.N_gamma) * C1 + 0.5 * (a_flat*a_flat / (self.N_gamma**2)) * C2_eff ) * (1.0 / (self.kappa**2 * float(P)))
        energy = (prior_w + prior_a + data)                                # (E,B)

        return grad_w.to(self.dtype), grad_a.to(self.dtype), energy.to(self.dtype), (f_acc if return_field else None)

    @torch.no_grad()
    def _field_from_particles_stream(self, X: torch.Tensor) -> torch.Tensor:
        """Streaming f(X) over P. Returns (E,P,1)."""
        E, P, _ = X.shape
        f = torch.zeros(E, P, device=self.device, dtype=self.dtype)
        step = self.algo.P_chunk_train or P
        for start in range(0, P, step):
            n = min(step, P - start)
            Xc = X[:, start:start+n, :]
            Z  = self._batched_mm_X_Wt(Xc, self.W)             # (E,n,B)
            Phi = activation(Z, self.mdl.act)                  # (E,n,B)
            fa = torch.bmm(Phi, self.a)                        # (E,n,1)
            f[:, start:start+n] = (self.scale_f * fa.squeeze(-1))
        return f.unsqueeze(-1)

    # ---------------- Reaction (1/N) computation ----------------

    @torch.no_grad()
    def _update_reaction(self, X: torch.Tensor):
        """Compute quenched back-reaction terms using ONLY the training set X.
        Builds Ĥ = (1/P) Φ^T Φ per replica, then R̃_j = Σ_b a_b^2 * (Ĥ_{jb})^2.
        Stores R̃ (E,B) and diagonal Σ̂ (E,B) for logging.
        Optionally subsample particles to reduce cost.
        """
        if not self.algo.reaction_use:
            self.reaction_ready = False
            return

        E, P, d = X.shape
        B = self.mdl.B

        # Optional particle subsample
        if self.algo.reaction_particle_subsample is not None and self.algo.reaction_particle_subsample < B:
            idx = torch.randperm(B, device=self.device)[: self.algo.reaction_particle_subsample]
            W_use = self.W[:, idx]
            a_use = self.a[:, idx]
            B_eff = int(idx.numel())
            idx_back = idx
        else:
            W_use = self.W
            a_use = self.a
            B_eff = B
            idx_back = None

        # Accumulate H = Φ^T Φ over P, then normalize by P to get Ĥ
        H = torch.zeros(self.E, B_eff, B_eff, device=self.device, dtype=self.dtype)
        diag_sum = torch.zeros(self.E, B_eff, device=self.device, dtype=self.dtype)  # Σ φ^2 (for Σ̂ diag)

        step = self.algo.P_chunk_train or P
        for start in range(0, P, step):
            n = min(step, P - start)
            Xc = X[:, start:start+n, :]                            # (E,n,d)
            Z  = self._batched_mm_X_Wt(Xc, W_use)                  # (E,n,B_eff)
            Phi = activation(Z, self.mdl.act)                      # (E,n,B_eff)
            # H += Φ^T Φ (batched)
            H += torch.bmm(Phi.transpose(1, 2), Phi)
            # track diagonal Σ φ^2
            diag_sum += (Phi * Phi).sum(dim=1)

        H_hat = H / float(P)                  # (E,B_eff,B_eff)
        Sigma_hat = diag_sum / float(P)       # (E,B_eff)

        # R̃ = (Ĥ ∘ Ĥ) @ (a^2)   (∘ is element-wise square)
        a2 = (a_use[:, :, 0] ** 2)            # (E,B_eff)
        H_sq = H_hat * H_hat                  # (E,B_eff,B_eff)
        R = torch.bmm(H_sq, a2.unsqueeze(-1)).squeeze(-1)   # (E,B_eff)

        # Save back into full-sized buffers
        if idx_back is None:
            self.reaction_R = R
            self.reaction_diagSigma = Sigma_hat
        else:
            # Zero then scatter
            self.reaction_R.zero_(); self.reaction_diagSigma.zero_()
            for e in range(self.E):
                self.reaction_R[e, idx_back] = R[e]
                self.reaction_diagSigma[e, idx_back] = Sigma_hat[e]

        self.reaction_ready = True

    # ---------------- MALA / SGLD inner steps ----------------

    @torch.no_grad()
    def _mala_inner(self, X: torch.Tensor, r: torch.Tensor, eta: float) -> float:
        # current grads + energy in ONE streamed pass
        gw, ga, E_curr, _ = self._stats_and_grads_stream(X, r, self.W, self.a, return_field=False)

        # propose
        xi_w = torch.randn_like(self.W, dtype=self.dtype)
        xi_a = torch.randn_like(self.a, dtype=self.dtype)
        Wp = self.W - eta * gw + math.sqrt(2.0*eta) * xi_w
        ap = self.a - eta * ga + math.sqrt(2.0*eta) * xi_a

        prop_finite = torch.isfinite(Wp).all(dim=2) & torch.isfinite(ap).all(dim=2)

        # Temporarily switch to proposed to compute proposal grads & energy ONCE
        W_saved, a_saved = self.W, self.a
        self.W, self.a = Wp, ap
        gw_p, ga_p, E_prop, _ = self._stats_and_grads_stream(X, r, self.W, self.a, return_field=False)
        self.W, self.a = W_saved, a_saved

        # MALA densities
        mw  = self.W - eta * gw;  ma  = self.a - eta * ga
        mpw = Wp - eta * gw_p;    mpa = ap  - eta * ga_p
        log_q_prop_given_curr = - ( (Wp - mw).pow(2).sum(dim=2) + (ap - ma).pow(2).sum(dim=2) ) / (4.0*eta)
        log_q_curr_given_prop = - ( (self.W - mpw).pow(2).sum(dim=2) + (self.a - mpa).pow(2).sum(dim=2) ) / (4.0*eta)

        log_acc = (-E_prop + E_curr) + (log_q_curr_given_prop - log_q_prop_given_curr)
        ok = torch.isfinite(log_acc) & prop_finite
        log_acc = torch.where(ok, log_acc, torch.full_like(log_acc, -float("inf")))
        u = torch.rand_like(log_acc)
        accept = (torch.log(u) < log_acc)

        if accept.any():
            self.W[accept] = Wp[accept]
            self.a[accept] = ap[accept]

        n_bad = 0
        if self.algo.kill_nan_particles:
            bad_now = ~torch.isfinite(self.W).all(dim=2) | ~torch.isfinite(self.a).all(dim=2)
            if bad_now.any():
                n_bad = self._reinit_particles(bad_now)

        self._clamp_params()
        self._last_bad = int(n_bad)
        self._last_bad_prop = int((~prop_finite).sum().item())

        return float(accept.float().mean().item())

    @torch.no_grad()
    def _sgld_inner(self, X: torch.Tensor, r: torch.Tensor, eta: float):
        # Streamed grads (ignore energy)
        gw, ga, _, _ = self._stats_and_grads_stream(X, r, self.W, self.a, return_field=False)
        self.W.add_(gw, alpha=-eta)
        self.a.add_(ga, alpha=-eta)
        self.W.add_(torch.randn_like(self.W, dtype=self.dtype), alpha=math.sqrt(2.0*eta))
        self.a.add_(torch.randn_like(self.a, dtype=self.dtype), alpha=math.sqrt(2.0*eta))
        self._clamp_params()

    @torch.no_grad()
    def _update_rho_ard(self):
        if not self.ard.use_ard:
            return
        B = self.mdl.B
        alpha_post = self.ard.alpha0 + 0.5 * B
        ss = 0.5 * (self.W * self.W).sum(dim=1)  # (E,d)
        beta_post = torch.tensor(self.beta0, device=self.device, dtype=self.dtype) + ss
        rho_hat = alpha_post / torch.clamp(beta_post, min=torch.tensor(1e-24, dtype=self.dtype, device=self.device))
        rho_hat = torch.clamp(rho_hat, min=self.ard.rho_min, max=self.ard.rho_max)
        self.rho.mul_(1.0 - self.ard.ema).add_(rho_hat, alpha=self.ard.ema)

    @torch.no_grad()
    def _eval_heldout(self, d: int, P_eval: int, chunk: int) -> Dict[str, Any]:
        """Evaluation with streaming over P, unchanged numerics (held-out, not used in reaction)."""
        E = self.E
        device = self.device
        M = len(self.sets)
        f2_sum = torch.zeros(E, device=device, dtype=self.dtype)
        v_list = torch.zeros(E, M, device=device, dtype=self.dtype) if M>0 else None
        G_list = torch.zeros(E, M, M, device=device, dtype=self.dtype) if M>0 else None

        scale = self.scale_f
        a = self.a.detach()

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

            Z = self._batched_mm_X_Wt(Xc, self.W)                 # (E,n,B)
            Phi = activation(Z, self.mdl.act)                      # (E,n,B)
            f = (scale * torch.bmm(Phi, a).squeeze(-1))            # (E,n)
            f2_sum += (f*f).sum(dim=1)

            if M>0:
                for e in range(E):
                    Ccols = [parity_character(Xc[e], S) for S in self.sets]
                    C = torch.stack(Ccols, dim=1)  # (n,M)
                    v_list[e] += C.t().matmul(f[e])        # (M,)
                    G_list[e] += C.t().matmul(C)           # (M,M)
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
                half_mse_empirical=mean_val,
                half_mse_total_ms=mean_val,
                half_mse_modes=0.0,
                half_noise=mean_val,
                m_S_per_exp=[[] for _ in range(E)],
                m_S=[]
            )
            return out

        # compute MS decomposition per replica
        half_modes = []
        half_noise = []
        half_total = []
        half_emp = []
        m_S_per_exp = []
        ones = torch.ones(M, device=device, dtype=self.dtype)
        for e in range(E):
            v = v_list[e] * invP
            G = G_list[e] * invP
            m_S = v
            m_S_per_exp.append(m_S.detach().cpu().tolist())
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

    @torch.no_grad()
    def run(self, X: torch.Tensor, y: torch.Tensor, out_dir: str, tag: str="", dev_tag: str=""):
        os.makedirs(out_dir, exist_ok=True)
        X = X.to(self.dtype); y = y.to(self.dtype)
        E, P, _ = X.shape
        f_mean = torch.zeros(E, P, 1, device=self.device, dtype=self.dtype)

        hist = {
            "iter": [], "train_mse": [], "train_mse_per_exp": [], "accept": [],
            "half_mse_modes": [], "half_noise": [],
            "half_mse_total_ms": [], "half_mse_empirical": [], "m_S": [],
            "half_mse_modes_per_exp": [], "half_noise_per_exp": [],
            "half_mse_total_ms_per_exp": [], "half_mse_empirical_per_exp": [], "m_S_per_exp": [],
            "rho_min": [], "rho_max": [], "elapsed_s": [],
            "bad_prop": [], "bad_reset": [], "dtype": str(self.dtype),
            # ---- NEW: back-reaction stats ----
            "reac_prefac": [],
            "reac_updated_it": [],
            "reac_mean_R": [], "reac_max_R": [],
            "reac_mean_Sigma_hat": [], "reac_mean_Sigma_eff": [],
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
            # Refresh reaction terms on schedule (quenched: use ONLY training X)
            if self.algo.reaction_use and ((it == 1) or (it % self.algo.reaction_update_every == 0)):
                self._update_reaction(X)

            r = (y - f_mean)

            acc_val = 0.0
            bad_prop_sum = 0
            bad_reset_sum = 0
            for _ in range(self.algo.inner_mala_steps):
                if self.algo.use_mala:
                    a_rate = self._mala_inner(X, r, self.algo.step_size)
                    acc_val += a_rate
                    if self.algo.log_bad_counts:
                        bad_prop_sum += getattr(self, "_last_bad_prop", 0)
                        bad_reset_sum += getattr(self, "_last_bad", 0)
                else:
                    self._sgld_inner(X, r, self.algo.step_size)
            if self.algo.use_mala:
                acc_val /= max(1, self.algo.inner_mala_steps)

            if self.ard.use_ard and (it % self.ard.update_every == 0):
                self._update_rho_ard()

            f_new = self._field_from_particles_stream(X)
            f_mean = (1.0 - self.algo.field_blend) * f_mean + self.algo.field_blend * f_new if self.algo.cg_like_update else f_new

            train_mse_per_e = ((y - f_mean)**2).mean(dim=(1,2))  # (E,)
            train_mse_mean = float(train_mse_per_e.mean().item())

            # -------- early stopping (consecutive <= threshold) --------
            if self.algo.early_stop_enabled:
                if train_mse_mean <= self.algo.early_stop_threshold:
                    best_counter += 1
                else:
                    best_counter = 0
                if best_counter >= self.algo.early_stop_patience:
                    ev = self._eval_heldout(self.mdl.d, self.algo.P_eval, self.algo.batch_eval)
                    rhomin = float(self.rho.min().item())
                    rhomax = float(self.rho.max().item())

                    # Reaction stats (means over all replicas/particles)
                    mean_R = float(self.reaction_R.mean().item()) if self.reaction_ready else 0.0
                    max_R = float(self.reaction_R.max().item()) if self.reaction_ready else 0.0
                    mean_Sigma_hat = float(self.reaction_diagSigma.mean().item()) if self.reaction_ready else 0.0
                    mean_Sigma_eff = float((self.reaction_diagSigma - self.reaction_prefac * self.reaction_R).mean().item()) if self.reaction_ready else mean_Sigma_hat

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

                    # back-reaction logging
                    hist["reac_prefac"].append(self.reaction_prefac if self.algo.reaction_use else 0.0)
                    hist["reac_updated_it"].append(it if (self.algo.reaction_use and self.reaction_ready) else -1)
                    hist["reac_mean_R"].append(mean_R)
                    hist["reac_max_R"].append(max_R)
                    hist["reac_mean_Sigma_hat"].append(mean_Sigma_hat)
                    hist["reac_mean_Sigma_eff"].append(mean_Sigma_eff)

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
                        "reac_mean_R": mean_R,
                        "reac_mean_Sigma_hat": mean_Sigma_hat,
                        "reac_mean_Sigma_eff": mean_Sigma_eff,
                    }))
                    break
            # -------- end early stopping --------

            if it % self.algo.log_every == 0:
                ev = self._eval_heldout(self.mdl.d, self.algo.P_eval, self.algo.batch_eval)
                rhomin = float(self.rho.min().item())
                rhomax = float(self.rho.max().item())

                # Reaction stats (means over all replicas/particles)
                mean_R = float(self.reaction_R.mean().item()) if self.reaction_ready else 0.0
                max_R = float(self.reaction_R.max().item()) if self.reaction_ready else 0.0
                mean_Sigma_hat = float(self.reaction_diagSigma.mean().item()) if self.reaction_ready else 0.0
                mean_Sigma_eff = float((self.reaction_diagSigma - self.reaction_prefac * self.reaction_R).mean().item()) if self.reaction_ready else mean_Sigma_hat

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

                # back-reaction logging
                hist["reac_prefac"].append(self.reaction_prefac if self.algo.reaction_use else 0.0)
                hist["reac_updated_it"].append(it if (self.algo.reaction_use and self.reaction_ready) else -1)
                hist["reac_mean_R"].append(mean_R)
                hist["reac_max_R"].append(max_R)
                hist["reac_mean_Sigma_hat"].append(mean_Sigma_hat)
                hist["reac_mean_Sigma_eff"].append(mean_Sigma_eff)

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
                    "reac_mean_R": mean_R,
                    "reac_mean_Sigma_hat": mean_Sigma_hat,
                    "reac_mean_Sigma_eff": mean_Sigma_eff,
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
        # greedy bin packing by P (proxy for cost); larger P first
        exps_sorted = sorted(exps, key=lambda e: e['P'], reverse=True)
        loads = [0 for _ in range(num_devices)]
        shards = [[] for _ in range(num_devices)]
        for e in exps_sorted:
            i = int(np.argmin(loads))
            shards[i].append(e)
            loads[i] += e['P']
        return shards
    else:  # round robin
        shards = [[] for _ in range(num_devices)]
        for i, e in enumerate(exps):
            shards[i % num_devices].append(e)
        return shards

# ----------------------------- worker ------------------------------

def worker_process(dev_id: int, shard: List[Dict[str, Any]], mdl_dict: Dict[str, Any], algo_dict: Dict[str, Any], ard_dict: Dict[str, Any],
                   out_dir: str, teacher_sets_spec: str, use_float64: bool):
    # set device
    if torch.cuda.is_available():
        torch.cuda.set_device(dev_id)
        device = torch.device(f"cuda:{dev_id}")
    else:
        device = torch.device("cpu")

    # rebuild configs
    mdl = Model(**mdl_dict)
    algo = Algo(**algo_dict)
    ard = ARD(**ard_dict)

    # teacher
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

    # ====== user-configurable block ======
    teacher_sets_spec = "{0,1,2,3}"
    d = 35

    P_train_list = [10000, 10, 100, 500, 750, 1000, 2133, 20000, 3666, 5000, 7500]
    kappa_list   = [7.5e-3]
    num_exp = 3
    base_seed = 123456

    use_float64 = False
    mdl = Model(d=d, B=512, N=512, gamma=0.5, sigma_a=1.0, sigma_w=1.0, act="relu")
    algo = Algo(
        outer_steps=25_000, inner_mala_steps=200, step_size=7e-7, use_mala=True,
        log_every=50, cg_like_update=False, field_blend=0.8,
        P_eval=50_000, batch_eval=4096*16,
        B_chunk=2048*16, B_chunk_eval=None,
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
        # --- NEW: turn on 1/N reaction ---
        reaction_use=True,
        reaction_lambda=1.0,              # tune if needed
        reaction_update_every=1,
        reaction_particle_subsample=256, # set (e.g., 256) to speed up Ĥ build
    )
    # Gentle ARD (as discussed):
    alpha0 = 1e-6
    beta01 = alpha0 / d
    ard = ARD(use_ard=False, alpha0=alpha0, ema=0.01, update_every=1,
              rho_min=1e-12, rho_max=1e12, beta0=beta01)

    out_dir = "/home/goring/mean_field_langevin/MCMC_sparse/results/d35_k4_3108_a1grid_fix_nomala_1N_"
    shard_strategy = "round_robin"

    # ====== launcher ======
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
