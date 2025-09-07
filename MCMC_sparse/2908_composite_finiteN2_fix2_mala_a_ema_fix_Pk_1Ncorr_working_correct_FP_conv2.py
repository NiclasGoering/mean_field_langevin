# rs_cavity_explicit_aw_ard_fast_reaction_fixed_ngamma_with_fp_metrics_lr_decay.py
# RS self-consistent cavity with explicit (a,w), per-particle MALA/SGLD,
# optional ARD, and finite-N (1/N) Onsager/back-reaction corrections.
#
# Langevin scaling aligned with "code 2":
#   - Temperature-coded: T = 2*kappa^2
#   - Drift uses ∇U = [T * prior-gradient] + [data-gradient (mean MSE, no kappa)]
#   - Noise std = sqrt(2*T*eta)
#   - MALA acceptance uses energy E = U/T and proposal covariances 2*T*eta I
#
# Fixes kept:
# - R_tilde uses a_bar = a / N^gamma (critical!)
# - Exact prefactor 2/N, no hidden P/kappa gains.
# - Quenched estimates from the P training samples only.
# - Two reaction modes: "exact_diag" and "lowrank_gram".
# - Logs additional sanity stats: c2_mean and c2_eff_mean.
#
# NEW in this version:
# - Quadratic (power-2) learning-rate decay with configurable start/end LRs and
#   a cutoff iteration after which the LR stays at the end value.
# - Fixed-point (FP) checks for f_mean residual and ARD precision rho stability
#   integrated into early stopping (configurable).

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
    N: int = 512          # f = N^{-γ} Σ a φ
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
    max_abs_a: Optional[float] = None
    max_l2_w: Optional[float] = None
    max_l2_a: Optional[float] = None
    nan_reinit_std_scale: float = 1.0
    log_bad_counts: bool = True

    # early stop (enhanced)
    early_stop_enabled: bool = False
    # Use separate toggles for each criterion
    early_stop_use_mse: bool = True
    early_stop_threshold: float = 0.0
    early_stop_patience: int = 0
    # FP residual on f_mean
    early_stop_use_fp: bool = True
    early_stop_fp_epsilon: float = 0.01
    # Stationarity of rho (ARD precisions)
    early_stop_use_rho: bool = False
    early_stop_rho_epsilon: float = 1e-3

    # finite-N reaction (Onsager)
    reaction_use: bool = False
    reaction_mode: str = "exact_diag"       # or "lowrank_gram"
    reaction_update_every: int = 10
    reaction_particle_subsample: Optional[int] = None

    # low-rank options (for "lowrank_gram")
    reaction_lowrank_rank: int = 32
    reaction_lowrank_power_iters: int = 0
    reaction_clamp_sigma_eff: bool = False

    # NEW: optional η ∝ κ
    linear_kappa_step: bool = False  # if True, eta = step_size * kappa

    # NEW: power-2 learning-rate decay schedule
    use_lr_decay: bool = False
    lr_start: Optional[float] = None   # if None, defaults to step_size
    lr_end: Optional[float] = None     # if None, defaults to step_size
    lr_decay_iters: int = 0            # number of outer iterations over which decay applies

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

        # Temperature to match "code 2"
        self.T = 2.0 * (self.kappa ** 2)

        # parameters
        W_list, a_list = [], []
        for e in range(E):
            g = torch.Generator(device=device).manual_seed(int(seeds_params[e]))
            We = torch.randn(mdl.B, mdl.d, generator=g, device=device, dtype=self.dtype) * (mdl.sigma_w / math.sqrt(mdl.d))
            ae = torch.randn(mdl.B, 1, generator=g, device=device, dtype=self.dtype) * mdl.sigma_a
            W_list.append(We); a_list.append(ae)
        self.W = torch.stack(W_list, dim=0)  # (E,B,d)
        self.a = torch.stack(a_list, dim=0)  # (E,B,1)

        # ARD
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
        self.reaction_prefac = 2.0 / float(self.mdl.N)  # exact 2/N
        self.reaction_ready = False
        self.reaction_R_tilde = torch.zeros(self.E, self.mdl.B, device=self.device, dtype=self.dtype)
        self.reaction_diagSigma = torch.zeros(self.E, self.mdl.B, device=self.device, dtype=self.dtype)
        self.reaction_offdiag_ratio = torch.zeros(self.E, device=self.device, dtype=self.dtype)

        # last-step C2 stats for logging
        self._last_c2_mean = 0.0
        self._last_c2eff_mean = 0.0

        # small P heuristic toggle set in run()
        self._smallP = False

    # ---------- helpers ----------

    @torch.no_grad()
    def _reinit_particles(self, mask: torch.Tensor):
        if not mask.any(): return 0
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
            norms = torch.linalg.vector_norm(self.W, dim=2, keepdim=True) + 1e-12
            self.W.mul_(torch.clamp(self.algo.max_l2_w / norms, max=1.0))
        if self.algo.max_l2_a is not None:
            norms = torch.sqrt((self.a*self.a).sum(dim=2, keepdim=True)) + 1e-12
            self.a.mul_((torch.clamp(self.algo.max_l2_a / norms, max=1.0)))

    @staticmethod
    def _batched_mm_X_Wt(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        return torch.bmm(X, W.transpose(1, 2))

    def _scheduled_eta(self, it: int, P: int) -> float:
        """Quadratic (power-2) LR schedule over outer iterations.
        If use_lr_decay is False or lr_decay_iters<=0, returns base step.
        Schedule (for t in [0, K]):
            eta_t = lr_end + (lr_start - lr_end) * (1 - t/K)^2
        and stays at lr_end for t > K.
        Also applies the small-P heuristic from the original code.
        """
        base = self.algo.step_size * (self.kappa**2 if self.algo.linear_kappa_step else 1.0)
        if self.algo.use_lr_decay and self.algo.lr_decay_iters > 0:
            lr0 = self.algo.lr_start if (self.algo.lr_start is not None) else base
            lr1 = self.algo.lr_end   if (self.algo.lr_end   is not None) else base
            t = min(max(it-1, 0), self.algo.lr_decay_iters)
            frac = 1.0 - (t / float(self.algo.lr_decay_iters))
            eta = lr1 + (lr0 - lr1) * (frac * frac)
        else:
            eta = base
        # original small-P dampening
        if P <= 100:
            eta *= 0.01
        return float(eta)

    # ---------------- streaming core ----------------

    @torch.no_grad()
    def _stats_and_grads_stream(self, X: torch.Tensor, r: torch.Tensor,
                                W: torch.Tensor, a: torch.Tensor,
                                return_field: bool = False):
        """
        Returns:
          grad_w_U, grad_a_U, E = U/T (for MALA acceptance), optional f(x)
        Where:
          ∇U = [ T * prior-gradient ] + [ data-gradient with mean MSE (1/P) ]
          E  = U/T = prior + (1/(T*P)) * quadratic-data-term
        """
        Eexp, P, d = X.shape
        B = W.shape[1]
        invP = 1.0 / float(P)
        invTP = 1.0 / (self.T * float(P))

        a_flat = a[:, :, 0]                 # (E,B)
        a_over = a / self.N_gamma           # (E,B,1)
        a_over_T = a_over.transpose(1, 2)   # (E,1,B)

        C1 = torch.zeros(Eexp, B, device=self.device, dtype=self.dtype)
        C2 = torch.zeros(Eexp, B, device=self.device, dtype=self.dtype)
        G  = torch.zeros(Eexp, B, d, device=self.device, dtype=self.dtype)
        f_acc = None
        if return_field:
            f_acc = torch.zeros(Eexp, P, device=self.device, dtype=self.dtype)

        step = self.algo.P_chunk_train or P
        for start in range(0, P, step):
            n = min(step, P - start)
            Xc = X[:, start:start+n, :]
            rc = r[:, start:start+n, :]

            Z   = self._batched_mm_X_Wt(Xc, W)    # (E,n,B)
            Phi = activation(Z, self.mdl.act)     # (E,n,B)

            if self.mdl.act == "relu":
                dPhi = (Z > 0).to(self.dtype)
            elif self.mdl.act == "tanh":
                dPhi = 1.0 - torch.tanh(Z) ** 2
            else:
                dPhi = act_prime(Z, self.mdl.act)

            # reductions
            C1 += torch.bmm(Phi.transpose(1, 2), rc).squeeze(-1)  # Σ Φ^T r
            C2 += (Phi * Phi).sum(dim=1)                          # Σ φ^2

            # data gradient w.r.t. w (mean MSE: 1/P)
            M = (rc - a_over_T * Phi) * dPhi * a_over_T
            G += torch.bmm(M.transpose(1, 2), Xc) * (-invP)

            if return_field:
                fa = torch.bmm(Phi, a)  # (E,n,1)
                f_acc[:, start:start+n] = (self.scale_f * fa.squeeze(-1))

        # reaction (optional)
        if self.algo.reaction_use and self.reaction_ready:
            delta_C2 = (float(P) * self.reaction_prefac) * self.reaction_R_tilde
            C2_eff = C2 - delta_C2
            if self.algo.reaction_clamp_sigma_eff:
                C2_eff = torch.maximum(C2_eff, torch.zeros_like(C2_eff))
        else:
            C2_eff = C2

        # store means for diagnostics
        self._last_c2_mean = float(C2.mean().item())
        self._last_c2eff_mean = float(C2_eff.mean().item())

        # ---- GRADIENTS OF U ----
        # a-grad: T * prior + data(1/P)
        term1_a_priorU = (self.T / (self.mdl.sigma_a**2)) * a_flat
        term2_a_dataU  = - C1 * (invP / self.N_gamma)
        term3_a_dataU  = (C2_eff * (invP / (self.N_gamma**2))) * a_flat
        grad_a_U  = (term1_a_priorU + term2_a_dataU + term3_a_dataU).unsqueeze(2)

        # w-grad: data + T * rho * W
        grad_w_U = G + (self.T) * W * self.rho.unsqueeze(1)

        # clip
        if self.algo.grad_clip_norm is not None:
            gw2 = (grad_w_U * grad_w_U).sum(dim=2, keepdim=True)
            ga2 = (grad_a_U * grad_a_U).sum(dim=2, keepdim=True)
            gn = torch.sqrt(gw2 + ga2) + 1e-12
            scale = torch.clamp(self.algo.grad_clip_norm / gn, max=1.0)
            grad_w_U = grad_w_U * scale
            grad_a_U = grad_a_U * scale

        # ---- ENERGY FOR MALA ACCEPTANCE: E = U/T ----
        # prior terms divide out T, so they look identical to standard prior energy
        prior_w_overT = 0.5 * (self.rho.unsqueeze(1) * (W * W)).sum(dim=2)
        prior_a_overT = 0.5 * (1.0 / (self.mdl.sigma_a**2)) * (a_flat ** 2)
        data_overT    = ( - (a_flat / self.N_gamma) * C1 + 0.5 * (a_flat*a_flat / (self.N_gamma**2)) * C2_eff ) * (invTP)
        energy_overT  = prior_w_overT + prior_a_overT + data_overT

        return grad_w_U.to(self.dtype), grad_a_U.to(self.dtype), energy_overT.to(self.dtype), (f_acc if return_field else None)

    @torch.no_grad()
    def _field_from_particles_stream(self, X: torch.Tensor) -> torch.Tensor:
        Eexp, P, _ = X.shape
        f = torch.zeros(Eexp, P, device=self.device, dtype=self.dtype)
        step = self.algo.P_chunk_train or P
        for start in range(0, P, step):
            n = min(step, P - start)
            Xc = X[:, start:start+n, :]
            Z  = self._batched_mm_X_Wt(Xc, self.W)
            Phi = activation(Z, self.mdl.act)
            fa = torch.bmm(Phi, self.a)
            f[:, start:start+n] = (self.scale_f * fa.squeeze(-1))
        return f.unsqueeze(-1)

    # ---------------- Reaction (1/N) ----------------

    @torch.no_grad()
    def _update_reaction(self, X: torch.Tensor):
        """Quenched back-reaction using ONLY training X.
        Build H_hat = (1/P) Φ^T Φ and Sigma_hat = diag(H_hat),
        then R_tilde = (H_hat ∘ H_hat) @ (a_bar^2), with a_bar = a / N^gamma.
        """
        if not self.algo.reaction_use:
            self.reaction_ready = False
            return

        Eexp, P, d = X.shape
        B = self.mdl.B

        # optional subsample (exact_diag only)
        if (self.algo.reaction_mode == "exact_diag" and
            self.algo.reaction_particle_subsample is not None and
            self.algo.reaction_particle_subsample < B):
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

        # accumulate H and diag sum
        H = torch.zeros(self.E, B_eff, B_eff, device=self.device, dtype=self.dtype)
        diag_sum = torch.zeros(self.E, B_eff, device=self.device, dtype=self.dtype)

        step = self.algo.P_chunk_train or P
        for start in range(0, P, step):
            n = min(step, P - start)
            Xc = X[:, start:start+n, :]
            Z  = self._batched_mm_X_Wt(Xc, W_use)      # (E,n,B_eff)
            Phi = activation(Z, self.mdl.act)
            H += torch.bmm(Phi.transpose(1, 2), Phi)
            diag_sum += (Phi * Phi).sum(dim=1)

        H_hat = H / float(P)                # (E,B_eff,B_eff)
        Sigma_hat = diag_sum / float(P)     # (E,B_eff)

        # IMPORTANT: use a_bar = a / N^gamma
        a_bar = a_use[:, :, 0] / self.N_gamma
        a2bar = a_bar * a_bar               # (E,B_eff)

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
            H_sq = H_hat * H_hat                      # (E,B_eff,B_eff)
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

    # ---------------- MALA / SGLD ----------------

    @torch.no_grad()
    def _mala_inner(self, X: torch.Tensor, r: torch.Tensor, eta: float) -> float:
        # grads of U, and E = U/T for acceptance
        gw, ga, E_curr_overT, _ = self._stats_and_grads_stream(X, r, self.W, self.a, return_field=False)

        xi_w = torch.randn_like(self.W, dtype=self.dtype)
        xi_a = torch.randn_like(self.a, dtype=self.dtype)

        # proposal with covariance 2*T*eta I and mean θ - η ∇U
        Wp = self.W - eta * gw + math.sqrt(2.0*self.T*eta) * xi_w
        ap = self.a - eta * ga + math.sqrt(2.0*self.T*eta) * xi_a

        prop_finite = torch.isfinite(Wp).all(dim=2) & torch.isfinite(ap).all(dim=2)

        W_saved, a_saved = self.W, self.a
        self.W, self.a = Wp, ap
        gw_p, ga_p, E_prop_overT, _ = self._stats_and_grads_stream(X, r, self.W, self.a, return_field=False)
        self.W, self.a = W_saved, a_saved

        # means of forward/backward kernels
        mw  = self.W - eta * gw;  ma  = self.a - eta * ga
        mpw = Wp - eta * gw_p;    mpa = ap  - eta * ga_p

        # proposal log-densities with variance 2*T*eta
        denom = (4.0*self.T*eta)
        log_q_prop_given_curr = - ( (Wp - mw).pow(2).sum(dim=2) + (ap - ma).pow(2).sum(dim=2) ) / denom
        log_q_curr_given_prop = - ( (self.W - mpw).pow(2).sum(dim=2) + (self.a - mpa).pow(2).sum(dim=2) ) / denom

        # target energy difference uses E = U/T
        log_acc = (-E_prop_overT + E_curr_overT) + (log_q_curr_given_prop - log_q_prop_given_curr)
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
        # grads of U, no acceptance; noise std = sqrt(2*T*eta)
        gw, ga, _, _ = self._stats_and_grads_stream(X, r, self.W, self.a, return_field=False)
        self.W.add_(gw, alpha=-eta)
        self.a.add_(ga, alpha=-eta)
        self.W.add_(torch.randn_like(self.W, dtype=self.dtype), alpha=math.sqrt(2.0*self.T*eta))
        self.a.add_(torch.randn_like(self.a, dtype=self.dtype), alpha=math.sqrt(2.0*self.T*eta))
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
        Eexp = self.E; device = self.device
        M = len(self.sets)
        f2_sum = torch.zeros(Eexp, device=device, dtype=self.dtype)
        v_list = torch.zeros(Eexp, M, device=device, dtype=self.dtype) if M>0 else None
        G_list = torch.zeros(Eexp, M, M, device=device, dtype=self.dtype) if M>0 else None

        scale = self.scale_f
        a = self.a.detach()

        g_base = 987654321
        step = max(1, chunk)
        for start in range(0, P_eval, step):
            n = min(step, P_eval-start)
            Xc = []
            for e in range(Eexp):
                g = torch.Generator(device=device).manual_seed(g_base + e + start)
                Xe = (torch.randint(0,2,(n,d),generator=g,device=device,dtype=torch.int8).to(self.dtype) * 2.0 - 1.0)
                Xc.append(Xe)
            Xc = torch.stack(Xc, dim=0)

            Z = self._batched_mm_X_Wt(Xc, self.W)
            Phi = activation(Z, self.mdl.act)
            f = (scale * torch.bmm(Phi, a).squeeze(-1))
            f2_sum += (f*f).sum(dim=1)

            if M>0:
                for e in range(Eexp):
                    Ccols = [parity_character(Xc[e], S) for S in self.sets]
                    C = torch.stack(Ccols, dim=1)
                    v_list[e] += C.t().matmul(f[e])
                    G_list[e] += C.t().matmul(C)
            del Xc, Z, Phi, f

        invP_eval = 1.0/float(P_eval)
        f2_bar = f2_sum * invP_eval
        out = {
            'half_mse_empirical_per_exp': (0.5*f2_bar).tolist(),
            'half_mse_total_ms_per_exp': (0.5*f2_bar).tolist(),
            'half_mse_modes_per_exp': [0.0]*Eexp,
            'half_noise_per_exp': (0.5*f2_bar).tolist(),
        }
        if M==0:
            mean_val = float((0.5*f2_bar).mean().item())
            out.update(
                half_mse_empirical=mean_val, half_mse_total_ms=mean_val,
                half_mse_modes=0.0, half_noise=mean_val,
                m_S_per_exp=[[] for _ in range(Eexp)], m_S=[]
            ); return out

        half_modes = []; half_noise = []; half_total = []; half_emp = []; m_S_per_exp = []
        ones = torch.ones(M, device=device, dtype=self.dtype)
        for e in range(Eexp):
            v = v_list[e] * invP_eval; G = G_list[e] * invP_eval
            m_S = v; m_S_per_exp.append(m_S.detach().cpu().tolist())
            mTm = float((m_S*m_S).sum().item())
            mTGm = float(m_S.view(1,-1).matmul(G).matmul(m_S.view(-1,1)).item())
            noise = float(f2_bar[e].item()) - 2.0*mTm + mTGm
            half_modes.append(0.5*float(((1.0-m_S)**2).sum().item()))
            half_noise.append(0.5*float(noise))
            half_total.append(half_modes[-1] + half_noise[-1])
            half_emp.append(0.5*(float(f2_bar[e].item()) - 2.0*float(ones.dot(v).item())
                                 + float(ones.view(1,-1).matmul(G).matmul(ones.view(-1,1)).item())))
        m_arr = np.array(m_S_per_exp, dtype=float) if m_S_per_exp else np.zeros((Eexp,0))
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
        Eexp, P, _ = X.shape
        f_mean = torch.zeros(Eexp, P, 1, device=self.device, dtype=self.dtype)

        # small-P heuristic flag (kept from original)
        self._smallP = (P <= 100)

        hist = {
            "iter": [], "train_mse": [], "train_mse_per_exp": [], "accept": [],
            "half_mse_modes": [], "half_noise": [],
            "half_mse_total_ms": [], "half_mse_empirical": [], "m_S": [],
            "half_mse_modes_per_exp": [], "half_noise_per_exp": [],
            "half_mse_total_ms_per_exp": [], "half_mse_empirical_per_exp": [], "m_S_per_exp": [],
            "rho_min": [], "rho_max": [], "elapsed_s": [],
            "bad_prop": [], "bad_reset": [], "dtype": str(self.dtype),
            "reac_prefac": [], "reac_updated_it": [],
            "reac_mean_R_tilde": [], "reac_mean_Sigma_hat": [], "reac_mean_Sigma_eff": [],
            "offdiag_ratio_mean": [], "sigma_var_mean": [], "reac_delta_sigma_mean": [],
            "c2_mean": [], "c2_eff_mean": [],
            "fp_residual": [], "fp_residual_per_exp": [],
            "rho_delta_rel": [],
            # NEW
            "eta": [],
        }
        t0 = time.time()

        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = (
            f"rs_aw_ard_multi_react_{tag or ts}_Ptr{P}_E{self.E}_Peval{self.algo.P_eval}_"
            f"kap{self.kappa:.3e}_T{self.T:.3e}_N{self.mdl.N}_B{self.mdl.B}_g{self.mdl.gamma}_"
            f"act{self.mdl.act}_{'f64' if self.algo.use_float64 else 'f32'}{('_'+dev_tag) if dev_tag else ''}.json"
        )
        save_path = os.path.join(out_dir, fname)

        best_counter = 0
        prev_rho = self.rho.detach().clone()
        eps = 1e-24

        for it in range(1, self.algo.outer_steps+1):
            # Update Onsager reaction occasionally
            if self.algo.reaction_use and ((it == 1) or (it % self.algo.reaction_update_every == 0)):
                self._update_reaction(X)

            # cavity residual
            r = (y - f_mean)

            # Scheduled LR for this outer iteration
            eta = self._scheduled_eta(it, P)

            acc_val = 0.0
            bad_prop_sum = 0
            bad_reset_sum = 0
            for _ in range(self.algo.inner_mala_steps):
                if self.algo.use_mala:
                    a_rate = self._mala_inner(X, r, eta)
                    acc_val += a_rate
                    if self.algo.log_bad_counts:
                        bad_prop_sum += getattr(self, "_last_bad_prop", 0)
                        bad_reset_sum += getattr(self, "_last_bad", 0)
                else:
                    self._sgld_inner(X, r, eta)
            if self.algo.use_mala:
                acc_val /= max(1, self.algo.inner_mala_steps)

            # compute f_new and FP residual BEFORE blending
            f_new = self._field_from_particles_stream(X)
            diff = (f_new - f_mean)
            fp_residual_per_e = torch.sqrt(torch.mean(diff*diff, dim=(1,2)))
            fp_residual_mean = float(fp_residual_per_e.mean().item())

            # ARD update and rho change metric
            if self.ard.use_ard and (it % self.ard.update_every == 0):
                self._update_rho_ard()
                drho = self.rho - prev_rho
                num = torch.linalg.vector_norm(drho).item()
                den = torch.linalg.vector_norm(prev_rho).item()
                rho_delta_rel = float(num / (den + eps))
                prev_rho = self.rho.detach().clone()
            else:
                rho_delta_rel = 0.0

            # update f_mean (Picard/CG-like)
            if self.algo.cg_like_update:
                f_mean = (1.0 - self.algo.field_blend) * f_mean + self.algo.field_blend * f_new
            else:
                f_mean = f_new

            train_mse_per_e = ((y - f_mean)**2).mean(dim=(1,2))
            train_mse_mean = float(train_mse_per_e.mean().item())

            # -------- early stopping with FP + rho checks --------
            if self.algo.early_stop_enabled:
                cond_mse = (not self.algo.early_stop_use_mse) or (train_mse_mean <= self.algo.early_stop_threshold)
                cond_fp  = (not self.algo.early_stop_use_fp)  or (fp_residual_mean <= self.algo.early_stop_fp_epsilon)
                cond_rho = (not self.algo.early_stop_use_rho) or (rho_delta_rel <= self.algo.early_stop_rho_epsilon)
                criteria_ok = cond_mse and cond_fp and cond_rho
                if criteria_ok:
                    best_counter += 1
                else:
                    best_counter = 0
                if best_counter >= self.algo.early_stop_patience:
                    ev = self._eval_heldout(self.mdl.d, self.algo.P_eval, self.algo.batch_eval)
                    rhomin = float(self.rho.min().item()); rhomax = float(self.rho.max().item())

                    if self.reaction_ready:
                        mean_R = float(self.reaction_R_tilde.mean().item())
                        mean_Sigma_hat = float(self.reaction_diagSigma.mean().item())
                        mean_delta_sigma = float((self.reaction_prefac * self.reaction_R_tilde).mean().item())
                        sigma_eff = self.reaction_diagSigma - self.reaction_prefac * self.reaction_R_tilde
                        if self.algo.reaction_clamp_sigma_eff:
                            sigma_eff = torch.maximum(sigma_eff, torch.tensor(0.0, device=self.device, dtype=self.dtype))
                        mean_Sigma_eff = float(sigma_eff.mean().item())
                        offdiag_ratio_mean = float(self.reaction_offdiag_ratio.mean().item())
                        sigma_var_mean = float(torch.var(self.reaction_diagSigma, dim=1).mean().item())
                    else:
                        mean_R = mean_Sigma_hat = mean_Sigma_eff = mean_delta_sigma = offdiag_ratio_mean = sigma_var_mean = 0.0

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
                    hist["c2_eff_mean"].append(self._last_c2eff_mean)
                    hist["fp_residual"].append(fp_residual_mean)
                    hist["fp_residual_per_exp"].append(fp_residual_per_e.detach().cpu().tolist())
                    hist["rho_delta_rel"].append(rho_delta_rel)
                    hist["eta"].append(eta)

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
                            "kappa": self.kappa, "T": self.T,
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

                    print(f"[early-stop] it={it} mean_train_mse={train_mse_mean:.6f} fp_residual={fp_residual_mean:.6e} rho_delta_rel={rho_delta_rel:.3e} — stopping early.")
                    break
            # -------- end early stopping --------

            if it % self.algo.log_every == 0:
                ev = self._eval_heldout(self.mdl.d, self.algo.P_eval, self.algo.batch_eval)
                rhomin = float(self.rho.min().item()); rhomax = float(self.rho.max().item())

                if self.reaction_ready:
                    mean_R = float(self.reaction_R_tilde.mean().item())
                    mean_Sigma_hat = float(self.reaction_diagSigma.mean().item())
                    mean_delta_sigma = float((self.reaction_prefac * self.reaction_R_tilde).mean().item())
                    sigma_eff = self.reaction_diagSigma - self.reaction_prefac * self.reaction_R_tilde
                    if self.algo.reaction_clamp_sigma_eff:
                        sigma_eff = torch.maximum(sigma_eff, torch.tensor(0.0, device=self.device, dtype=self.dtype))
                    mean_Sigma_eff = float(sigma_eff.mean().item())
                    offdiag_ratio_mean = float(self.reaction_offdiag_ratio.mean().item())
                    sigma_var_mean = float(torch.var(self.reaction_diagSigma, dim=1).mean().item())
                else:
                    mean_R = mean_Sigma_hat = mean_Sigma_eff = mean_delta_sigma = offdiag_ratio_mean = sigma_var_mean = 0.0

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
                hist["c2_eff_mean"].append(self._last_c2eff_mean)
                hist["fp_residual"].append(fp_residual_mean)
                hist["fp_residual_per_exp"].append(fp_residual_per_e.detach().cpu().tolist())
                hist["rho_delta_rel"].append(rho_delta_rel)
                hist["eta"].append(eta)

                payload = {
                    "summary": {
                        "train_mse_last": hist["train_mse"][-1],
                        "accept_last": hist["accept"][-1] if self.algo.use_mala else None,
                        "P_eval": self.algo.P_eval,
                        "E": self.E
                    },
                    "traj": hist,
                    "config": {
                        "model": self.mdl.__dict__, "algo": self.algo.__dict__,
                        "kappa": self.kappa, "T": self.T,
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
                    "B": self.mdl.B, "N": self.mdl.N, "gamma": self.mdl.gamma,
                    "kappa": self.kappa, "T": self.T,
                    "rho_min": rhomin, "rho_max": rhomax,
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
                    "c2_eff_mean": self._last_c2eff_mean,
                    # NEW key logs
                    "fp_residual": fp_residual_mean,
                    "rho_delta_rel": rho_delta_rel,
                    "eta": eta,
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
        print(f"\n===== RUN start: P={P}, kappa={kappa:.6g}, T={solver.T:.6g}, E={E}, device={device}, dtype={'float64' if algo.use_float64 else 'float32'} =====")
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

    P_train_list =[500,1000,10000, 10, 100, 750, 2133, 3666, 5000, 7500]
    kappa_list   =  [ 5e-2, 5e-4]
    num_exp = 3
    base_seed = 123456

    use_float64 = False
    mdl = Model(d=d, B=512, N=512, gamma=0.5, sigma_a=1.0, sigma_w=1.0, act="relu")
    algo = Algo(
        outer_steps=1000000,
        inner_mala_steps=25,
        step_size=1e-2,             # base step (used if decay disabled or lr_start/lr_end omitted)
        use_mala=False,
        log_every=500,
        cg_like_update=True,
        field_blend=0.9,
        P_eval=100_000,
        batch_eval=131072,
        B_chunk=131072,
        B_chunk_eval=None,
        use_float64=use_float64,
        grad_clip_norm=None,
        kill_nan_particles=True,
        max_abs_w=None, max_abs_a=None,
        max_l2_w=None, max_l2_a=None,
        nan_reinit_std_scale=1.0,
        log_bad_counts=True,

        # --- Early stop with FP + rho criteria ---
        early_stop_enabled=True,
        early_stop_use_mse=True,
        early_stop_threshold=0.0003,
        early_stop_patience=100,
        early_stop_use_fp=True,
        early_stop_fp_epsilon=0.001,
        early_stop_use_rho=True,
        early_stop_rho_epsilon=1e-3,

        P_chunk_train=4096*8,

        # --- 1/N reaction (leave off by default here) ---
        reaction_use=False,
        reaction_mode="lowrank_gram",
        reaction_update_every=1,
        reaction_particle_subsample=None,
        reaction_lowrank_rank=128,
        reaction_lowrank_power_iters=0,
        reaction_clamp_sigma_eff=False,

        # NEW
        linear_kappa_step=False,

        # --- NEW LR decay (power 2) ---
        use_lr_decay=True,
        lr_start=1e-2,          # start LR at iter 1
        lr_end=5e-4,            # end LR after lr_decay_iters
        lr_decay_iters=200_000, # after this many outer iters, LR stays at lr_end
    )
    alpha0 = 0.01
    beta01 = alpha0 / d
    ard = ARD(use_ard=True, alpha0=alpha0, ema=0.25, update_every=1,
              rho_min=0.0, rho_max=1e18, beta0=beta01)

    out_dir = "/home/goring/mean_field_langevin/MCMC_composite/results_d35k4hm_paper_final_conv2_long/4"
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
