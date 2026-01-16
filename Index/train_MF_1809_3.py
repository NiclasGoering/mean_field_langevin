import os, time, math, json, random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch._dynamo as dynamo

dynamo.config.suppress_errors = True
# dynamo.config.capture_scalar_outputs = True

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

# -------- single-index Hermite teacher (batched for E exps) --------

def _hermite_he(z: torch.Tensor, n: int) -> torch.Tensor:
    """Probabilists' Hermite polynomials He_n(z): He_0=1, He_1=z, He_{n+1}=z*He_n - n*He_{n-1}."""
    if n < 0:
        raise ValueError("Hermite degree n must be >= 0")
    if n == 0:
        return torch.ones_like(z)
    if n == 1:
        return z
    He_nm1 = torch.ones_like(z)   # He_0
    He_n_  = z                    # He_1
    for k in range(1, n):
        He_np1 = z * He_n_ - k * He_nm1
        He_nm1, He_n_ = He_n_, He_np1
    return He_n_

def generate_single_index_hermite_multi(
    P: int, d: int, k: int, E: int,
    data_seeds: List[int], device, dtype,
    hermite_degree: int = 5, random_support: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate E independent datasets for a single-index model:
      X ~ N(0, I_d), choose support S of size k (random per exp by default),
      w_i = 1/sqrt(k) on S else 0, z = X @ w, y = He_p(z) (continuous labels).
    Returns:
      X ∈ ℝ[E,P,d], y ∈ ℝ[E,P,1], W ∈ ℝ[E,d] (teacher vectors per experiment).
    """
    assert len(data_seeds) == E
    Xs, ys, Ws = [], [], []
    for e in range(E):
        g = torch.Generator(device=device).manual_seed(int(data_seeds[e]))
        # support
        if k > d:
            raise ValueError("k cannot exceed d")
        if k > 0:
            if random_support:
                idx = torch.randperm(d, generator=g, device=device)[:k]
            else:
                idx = torch.arange(k, device=device)
        # teacher vector
        we = torch.zeros(d, device=device, dtype=dtype)
        if k > 0:
            we[idx] = 1.0 / math.sqrt(k)

        Xe = torch.randn(P, d, generator=g, device=device, dtype=dtype)
        ze = Xe @ we
        gz = _hermite_he(ze, hermite_degree)
        ye = gz.unsqueeze(1)  # continuous labels

        Xs.append(Xe); ys.append(ye); Ws.append(we)
    X = torch.stack(Xs, dim=0)            # (E,P,d)
    y = torch.stack(ys, dim=0)            # (E,P,1)
    W = torch.stack(Ws, dim=0)            # (E,d)
    return X, y, W

# ----------------------------- config -----------------------------

@dataclass
class Model:
    d: int = 35
    B: int = 16384
    N: int = 512          # f = (N^{1-γ}/B) * Σ a φ  == (1/N^γ) * Σ a φ if B=N
    gamma: float = 0.5
    sigma_a: float = 1.0
    sigma_w: float = 1.0
    sigma_b: float = 1.0   # std for hidden bias prior
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
    # core loop
    outer_steps: int = 2000
    step_size: float = 1e-6
    log_every: int = 10
    batch_eval: int = 262_144

    # chunking
    P_chunk_train: Optional[int] = 4096

    # safety
    grad_clip_norm: Optional[float] = None
    use_float64: bool = False
    kill_nan_particles: bool = True
    max_abs_w: Optional[float] = None
    max_abs_a: Optional[float] = None
    max_l2_w: Optional[float] = None
    max_l2_a: Optional[float] = None
    # clamps for bias
    max_abs_b: Optional[float] = None
    max_l2_b: Optional[float] = None
    nan_reinit_std_scale: float = 1.0
    log_bad_counts: bool = True

    # early stop (enhanced)
    early_stop_enabled: bool = False
    early_stop_use_mse: bool = True
    early_stop_threshold: float = 0.0
    early_stop_patience: int = 0
    early_stop_use_fp: bool = True
    early_stop_fp_epsilon: float = 0.01
    early_stop_use_rho: bool = False
    early_stop_rho_epsilon: float = 1e-3

    # independent test-MSE early stop (use FULL eval set)
    early_stop_test_mse_enabled: bool = True
    early_stop_test_mse_threshold: float = 0.01
    test_mse_check_every: int = 1
    test_mse_use_small_eval: bool = False  # we will always use the full eval set

    # optional η ∝ κ
    linear_kappa_step: bool = False

    # power-2 learning-rate decay schedule
    use_lr_decay: bool = False
    lr_start: Optional[float] = None
    lr_end: Optional[float] = None
    lr_decay_iters: int = 0

    # SGLD inner steps schedule
    K0: int = 8
    Kmin: int = 3
    K_decay: int = 50_000

    # Anderson acceleration
    use_anderson: bool = True
    aa_depth: int = 3
    aa_reg: float = 1e-8
    aa_every: int = 1

    # Eval/log slimming
    P_eval_full: int = 100_000
    P_eval_routine: int = 32_768
    save_every_logs: int = 10

# ----------------------------- core ------------------------------

class RSCavityExplicitMulti:
    def __init__(self, mdl: Model, algo: Algo, kappa: float, device: torch.device,
                 E: int, seeds_params: List[int],
                 ard: Optional[ARD] = None,
                 # teacher info
                 teacher_ws: Optional[torch.Tensor] = None,
                 hermite_degree: int = 5,
                 k_support: int = 0):
        self.mdl, self.algo, self.kappa = mdl, algo, float(kappa)
        self.device = device
        self.ard = ard or ARD()
        self.dtype = torch.float64 if algo.use_float64 else torch.float32
        self.E = int(E)
        assert len(seeds_params) == E

        # Keep teacher (for eval label generation)
        self.teacher_ws = teacher_ws  # (E,d) or None
        self.hermite_degree = int(hermite_degree)
        self.k_support = int(k_support)

        # Temperature to match "code 2"
        self.T = 2.0 * (self.kappa ** 2)

        # parameters (master in self.dtype)
        W_list, a_list, b_list = [], [], []
        for e in range(E):
            g = torch.Generator(device=device).manual_seed(int(seeds_params[e]))
            We = torch.randn(mdl.B, mdl.d, generator=g, device=device, dtype=self.dtype) * (mdl.sigma_w / math.sqrt(mdl.d))
            ae = torch.randn(mdl.B, 1, generator=g, device=device, dtype=self.dtype) * mdl.sigma_a
            be = torch.randn(mdl.B, 1, generator=g, device=device, dtype=self.dtype) * mdl.sigma_b
            W_list.append(We); a_list.append(ae); b_list.append(be)
        self.W = torch.stack(W_list, dim=0)  # (E,B,d)
        self.a = torch.stack(a_list, dim=0)  # (E,B,1)
        self.b = torch.stack(b_list, dim=0)  # (E,B,1)

        # ARD
        rho0 = (mdl.d / (mdl.sigma_w**2))
        self.rho = torch.full((E, mdl.d), rho0, device=device, dtype=self.dtype)
        self.beta0 = float(self.ard.alpha0 / rho0) if self.ard.beta0 is None else float(self.ard.beta0)

        # constants
        self.N_gamma = self.mdl.N ** self.mdl.gamma
        # unified mean-field scale valid for general B (matches Code 2 when B=N)
        self.scale_f = (self.mdl.N ** (1.0 - self.mdl.gamma)) / float(self.mdl.B)

        # fast matmuls
        try:
            torch.set_float32_matmul_precision('high')
            torch.backends.cuda.matmul.allow_tf32 = not algo.use_float64
            torch.backends.cudnn.allow_tf32 = not algo.use_float64
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

        # Autocast control
        self.autocast_enabled = (self.device.type == "cuda") and (not self.algo.use_float64)
        self._autocast_dtype = torch.bfloat16

        # Preallocated buffers for stats/grads
        self._buf_C1 = torch.empty(self.E, self.mdl.B, device=self.device, dtype=self.dtype)
        self._buf_C2 = torch.empty_like(self._buf_C1)
        self._buf_G  = torch.empty(self.E, self.mdl.B, self.mdl.d, device=self.device, dtype=self.dtype)
        self._buf_Hb = torch.empty(self.E, self.mdl.B, device=self.device, dtype=self.dtype)

        # last-step C2 stats for logging
        self._last_c2_mean = 0.0
        self._last_c2eff_mean = 0.0

        # small P heuristic toggle set in run()
        self._smallP = False

        # Held-out eval sets (pre-generate once)
        self._X_eval_small = None
        self._y_eval_small = None
        self._X_eval_full = None
        self._y_eval_full = None
        self._make_eval_sets()

        # Counters for buffered saving
        self._log_event_count = 0

        # Compile hot paths (PyTorch 2.x+)
        if hasattr(torch, "compile") and (self.device.type == "cuda"):
            try:
                self._stats_and_grads_stream = torch.compile(self._stats_and_grads_stream, mode="max-autotune", dynamic=True)
                self._field_from_particles_stream = torch.compile(self._field_from_particles_stream, mode="max-autotune", dynamic=True)
                self._sgld_inner = torch.compile(self._sgld_inner, mode="max-autotune", dynamic=True)
            except Exception as e:
                print(f"[warn] torch.compile failed, continuing without compile: {e}")

    # ---------- helpers ----------

    def _make_eval_sets(self):
        """Pre-generate and cache routine (small) eval X and y based on the Hermite teacher."""
        d = self.mdl.d
        P_small = self.algo.P_eval_routine
        g_base = 987654321
        Xs = []; Ys = []
        for e in range(self.E):
            g = torch.Generator(device=self.device).manual_seed(g_base + e)
            Xe = torch.randn(P_small, d, generator=g, device=self.device, dtype=self.dtype)
            Xs.append(Xe)
            if self.teacher_ws is not None:
                we = self.teacher_ws[e].to(self.dtype)
                ze = Xe @ we
                gz = _hermite_he(ze, self.hermite_degree)
                Ye = gz.unsqueeze(1)
                Ys.append(Ye)
        self._X_eval_small = torch.stack(Xs, dim=0)
        self._y_eval_small = torch.stack(Ys, dim=0) if Ys else None

    def _ensure_full_eval_set(self):
        if self._X_eval_full is not None:
            return
        d = self.mdl.d
        P_full = self.algo.P_eval_full
        g_base = 987654321 + 12345  # different seed stream
        Xs = []; Ys = []
        for e in range(self.E):
            g = torch.Generator(device=self.device).manual_seed(g_base + e)
            Xe = torch.randn(P_full, d, generator=g, device=self.device, dtype=self.dtype)
            Xs.append(Xe)
            if self.teacher_ws is not None:
                we = self.teacher_ws[e].to(self.dtype)
                ze = Xe @ we
                gz = _hermite_he(ze, self.hermite_degree)
                Ye = gz.unsqueeze(1)
                Ys.append(Ye)
        self._X_eval_full = torch.stack(Xs, dim=0)
        self._y_eval_full = torch.stack(Ys, dim=0) if Ys else None

    @torch.no_grad()
    def _reinit_particles(self, mask: torch.Tensor):
        if not mask.any(): return 0
        sw = self.mdl.sigma_w * self.algo.nan_reinit_std_scale
        sa = self.mdl.sigma_a * self.algo.nan_reinit_std_scale
        sb = self.mdl.sigma_b * self.algo.nan_reinit_std_scale
        total = 0
        for e in range(self.E):
            m = mask[e]
            if m.any():
                n = int(m.sum().item()); total += n
                g = torch.Generator(device=self.device).manual_seed(int(17_123 + e))
                self.W[e, m] = torch.randn(n, self.mdl.d, generator=g, device=self.device, dtype=self.dtype) * (sw / math.sqrt(self.mdl.d))
                self.a[e, m] = torch.randn(n, 1, generator=g, device=self.device, dtype=self.dtype) * sa
                self.b[e, m] = torch.randn(n, 1, generator=g, device=self.device, dtype=self.dtype) * sb
        return total

    @torch.no_grad()
    def _clamp_params(self):
        if self.algo.max_abs_w is not None:
            self.W.clamp_(-self.algo.max_abs_w, self.algo.max_abs_w)
        if self.algo.max_abs_a is not None:
            self.a.clamp_(-self.algo.max_abs_a, self.algo.max_abs_a)
        if self.algo.max_l2_w is not None:
            norms = torch.linalg.vector_norm(self.W, dim=2, keepdim=True) + 1e-12
            self.W.mul_((torch.clamp(self.algo.max_l2_w / norms, max=1.0)))
        if self.algo.max_l2_a is not None:
            norms = torch.sqrt((self.a*self.a).sum(dim=2, keepdim=True)) + 1e-12
            self.a.mul_((torch.clamp(self.algo.max_l2_a / norms, max=1.0)))
        if self.algo.max_abs_b is not None:
            self.b.clamp_(-self.algo.max_abs_b, self.algo.max_abs_b)
        if self.algo.max_l2_b is not None:
            norms = torch.sqrt((self.b*self.b).sum(dim=2, keepdim=True)) + 1e-12
            self.b.mul_((torch.clamp(self.algo.max_l2_b / norms, max=1.0)))

    @staticmethod
    def _batched_mm_X_Wt(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        return torch.bmm(X, W.transpose(1, 2))

    def _scheduled_eta(self, it: int, P: int) -> float:
        """
        Learning-rate schedule: power-2 poly-decay
        η_t = η_end + (η_start - η_end) * (1 - min(t, K)/K)^2
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
        if P <= 100:
            eta *= 0.5
        return float(eta)

    def _scheduled_K(self, it: int) -> int:
        K0, Kmin, Kd = self.algo.K0, self.algo.Kmin, self.algo.K_decay
        t = min(max(it-1, 0), Kd)
        frac = 1.0 - (t / float(Kd)) if Kd > 0 else 0.0
        K = max(Kmin, int(round(K0 * (frac*frac) + Kmin*(1-frac*frac))))
        return int(K)

    # ---------------- streaming core ----------------

    @torch.no_grad()
    def _stats_and_grads_stream(self, X: torch.Tensor, r: torch.Tensor,
                                W: torch.Tensor, a: torch.Tensor,
                                return_field: bool = False):
        """
        Returns:
          grad_w_U, grad_a_U, grad_b_U, optional f(x)
        Where:
          ∇U = [ T * prior-gradient ] + [ data-gradient with mean MSE (1/P) ]
        """
        Eexp, P, d = X.shape
        B = W.shape[1]
        invP = 1.0 / float(P)

        a_flat = a[:, :, 0]                 # (E,B)
        b_flat = self.b[:, :, 0]            # (E,B)

        s = self.scale_f
        a_scaled = a * s                    # (E,B,1)
        a_scaled_T = a_scaled.transpose(1, 2)   # (E,1,B)

        # preallocated accumulators
        C1 = self._buf_C1.zero_()           # (E,B)   accumulates Σ Φ^T r
        C2 = self._buf_C2.zero_()           # (E,B)   accumulates Σ φ^2 per particle
        G  = self._buf_G.zero_()            # (E,B,d) accumulates w-gradient
        Hb = self._buf_Hb.zero_()           # (E,B)   accumulates b-gradient (data term)
        f_acc = None
        if return_field:
            f_acc = torch.zeros(Eexp, P, device=self.device, dtype=self.dtype)

        step = self.algo.P_chunk_train or P

        ctx = torch.autocast(self.device.type, dtype=self._autocast_dtype, enabled=self.autocast_enabled)
        with ctx:
            for start in range(0, P, step):
                n = min(step, P - start)
                Xc = X[:, start:start+n, :].contiguous()
                rc = r[:, start:start+n, :].contiguous()

                Z   = self._batched_mm_X_Wt(Xc, W.contiguous())    # (E,n,B)
                Z   = Z + self.b.transpose(1, 2)                   # (E,1,B)
                Phi = activation(Z, self.mdl.act)                  # (E,n,B)

                if self.mdl.act == "relu":
                    dPhi = (Z > 0)
                elif self.mdl.act == "tanh":
                    dPhi = 1.0 - torch.tanh(Z) ** 2
                else:
                    dPhi = act_prime(Z, self.mdl.act)

                # reductions
                C1 += torch.bmm(Phi.transpose(1, 2).contiguous(), rc).squeeze(-1)  # Σ Φ^T r
                C2 += (Phi * Phi).sum(dim=1)                                      # Σ φ^2

                # data gradient wrt w for mean MSE (1/P)
                M = (rc - a_scaled_T * Phi) * dPhi * a_scaled_T                   # (E,n,B)
                G += torch.bmm(M.transpose(1, 2).contiguous(), Xc) * (-2.0 * invP)
                Hb += M.sum(dim=1).to(self.dtype) * (-2.0 * invP)

                if return_field:
                    fa = torch.bmm(Phi, a)  # (E,n,1)
                    f_acc[:, start:start+n] = (self.scale_f * fa.squeeze(-1)).to(self.dtype)

        # store means for diagnostics
        self._last_c2_mean = float(C2.mean().item())
        self._last_c2eff_mean = float(C2.mean().item())

        # GRADIENTS OF U
        term1_a_priorU = (self.T / (self.mdl.sigma_a**2)) * a_flat
        term2_a_dataU  = - 2.0 * C1 * (invP * s)
        term3_a_dataU  = + 2.0 * (C2 * (invP * (s * s))) * a_flat
        grad_a_U  = (term1_a_priorU + term2_a_dataU + term3_a_dataU).unsqueeze(2).to(self.dtype)

        grad_w_U = (G + (self.T) * W * self.rho.unsqueeze(1)).to(self.dtype)

        term1_b_priorU = (self.T / (self.mdl.sigma_b**2)) * b_flat
        term2_b_dataU  = Hb
        grad_b_U = (term1_b_priorU + term2_b_dataU).unsqueeze(2).to(self.dtype)  # (E,B,1)

        # clip
        if self.algo.grad_clip_norm is not None:
            gw2 = (grad_w_U * grad_w_U).sum(dim=2, keepdim=True)
            ga2 = (grad_a_U * grad_a_U).sum(dim=2, keepdim=True)
            gb2 = (grad_b_U * grad_b_U).sum(dim=2, keepdim=True)
            gn = torch.sqrt(gw2 + ga2 + gb2) + 1e-12
            scale = torch.clamp(self.algo.grad_clip_norm / gn, max=1.0)
            grad_w_U = grad_w_U * scale
            grad_a_U = grad_a_U * scale
            grad_b_U = grad_b_U * scale

        return grad_w_U, grad_a_U, grad_b_U, (f_acc.unsqueeze(-1) if return_field else None)

    @torch.no_grad()
    def _field_from_particles_stream(self, X: torch.Tensor) -> torch.Tensor:
        Eexp, P, _ = X.shape
        f = torch.zeros(Eexp, P, device=self.device, dtype=self.dtype)
        step = self.algo.P_chunk_train or P
        ctx = torch.autocast(self.device.type, dtype=self._autocast_dtype, enabled=self.autocast_enabled)
        with ctx:
            for start in range(0, P, step):
                n = min(step, P - start)
                Xc = X[:, start:start+n, :].contiguous()
                Z  = self._batched_mm_X_Wt(Xc, self.W.contiguous())
                Z  = Z + self.b.transpose(1, 2)
                Phi = activation(Z, self.mdl.act)
                fa = torch.bmm(Phi, self.a)
                f[:, start:start+n] = (self.scale_f * fa.squeeze(-1)).to(self.dtype)
        return f.unsqueeze(-1)

    # ---------------- Label-aware evaluation ----------------

    @torch.no_grad()
    def _eval_on_X(self, X_eval: torch.Tensor, y_eval: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Evaluate on provided X_eval (shape E,P_eval,d).
           If y_eval is provided, compute MSE vs labels.
           Otherwise, report 0.5 * E[f^2] as 'half_f2'.
        """
        Eexp, P_eval, _ = X_eval.shape
        scale = self.scale_f
        a = self.a.detach()

        # accumulators
        if y_eval is not None:
            mse_sum = torch.zeros(Eexp, device=self.device, dtype=self.dtype)
        else:
            f2_sum = torch.zeros(Eexp, device=self.device, dtype=self.dtype)

        step = max(1, self.algo.batch_eval)
        for start in range(0, P_eval, step):
            n = min(step, P_eval-start)
            Xc = X_eval[:, start:start+n, :].contiguous()
            Z = self._batched_mm_X_Wt(Xc, self.W.contiguous())
            Z = Z + self.b.transpose(1, 2)
            Phi = activation(Z, self.mdl.act)
            f = (scale * torch.bmm(Phi, a).squeeze(-1))  # (E,n)

            if y_eval is not None:
                ye = y_eval[:, start:start+n, 0].to(self.dtype)
                diff = f - ye
                mse_sum += (diff * diff).sum(dim=1)
            else:
                f2_sum += (f * f).sum(dim=1)

        invP = 1.0 / float(P_eval)
        if y_eval is not None:
            mse_per_exp = mse_sum * invP
            return {
                'mse_per_exp': mse_per_exp.tolist(),
                'mse': float(mse_per_exp.mean().item()),
                'rmse': float(torch.sqrt(mse_per_exp.mean()).item()),
            }
        else:
            half_f2_per_exp = 0.5 * (f2_sum * invP)
            return {
                'half_f2_per_exp': half_f2_per_exp.tolist(),
                'half_f2': float(half_f2_per_exp.mean().item()),
            }

    # ---------------- SGLD ----------------

    @torch.no_grad()
    def _sgld_inner(self, X: torch.Tensor, r: torch.Tensor, eta: float, return_field: bool = False):
        """One inner SGLD step. Optionally piggybacks the field on the last step."""
        gw, ga, gb, f_acc = self._stats_and_grads_stream(X, r, self.W, self.a, return_field=return_field)
        self.W.add_(gw, alpha=-eta)
        self.a.add_(ga, alpha=-eta)
        self.b.add_(gb, alpha=-eta)

        # noise std = sqrt(2*T*eta)
        nstd = math.sqrt(2.0*self.T*eta)
        self.W.add_(torch.randn_like(self.W, dtype=self.dtype), alpha=nstd)
        self.a.add_(torch.randn_like(self.a, dtype=self.dtype), alpha=nstd)
        self.b.add_(torch.randn_like(self.b, dtype=self.dtype), alpha=nstd)

        if self.algo.kill_nan_particles:
            bad_now = (~torch.isfinite(self.W).all(dim=2)) | (~torch.isfinite(self.a).all(dim=2)) | (~torch.isfinite(self.b).all(dim=2))
            if bad_now.any():
                self._reinit_particles(bad_now)

        self._clamp_params()
        return f_acc  # None unless return_field=True

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

    # ---------------- Anderson acceleration ----------------

    @torch.no_grad()
    def _anderson_update(self, f_prev: torch.Tensor, f_new: torch.Tensor,
                         aa_hist: List[torch.Tensor], g_hist: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Walker-Ni Anderson acceleration."""
        if not self.algo.use_anderson or self.algo.aa_depth <= 0:
            return f_new, aa_hist, g_hist

        r_k = (f_new - f_prev).detach().reshape(-1)
        g_k = f_new.detach().reshape(-1)

        aa_hist.append(r_k)
        g_hist.append(g_k)
        if len(aa_hist) <= 1:
            if len(aa_hist) > self.algo.aa_depth:
                aa_hist.pop(0); g_hist.pop(0)
            return f_new, aa_hist, g_hist

        m = min(self.algo.aa_depth, len(aa_hist)-1)
        R_cols = []
        G_cols = []
        for i in range(-m-1, -1):
            R_cols.append(aa_hist[i])
            G_cols.append(g_hist[i])
        R = torch.stack(R_cols, dim=1)
        dR = R[:, 1:] - R[:, :-1]
        dG = torch.stack(G_cols, dim=1)[:, 1:] - torch.stack(G_cols, dim=1)[:, :-1]

        lam = self.algo.aa_reg
        Gram = dR.T @ dR
        rhs  = dR.T @ r_k
        Gram = Gram + lam * torch.eye(Gram.shape[0], device=self.device, dtype=Gram.dtype)
        try:
            alpha = torch.linalg.solve(Gram, rhs)
        except RuntimeError:
            alpha = torch.zeros(rhs.shape, device=self.device, dtype=rhs.dtype)

        g_next = g_k - (dG @ alpha)
        f_next = g_next.reshape_as(f_new)

        if len(aa_hist) > self.algo.aa_depth + 1:
            aa_hist.pop(0); g_hist.pop(0)

        return f_next, aa_hist, g_hist

    # ---------------- Main loop ----------------

    @torch.no_grad()
    def run(self, X: torch.Tensor, y: torch.Tensor, out_dir: str, tag: str="", dev_tag: str=""):
        os.makedirs(out_dir, exist_ok=True)
        X = X.to(self.dtype); y = y.to(self.dtype)
        Eexp, P, _ = X.shape
        f_mean = torch.zeros(Eexp, P, 1, device=self.device, dtype=self.dtype)

        # small-P heuristic flag (kept from original)
        self._smallP = (P <= 100)

        hist = {
            "iter": [],
            "train_mse": [], "train_mse_per_exp": [],
            "eval_mse_small": [], "eval_mse_small_per_exp": [],
            "eval_mse_full": [], "eval_mse_full_per_exp": [],
            "rho_min": [], "rho_max": [], "elapsed_s": [],
            "c2_mean": [], "c2_eff_mean": [],
            "fp_residual": [], "fp_residual_per_exp": [],
            "rho_delta_rel": [], "eta": [], "K": [],
        }
        t0 = time.time()

        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = (
            f"rs_aw_ard_sgld_FAST_{tag or ts}_Ptr{P}_E{self.E}_Peval{self.algo.P_eval_full}_"
            f"kap{self.kappa:.3e}_T{self.T:.3e}_N{self.mdl.N}_B{self.mdl.B}_g{self.mdl.gamma}_"
            f"act{self.mdl.act}_{'f64' if self.algo.use_float64 else 'f32'}{('_'+dev_tag) if dev_tag else ''}.json"
        )
        save_path = os.path.join(out_dir, fname)

        # Early-stop counters & ARD trackers
        best_counter = 0
        prev_rho = self.rho.detach().clone()
        eps = 1e-24

        # Anderson state
        aa_hist: List[torch.Tensor] = []
        g_hist: List[torch.Tensor]  = []

        # For reduced saving frequency
        def maybe_save(payload, force=False):
            if force or (self._log_event_count % self.algo.save_every_logs == 0):
                with open(save_path, "w") as f:
                    json.dump(payload, f, indent=2)

        for it in range(1, self.algo.outer_steps+1):
            # cavity residual
            r = (y - f_mean)

            # Scheduled LR and inner steps
            eta = self._scheduled_eta(it, P)
            K = self._scheduled_K(it)

            # K inner SGLD steps (piggyback field on the last one)
            f_last = None
            for kstep in range(K):
                f_last = self._sgld_inner(X, r, eta, return_field=(kstep == K-1))

            # f_new from piggyback; fallback to explicit forward if None
            f_new = f_last if f_last is not None else self._field_from_particles_stream(X)

            # compute FP residual BEFORE blending
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

            # Anderson acceleration (outer map)
            if self.algo.use_anderson and ((it % self.algo.aa_every) == 0):
                f_next, aa_hist, g_hist = self._anderson_update(f_mean, f_new, aa_hist, g_hist)
                f_mean = f_next
            else:
                f_mean = f_new

            train_mse_per_e = ((y - f_mean)**2).mean(dim=(1,2))
            train_mse_mean = float(train_mse_per_e.mean().item())

            # -------- test-MSE early stop (FULL EVAL SET) --------
            test_mse_val = None
            if (
                self.algo.early_stop_test_mse_enabled
                and (self._X_eval_small is not None)  # small set exists -> teacher exists
                and (it % max(1, self.algo.test_mse_check_every) == 0)
            ):
                # ensure full set exists
                self._ensure_full_eval_set()
                # compute MSE on FULL held-out set with continuous labels
                ev_full = self._eval_on_X(self._X_eval_full, self._y_eval_full)
                test_mse_val = ev_full["mse"]
                if test_mse_val < self.algo.early_stop_test_mse_threshold:
                    # snapshot with full eval
                    rhomin = float(self.rho.min().item()); rhomax = float(self.rho.max().item())

                    hist["iter"].append(it)
                    hist["train_mse"].append(train_mse_mean)
                    hist["train_mse_per_exp"].append(train_mse_per_e.detach().cpu().tolist())
                    hist["eval_mse_full"].append(ev_full["mse"])  # log full-set test error
                    hist["eval_mse_full_per_exp"].append(ev_full["mse_per_exp"])
                    hist["rho_min"].append(rhomin); hist["rho_max"].append(rhomax)
                    hist["elapsed_s"].append(round(time.time()-t0,2))
                    hist["c2_mean"].append(self._last_c2_mean)
                    hist["c2_eff_mean"].append(self._last_c2eff_mean)
                    hist["fp_residual"].append(fp_residual_mean)
                    hist["fp_residual_per_exp"].append(fp_residual_per_e.detach().cpu().tolist())
                    hist["rho_delta_rel"].append(rho_delta_rel)
                    hist["eta"].append(eta); hist["K"].append(K)

                    payload = {
                        "summary": {
                            "train_mse_last": hist["train_mse"][-1],
                            "P_eval_full": self.algo.P_eval_full,
                            "E": self.E,
                            "early_stop_reason": f"test_mse_full<{self.algo.early_stop_test_mse_threshold}",
                            "test_mse_full": ev_full["mse"],
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
                            },
                            "teacher": {
                                "type": "single_index_hermite",
                                "k_support": self.k_support,
                                "hermite_degree": self.hermite_degree,
                                "support_selection": "random_per_experiment"
                            }
                        }
                    }
                    with open(save_path, "w") as f:
                        json.dump(payload, f, indent=2)
                    print(f"[early-stop:TEST] it={it} FULL-eval test_mse={test_mse_val:.6f} < {self.algo.early_stop_test_mse_threshold}. Stopping.")
                    return {"path": save_path, "traj": hist}

            # -------- legacy/other early stopping with FP + rho checks --------
            early_stopped = False
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
                    self._ensure_full_eval_set()
                    ev_full = self._eval_on_X(self._X_eval_full, self._y_eval_full)
                    rhomin = float(self.rho.min().item()); rhomax = float(self.rho.max().item())

                    hist["iter"].append(it)
                    hist["train_mse"].append(train_mse_mean)
                    hist["train_mse_per_exp"].append(train_mse_per_e.detach().cpu().tolist())
                    hist["eval_mse_full"].append(ev_full["mse"])
                    hist["eval_mse_full_per_exp"].append(ev_full["mse_per_exp"])
                    hist["rho_min"].append(rhomin); hist["rho_max"].append(rhomax)
                    hist["elapsed_s"].append(round(time.time()-t0,2))
                    hist["c2_mean"].append(self._last_c2_mean)
                    hist["c2_eff_mean"].append(self._last_c2eff_mean)
                    hist["fp_residual"].append(fp_residual_mean)
                    hist["fp_residual_per_exp"].append(fp_residual_per_e.detach().cpu().tolist())
                    hist["rho_delta_rel"].append(rho_delta_rel)
                    hist["eta"].append(eta); hist["K"].append(K)

                    payload = {
                        "summary": {
                            "train_mse_last": hist["train_mse"][-1],
                            "P_eval_full": self.algo.P_eval_full,
                            "E": self.E,
                            "early_stop_reason": "fp/rho/mse composite",
                            "test_mse_full": ev_full["mse"],
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
                            },
                            "teacher": {
                                "type": "single_index_hermite",
                                "k_support": self.k_support,
                                "hermite_degree": self.hermite_degree,
                                "support_selection": "random_per_experiment"
                            }
                        }
                    }
                    maybe_save(payload, force=True)
                    print(f"[early-stop] it={it} mean_train_mse={train_mse_mean:.6f} fp_residual={fp_residual_mean:.6e} rho_delta_rel={rho_delta_rel:.3e}.")
                    early_stopped = True

            if early_stopped:
                break

            # Routine logging/eval
            if it % self.algo.log_every == 0:
                # cheap small-set eval (if labels exist)
                ev_small = self._eval_on_X(self._X_eval_small, self._y_eval_small) if (self._y_eval_small is not None) else {"mse": float('nan'), "mse_per_exp": [float('nan')]*self.E}

                # full-set eval for accurate test error
                self._ensure_full_eval_set()
                ev_full = self._eval_on_X(self._X_eval_full, self._y_eval_full) if (self._y_eval_full is not None) else {"mse": float('nan'), "mse_per_exp": [float('nan')]*self.E}

                rhomin = float(self.rho.min().item()); rhomax = float(self.rho.max().item())

                hist["iter"].append(it)
                hist["train_mse"].append(train_mse_mean)
                hist["train_mse_per_exp"].append(train_mse_per_e.detach().cpu().tolist())
                hist["eval_mse_small"].append(ev_small["mse"])
                hist["eval_mse_small_per_exp"].append(ev_small["mse_per_exp"])
                hist["eval_mse_full"].append(ev_full["mse"])
                hist["eval_mse_full_per_exp"].append(ev_full["mse_per_exp"])
                hist["rho_min"].append(rhomin); hist["rho_max"].append(rhomax)
                hist["elapsed_s"].append(round(time.time()-t0,2))
                hist["c2_mean"].append(self._last_c2_mean)
                hist["c2_eff_mean"].append(self._last_c2eff_mean)
                hist["fp_residual"].append(fp_residual_mean)
                hist["fp_residual_per_exp"].append(fp_residual_per_e.detach().cpu().tolist())
                hist["rho_delta_rel"].append(rho_delta_rel)
                hist["eta"].append(eta); hist["K"].append(K)

                payload = {
                    "summary": {
                        "train_mse_last": hist["train_mse"][-1],
                        "P_eval_small": self.algo.P_eval_routine,
                        "P_eval_full": self.algo.P_eval_full,
                        "E": self.E,
                        "eval_mse_full": ev_full["mse"],
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
                        },
                        "teacher": {
                            "type": "single_index_hermite",
                            "k_support": self.k_support,
                            "hermite_degree": self.hermite_degree,
                            "support_selection": "random_per_experiment"
                        }
                    }
                }
                self._log_event_count += 1
                maybe_save(payload, force=False)

                print(json.dumps({
                    "iter": it,
                    "mean_train_mse": train_mse_mean,
                    "B": self.mdl.B,
                    "N": self.mdl.N,
                    "gamma": self.mdl.gamma,
                    "kappa": self.kappa,
                    "T": self.T,
                    "rho_min": rhomin,
                    "rho_max": rhomax,
                    "dtype": "float64" if self.algo.use_float64 else "float32",
                    "elapsed_s": round(time.time()-t0,2),
                    "saved": save_path,
                    "c2_mean": self._last_c2_mean,
                    "c2_eff_mean": self._last_c2eff_mean,
                    "fp_residual": fp_residual_mean,
                    "rho_delta_rel": rho_delta_rel,
                    "eta": eta,
                    "K": K,
                    "eval_mse_small": ev_small["mse"],
                    "eval_mse_full": ev_full["mse"],
                }))

        # Final FULL EVAL if not early-stopped
        self._ensure_full_eval_set()
        ev_full = self._eval_on_X(self._X_eval_full, self._y_eval_full)
        rhomin = float(self.rho.min().item()); rhomax = float(self.rho.max().item())
        hist["iter"].append(it)
        hist["train_mse"].append(float(((y - f_mean)**2).mean().item()))
        hist["train_mse_per_exp"].append(((y - f_mean)**2).mean(dim=(1,2)).detach().cpu().tolist())
        hist["eval_mse_full"].append(ev_full["mse"])
        hist["eval_mse_full_per_exp"].append(ev_full["mse_per_exp"])
        hist["rho_min"].append(rhomin); hist["rho_max"].append(rhomax)
        hist["elapsed_s"].append(round(time.time()-t0,2))
        hist["c2_mean"].append(self._last_c2_mean)
        hist["c2_eff_mean"].append(self._last_c2eff_mean)
        hist["fp_residual"].append(0.0)
        hist["fp_residual_per_exp"].append([0.0]*self.E)
        hist["rho_delta_rel"].append(0.0)
        hist["eta"].append(self._scheduled_eta(it, P))
        hist["K"].append(self._scheduled_K(it))

        payload = {
            "summary": {
                "train_mse_last": hist["train_mse"][-1],
                "P_eval_full": self.algo.P_eval_full,
                "E": self.E,
                "test_mse_full": ev_full["mse"],
            },
            "traj": hist,
            "config": {
                "model": self.mdl.__dict__,
                "algo": self.algo.__dict__,
                "kappa": self.kappa,
                "T": self.T,
                "ard": {
                    "alpha0": self.ard.alpha0,
                    "beta0": self.beta0,
                    "ema": self.ard.ema,
                    "update_every": self.ard.update_every,
                    "rho_min": self.ard.rho_min,
                    "rho_max": self.ard.rho_max,
                    "use_ard": self.ard.use_ard
                },
                "teacher": {
                    "type": "single_index_hermite",
                    "k_support": self.k_support,
                    "hermite_degree": self.hermite_degree,
                    "support_selection": "random_per_experiment"
                }
            }
        }
        with open(save_path, "w") as f:
            json.dump(payload, f, indent=2)

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
                   out_dir: str, hermite_degree: int, k_support: int):
    if torch.cuda.is_available():
        torch.cuda.set_device(dev_id)
        device = torch.device(f"cuda:{dev_id}")
    else:
        device = torch.device("cpu")

    mdl = Model(**mdl_dict)
    algo = Algo(**algo_dict)
    ard = ARD(**ard_dict)

    for econf in shard:
        P = econf['P']; kappa = econf['kappa']; E = econf['E']
        data_seeds = econf['data_seeds']; param_seeds = econf['param_seeds']

        dtype = torch.float64 if algo.use_float64 else torch.float32

        # Single-index Hermite data (continuous labels)
        X, y, W_teacher = generate_single_index_hermite_multi(
            P, mdl.d, k_support, E, data_seeds, device, dtype, hermite_degree=hermite_degree, random_support=True
        )

        solver = RSCavityExplicitMulti(
            mdl, algo, kappa, device, E=E, seeds_params=param_seeds, ard=ard,
            teacher_ws=W_teacher, hermite_degree=hermite_degree, k_support=k_support
        )
        tag = f"P{P}_kap{kappa:.3e}"
        dev_tag = f"dev{dev_id}"
        print(f"\n===== RUN start: P={P}, kappa={kappa:.6g}, T={solver.T:.6g}, E={E}, device={device}, dtype={'float64' if algo.use_float64 else 'float32'} =====")
        result = solver.run(X, y, out_dir, tag=tag, dev_tag=dev_tag)
        print(f"===== RUN done: saved -> {result['path']} =====\n")

        del solver, X, y, W_teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ----------------------------- main ------------------------------

if __name__ == "__main__":
    set_seed(42)

    # Teacher settings (single-index Hermite)
    hermite_degree = 4   # degree p
    k_support = 2        # number of active coords in the teacher vector

    d = 18

    P_train_list =  [5000]#[50,100,1000,5000,10000,25000,50000,75000]
    P_train_list = sorted(P_train_list, reverse=True)
    kappa_list   = [1e-1]
    num_exp = 4
    base_seed = 123456

    use_float64 = False
    mdl = Model(d=d, B=1024, N=1024, gamma=0.5, sigma_a=1.0, sigma_w=0.5, sigma_b=1.0, act="relu")
    algo = Algo(
        outer_steps=4_000_000,

        step_size=1e-2,
        log_every=50_000,            # less frequent logs
        batch_eval=131_072,
        P_chunk_train=131_072*2,     # big chunks for better GPU utilization

        use_float64=use_float64,
        grad_clip_norm=None,
        kill_nan_particles=True,
        nan_reinit_std_scale=1.0,
        log_bad_counts=True,

        # --- Composite early stop (unchanged) ---
        early_stop_enabled=True,
        early_stop_use_mse=True,
        early_stop_threshold=0.005,
        early_stop_patience=100,
        early_stop_use_fp=True,
        early_stop_fp_epsilon=0.005,
        early_stop_use_rho=True,
        early_stop_rho_epsilon=1e-3,

        # --- Independent TEST MSE early stop (now uses FULL held-out) ---
        early_stop_test_mse_enabled=True,
        early_stop_test_mse_threshold=0.01,
        test_mse_check_every=5000,
        test_mse_use_small_eval=False,

        # --- LR decay (power 2) ---
        use_lr_decay=True,
        lr_start=1e-2,
        lr_end=5e-4,
        lr_decay_iters=2_500_000,

        # --- Inner SGLD steps schedule ---
        K0=12,
        Kmin=2,
        K_decay=600_000,

        # --- Anderson acceleration ---
        use_anderson=True,
        aa_depth=3,
        aa_reg=1e-8,
        aa_every=1,

        # --- Eval/log slimming ---
        P_eval_full=100_000,
        P_eval_routine=32_768,
        save_every_logs=2,

        # (optional) clamps for bias if you want them at runtime
        max_abs_b=None,
        max_l2_b=None,
    )
    alpha0 = 0.1
    beta01 = alpha0 / d
    ard = ARD(use_ard=True, alpha0=alpha0, ema=0.5, update_every=1,
              rho_min=0.0, rho_max=1e18, beta0=beta01)

    out_dir = "/home/goring/mean_field_langevin/Index/results/2009_MF_d18_k2_p4_a=0.1_correct2_21/1e-1"
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
                      out_dir, hermite_degree, k_support),
            )
            p.start(); procs.append(p)
        for p in procs:
            p.join()
    else:
        worker_process(0, shards[0], mdl.__dict__, algo.__dict__, ard.__dict__, out_dir, hermite_degree, k_support)
