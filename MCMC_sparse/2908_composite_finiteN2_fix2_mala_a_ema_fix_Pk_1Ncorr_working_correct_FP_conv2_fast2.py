import os, time, math, json, random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
# import torch._dynamo as dynamo
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
    Xs, ys = [], []
    for e in range(E):
        g = torch.Generator(device=device).manual_seed(int(data_seeds[e]))
        Xe = (torch.randint(0,2,(P,d),generator=g,device=device,dtype=torch.int8).to(dtype) * 2.0 - 1.0)
        Ccols = [parity_character(Xe, S) for S in sets]
        Ce = torch.stack(Ccols, dim=1) if Ccols else torch.zeros(P,0,device=device, dtype=dtype)
        ye = Ce.sum(dim=1, keepdim=True)
        Xs.append(Xe); ys.append(ye)
    X = torch.stack(Xs, dim=0)
    y = torch.stack(ys, dim=0)
    return X, y

# ----------------------------- config -----------------------------

@dataclass
class Model:
    d: int = 35
    B: int = 16384
    N: int = 512          # f = (N^{1-γ}/B) * Σ a φ  == (1/N^γ) * Σ a φ if B=N
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

    # NEW: independent test-MSE early stop (immediate)
    early_stop_test_mse_enabled: bool = True
    early_stop_test_mse_threshold: float = 0.01
    test_mse_check_every: int = 1          # check every outer iter by default
    test_mse_use_small_eval: bool = True   # use the small eval set for speed

    # NEW: optional η ∝ κ
    linear_kappa_step: bool = False  # if True, eta = step_size * kappa

    # NEW: power-2 learning-rate decay schedule
    use_lr_decay: bool = False
    lr_start: Optional[float] = None   # if None, defaults to step_size
    lr_end: Optional[float] = None     # if None, defaults to step_size
    lr_decay_iters: int = 0            # number of outer iterations over which decay applies

    # --------- NEW: SGLD inner steps schedule ----------
    K0: int = 8           # starting inner steps
    Kmin: int = 3         # min inner steps
    K_decay: int = 50_000 # iterations to decay K0 -> Kmin

    # --------- NEW: Anderson acceleration --------------
    use_anderson: bool = True
    aa_depth: int = 3
    aa_reg: float = 1e-8
    aa_every: int = 1     # apply AA every N outer steps

    # --------- NEW: Eval/log slimming ------------------
    P_eval_full: int = 100_000         # big eval set (run at ES/final)
    P_eval_routine: int = 32_768       # cheap routine eval set
    save_every_logs: int = 10          # write JSON every N log events

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

        # parameters (master in self.dtype)
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
        # [FIX] unified mean-field scale valid for general B (matches Code 2 when B=N)
        self.scale_f = (self.mdl.N ** (1.0 - self.mdl.gamma)) / float(self.mdl.B)

        # fast matmuls
        try:
            torch.set_float32_matmul_precision('high')
            torch.backends.cuda.matmul.allow_tf32 = not algo.use_float64
            torch.backends.cudnn.allow_tf32 = not algo.use_float64
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

        # ---- Autocast control ----
        self.autocast_enabled = (self.device.type == "cuda") and (not self.algo.use_float64)
        self._autocast_dtype = torch.bfloat16

        # ---- Preallocated buffers for stats/grads ----
        self._buf_C1 = torch.empty(self.E, self.mdl.B, device=self.device, dtype=self.dtype)
        self._buf_C2 = torch.empty_like(self._buf_C1)
        self._buf_G  = torch.empty(self.E, self.mdl.B, self.mdl.d, device=self.device, dtype=self.dtype)

        # last-step C2 stats for logging
        self._last_c2_mean = 0.0
        self._last_c2eff_mean = 0.0

        # small P heuristic toggle set in run()
        self._smallP = False

        # ---- Cheap held-out eval set (pre-generate once) ----
        self._X_eval_small = None
        self._y_eval_small = None  # labels for small eval set (for test-MSE stop)
        self._X_eval_full = None
        self._y_eval_full = None   # labels for full eval set (optional)
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
        """Pre-generate and cache routine (small) eval X (and y if teacher sets available).
        Defer full eval X until needed."""
        d = self.mdl.d
        P_small = self.algo.P_eval_routine
        g_base = 987654321
        Xs = []; Ys = []
        for e in range(self.E):
            g = torch.Generator(device=self.device).manual_seed(g_base + e)
            Xe = (torch.randint(0,2,(P_small,d),generator=g,device=self.device,dtype=torch.int8).to(self.dtype) * 2.0 - 1.0)
            Xs.append(Xe)
            if len(self.sets) > 0:
                Ccols = [parity_character(Xe, S) for S in self.sets]
                Ce = torch.stack(Ccols, dim=1) if Ccols else torch.zeros(P_small,0,device=self.device, dtype=self.dtype)
                Ye = Ce.sum(dim=1, keepdim=True)
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
            Xe = (torch.randint(0,2,(P_full,d),generator=g,device=self.device,dtype=torch.int8).to(self.dtype) * 2.0 - 1.0)
            Xs.append(Xe)
            if len(self.sets) > 0:
                Ccols = [parity_character(Xe, S) for S in self.sets]
                Ce = torch.stack(Ccols, dim=1) if Ccols else torch.zeros(P_full,0,device=self.device, dtype=self.dtype)
                Ye = Ce.sum(dim=1, keepdim=True)
                Ys.append(Ye)
        self._X_eval_full = torch.stack(Xs, dim=0)
        self._y_eval_full = torch.stack(Ys, dim=0) if Ys else None

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
        """
        [FIX] Learning-rate schedule now explicitly matches Code 2's poly-decay (power=2):
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
          grad_w_U, grad_a_U, optional f(x)
        Where:
          ∇U = [ T * prior-gradient ] + [ data-gradient with mean MSE (1/P) ]  # [FIX] matches PyTorch MSE (no 1/2)
        """
        Eexp, P, d = X.shape
        B = W.shape[1]
        invP = 1.0 / float(P)

        a_flat = a[:, :, 0]                 # (E,B)

        # [FIX] Use unified scale s = (N^{1-γ}/B) consistently (general B), instead of 1/N^γ.
        s = self.scale_f                    # scalar float
        a_scaled = a * s                    # (E,B,1)
        a_scaled_T = a_scaled.transpose(1, 2)   # (E,1,B)

        # preallocated accumulators
        C1 = self._buf_C1.zero_()           # (E,B)   accumulates Σ Φ^T r
        C2 = self._buf_C2.zero_()           # (E,B)   accumulates Σ φ^2 per particle
        G  = self._buf_G.zero_()            # (E,B,d) accumulates w-gradient
        f_acc = None
        if return_field:
            f_acc = torch.zeros(Eexp, P, device=self.device, dtype=self.dtype)

        step = self.algo.P_chunk_train or P

        # autocast for heavy blocks
        ctx = torch.autocast(self.device.type, dtype=self._autocast_dtype, enabled=self.autocast_enabled)
        with ctx:
            for start in range(0, P, step):
                n = min(step, P - start)
                Xc = X[:, start:start+n, :].contiguous()
                rc = r[:, start:start+n, :].contiguous()

                Z   = self._batched_mm_X_Wt(Xc, W.contiguous())    # (E,n,B)
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

                # [FIX] Data gradient wrt w for mean MSE (1/P), with factor 2 and general-B scaling s
                # M corresponds to (rc - s*Phi*a) * φ'(Z) * (s*a)
                M = (rc - a_scaled_T * Phi) * dPhi * a_scaled_T                   # (E,n,B)
                G += torch.bmm(M.transpose(1, 2).contiguous(), Xc) * (-2.0 * invP)

                if return_field:
                    fa = torch.bmm(Phi, a)  # (E,n,1)
                    f_acc[:, start:start+n] = (self.scale_f * fa.squeeze(-1)).to(self.dtype)

        # store means for diagnostics
        self._last_c2_mean = float(C2.mean().item())
        self._last_c2eff_mean = float(C2.mean().item())  # no reaction => same

        # ---- GRADIENTS OF U ----
        term1_a_priorU = (self.T / (self.mdl.sigma_a**2)) * a_flat
        # [FIX] mean MSE (1/P) => factor 2 in data terms; use s and s^2 (general B)
        term2_a_dataU  = - 2.0 * C1 * (invP * s)
        term3_a_dataU  = + 2.0 * (C2 * (invP * (s * s))) * a_flat
        grad_a_U  = (term1_a_priorU + term2_a_dataU + term3_a_dataU).unsqueeze(2).to(self.dtype)

        # [unchanged] prior term for W still uses ARD precision ρ; data part already fixed above
        grad_w_U = (G + (self.T) * W * self.rho.unsqueeze(1)).to(self.dtype)

        # clip
        if self.algo.grad_clip_norm is not None:
            gw2 = (grad_w_U * grad_w_U).sum(dim=2, keepdim=True)
            ga2 = (grad_a_U * grad_a_U).sum(dim=2, keepdim=True)
            gn = torch.sqrt(gw2 + ga2) + 1e-12
            scale = torch.clamp(self.algo.grad_clip_norm / gn, max=1.0)
            grad_w_U = grad_w_U * scale
            grad_a_U = grad_a_U * scale

        return grad_w_U, grad_a_U, (f_acc.unsqueeze(-1) if return_field else None)

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
                Phi = activation(Z, self.mdl.act)
                fa = torch.bmm(Phi, self.a)
                # [consistent] forward uses same general-B scale
                f[:, start:start+n] = (self.scale_f * fa.squeeze(-1)).to(self.dtype)
        return f.unsqueeze(-1)

    # ---------------- SGLD ----------------

    @torch.no_grad()
    def _sgld_inner(self, X: torch.Tensor, r: torch.Tensor, eta: float, return_field: bool = False):
        """One inner SGLD step. Optionally piggybacks the field on the last step.

        Args:
            X, r: training inputs and residual.
            eta: step size.
            return_field: if True, also compute and return f(X) for current params.
        Returns:
            f_acc: (E,P,1) field if return_field is True, else None.
        """
        gw, ga, f_acc = self._stats_and_grads_stream(X, r, self.W, self.a, return_field=return_field)
        self.W.add_(gw, alpha=-eta)
        self.a.add_(ga, alpha=-eta)
        # noise std = sqrt(2*T*eta)
        nstd = math.sqrt(2.0*self.T*eta)
        self.W.add_(torch.randn_like(self.W, dtype=self.dtype), alpha=nstd)
        self.a.add_(torch.randn_like(self.a, dtype=self.dtype), alpha=nstd)

        if self.algo.kill_nan_particles:
            bad_now = ~torch.isfinite(self.W).all(dim=2) | ~torch.isfinite(self.a).all(dim=2)
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

    # ---------------- Held-out eval ----------------

    @torch.no_grad()
    def _eval_on_X(self, X_eval: torch.Tensor) -> Dict[str, Any]:
        """Evaluate metrics on provided X_eval (shape E,P_eval,d)."""
        d = self.mdl.d
        Eexp, P_eval, _ = X_eval.shape
        M = len(self.sets)
        f2_sum = torch.zeros(Eexp, device=self.device, dtype=self.dtype)
        v_list = torch.zeros(Eexp, M, device=self.device, dtype=self.dtype) if M>0 else None
        G_list = torch.zeros(Eexp, M, M, device=self.device, dtype=self.dtype) if M>0 else None

        scale = self.scale_f
        a = self.a.detach()

        step = max(1, self.algo.batch_eval)
        for start in range(0, P_eval, step):
            n = min(step, P_eval-start)
            Xc = X_eval[:, start:start+n, :].contiguous()
            Z = self._batched_mm_X_Wt(Xc, self.W.contiguous())
            Phi = activation(Z, self.mdl.act)
            f = (scale * torch.bmm(Phi, a).squeeze(-1))
            f2_sum += (f*f).sum(dim=1)

            if M>0:
                for e in range(Eexp):
                    Ccols = [parity_character(Xc[e], S) for S in self.sets]
                    C = torch.stack(Ccols, dim=1)
                    v_list[e] += C.t().matmul(f[e])
                    G_list[e] += C.t().matmul(C)

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
        ones = torch.ones(M, device=self.device, dtype=self.dtype)
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

    # ---------------- Anderson acceleration ----------------

    @torch.no_grad()
    def _anderson_update(self, f_prev: torch.Tensor, f_new: torch.Tensor,
                         aa_hist: List[torch.Tensor], g_hist: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Walker-Ni Anderson (type-I/II hybrid): use differences of residuals to build a tiny LS problem.
        f_prev: current estimate (E,P,1); f_new = G(f_prev).
        aa_hist: list of past residuals r_i = f_i_new - f_i (flattened).
        g_hist: list of past g_i = f_i_new (flattened).
        """
        if not self.algo.use_anderson or self.algo.aa_depth <= 0:
            return f_new, aa_hist, g_hist

        # residual at k
        r_k = (f_new - f_prev).detach().reshape(-1)
        g_k = f_new.detach().reshape(-1)

        # append
        aa_hist.append(r_k)
        g_hist.append(g_k)
        if len(aa_hist) <= 1:
            # not enough history
            if len(aa_hist) > self.algo.aa_depth:
                aa_hist.pop(0); g_hist.pop(0)
            return f_new, aa_hist, g_hist

        m = min(self.algo.aa_depth, len(aa_hist)-1)
        # use last m+1 residuals
        R_cols = []
        G_cols = []
        for i in range(-m-1, -1):
            R_cols.append(aa_hist[i])
            G_cols.append(g_hist[i])
        R = torch.stack(R_cols, dim=1)   # [n, m+1]
        # differences
        dR = R[:, 1:] - R[:, :-1]        # [n, m]
        dG = torch.stack(G_cols, dim=1)[:, 1:] - torch.stack(G_cols, dim=1)[:, :-1]  # [n, m]

        # Solve (dR^T dR + λI) * alpha = dR^T r_k
        # alpha shape [m]
        lam = self.algo.aa_reg
        Gram = dR.T @ dR
        rhs  = dR.T @ r_k
        Gram = Gram + lam * torch.eye(Gram.shape[0], device=self.device, dtype=Gram.dtype)
        try:
            alpha = torch.linalg.solve(Gram, rhs)
        except RuntimeError:
            alpha = torch.zeros(rhs.shape, device=self.device, dtype=rhs.dtype)

        # accelerated update: f_{k+1} = g_k - dG * alpha
        g_next = g_k - (dG @ alpha)
        f_next = g_next.reshape_as(f_new)

        # trim history
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
            "iter": [], "train_mse": [], "train_mse_per_exp": [],
            "half_mse_modes": [], "half_noise": [],
            "half_mse_total_ms": [], "half_mse_empirical": [], "m_S": [],
            "half_mse_modes_per_exp": [], "half_noise_per_exp": [],
            "half_mse_total_ms_per_exp": [], "half_mse_empirical_per_exp": [], "m_S_per_exp": [],
            "rho_min": [], "rho_max": [], "elapsed_s": [],
            "bad_prop": [], "bad_reset": [], "dtype": str(self.dtype),
            "c2_mean": [], "c2_eff_mean": [],
            "fp_residual": [], "fp_residual_per_exp": [],
            "rho_delta_rel": [],
            "eta": [], "K": [],
            "test_mse_small": [],
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

            # -------- independent test-MSE early stop (IMMEDIATE) --------
            test_es_triggered = False
            test_mse_val = None
            if (
                self.algo.early_stop_test_mse_enabled
                and (len(self.sets) > 0)
                and (self._X_eval_small is not None)
                and (self._y_eval_small is not None)
                and (it % max(1, self.algo.test_mse_check_every) == 0)
            ):
                f_eval_small = self._field_from_particles_stream(self._X_eval_small)
                test_mse_val = float(((f_eval_small - self._y_eval_small.to(self.dtype))**2).mean().item())
                if test_mse_val < self.algo.early_stop_test_mse_threshold:
                    self._ensure_full_eval_set()
                    ev = self._eval_on_X(self._X_eval_full)
                    rhomin = float(self.rho.min().item()); rhomax = float(self.rho.max().item())

                    hist["iter"].append(it)
                    hist["train_mse"].append(train_mse_mean)
                    hist["train_mse_per_exp"].append(train_mse_per_e.detach().cpu().tolist())
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
                    hist["bad_prop"].append(0)
                    hist["bad_reset"].append(0)
                    hist["c2_mean"].append(self._last_c2_mean)
                    hist["c2_eff_mean"].append(self._last_c2eff_mean)
                    hist["fp_residual"].append(fp_residual_mean)
                    hist["fp_residual_per_exp"].append(fp_residual_per_e.detach().cpu().tolist())
                    hist["rho_delta_rel"].append(rho_delta_rel)
                    hist["eta"].append(eta); hist["K"].append(K)
                    hist["test_mse_small"].append(test_mse_val)

                    payload = {
                        "summary": {
                            "train_mse_last": hist["train_mse"][-1],
                            "P_eval_full": self.algo.P_eval_full,
                            "E": self.E,
                            "early_stop_reason": f"test_mse<{self.algo.early_stop_test_mse_threshold}"
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
                    print(f"[early-stop:TEST] it={it} small-eval test_mse={test_mse_val:.6f} < {self.algo.early_stop_test_mse_threshold}. Stopping.")
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
                    ev = self._eval_on_X(self._X_eval_full)
                    rhomin = float(self.rho.min().item()); rhomax = float(self.rho.max().item())

                    hist["iter"].append(it)
                    hist["train_mse"].append(train_mse_mean)
                    hist["train_mse_per_exp"].append(train_mse_per_e.detach().cpu().tolist())
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
                    hist["bad_prop"].append(0)
                    hist["bad_reset"].append(0)
                    hist["c2_mean"].append(self._last_c2_mean)
                    hist["c2_eff_mean"].append(self._last_c2eff_mean)
                    hist["fp_residual"].append(fp_residual_mean)
                    hist["fp_residual_per_exp"].append(fp_residual_per_e.detach().cpu().tolist())
                    hist["rho_delta_rel"].append(rho_delta_rel)
                    hist["eta"].append(eta); hist["K"].append(K)
                    if test_mse_val is not None:
                        hist["test_mse_small"].append(test_mse_val)

                    payload = {
                        "summary": {
                            "train_mse_last": hist["train_mse"][-1],
                            "P_eval_full": self.algo.P_eval_full,
                            "E": self.E,
                            "early_stop_reason": "fp/rho/mse composite"
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
                    maybe_save(payload, force=True)
                    print(f"[early-stop] it={it} mean_train_mse={train_mse_mean:.6f} fp_residual={fp_residual_mean:.6e} rho_delta_rel={rho_delta_rel:.3e}.")
                    early_stopped = True

            if early_stopped:
                break

            # Routine logging/eval
            if it % self.algo.log_every == 0:
                ev = self._eval_on_X(self._X_eval_small)   # cheap eval
                rhomin = float(self.rho.min().item()); rhomax = float(self.rho.max().item())

                # also record small-set test MSE if labels available
                if (self._y_eval_small is not None):
                    f_eval_small = self._field_from_particles_stream(self._X_eval_small)
                    test_mse_small = float(((f_eval_small - self._y_eval_small.to(self.dtype))**2).mean().item())
                else:
                    test_mse_small = float('nan')

                hist["iter"].append(it)
                hist["train_mse"].append(train_mse_mean)
                hist["train_mse_per_exp"].append(train_mse_per_e.detach().cpu().tolist())
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
                hist["bad_prop"].append(0)
                hist["bad_reset"].append(0)
                hist["c2_mean"].append(self._last_c2_mean)
                hist["c2_eff_mean"].append(self._last_c2eff_mean)
                hist["fp_residual"].append(fp_residual_mean)
                hist["fp_residual_per_exp"].append(fp_residual_per_e.detach().cpu().tolist())
                hist["rho_delta_rel"].append(rho_delta_rel)
                hist["eta"].append(eta); hist["K"].append(K)
                hist["test_mse_small"].append(test_mse_small)

                payload = {
                    "summary": {
                        "train_mse_last": hist["train_mse"][-1],
                        "P_eval_small": self.algo.P_eval_routine,
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
                self._log_event_count += 1
                maybe_save(payload, force=False)

                print(json.dumps({
                    "iter": it, "mean_train_mse": train_mse_mean,
                    "B": self.mdl.B, "N": self.mdl.N, "gamma": self.mdl.gamma,
                    "kappa": self.kappa, "T": self.T,
                    "rho_min": rhomin, "rho_max": rhomax,
                    "dtype": "float64" if self.algo.use_float64 else "float32",
                    "elapsed_s": round(time.time()-t0,2),
                    "saved": save_path,
                    "c2_mean": self._last_c2_mean,
                    "c2_eff_mean": self._last_c2eff_mean,
                    "fp_residual": fp_residual_mean,
                    "rho_delta_rel": rho_delta_rel,
                    "eta": eta, "K": K,
                    "test_mse_small": test_mse_small,
                }))

        # Final FULL EVAL if not early-stopped
        self._ensure_full_eval_set()
        ev = self._eval_on_X(self._X_eval_full)
        rhomin = float(self.rho.min().item()); rhomax = float(self.rho.max().item())
        hist["iter"].append(it)
        hist["train_mse"].append(float(((y - f_mean)**2).mean().item()))
        hist["train_mse_per_exp"].append(((y - f_mean)**2).mean(dim=(1,2)).detach().cpu().tolist())
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
        hist["bad_prop"].append(0)
        hist["bad_reset"].append(0)
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

    P_train_list =[500,1000,10000,2133 ,10, 100, 750, 3666, 5000, 7500]
    kappa_list   =  [1e-2]  #5e-3,7-5e-3,5e-4,1e-3,1e-2,5e-2
    num_exp = 3
    base_seed = 123456

    use_float64 = False
    mdl = Model(d=d, B=512, N=512, gamma=0.5, sigma_a=1.0, sigma_w=1.0, act="relu")
    algo = Algo(
        outer_steps=7_500_000,
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

        # --- Independent TEST MSE early stop (new) ---
        early_stop_test_mse_enabled=True,
        early_stop_test_mse_threshold=0.01,
        test_mse_check_every=5000,
        test_mse_use_small_eval=True,

        # --- LR decay (power 2) ---
        use_lr_decay=True,
        lr_start=1e-3,     # you can set to 1e-3 to mirror Code 2 exactly
        lr_end=5e-4,       # set to 5e-4 to mirror Code 2 exactly
        lr_decay_iters=2_000_000,  # set to 2_000_000 to mirror Code 2 exactly

        # --- Inner SGLD steps schedule ---
        K0=12,
        Kmin=2,
        K_decay=600_000,

        # --- Anderson acceleration ---
        use_anderson=False,
        aa_depth=3,
        aa_reg=1e-8,
        aa_every=1,

        # --- Eval/log slimming ---
        P_eval_full=100_000,    # full eval only at ES/final
        P_eval_routine=32_768,  # cheap routine eval
        save_every_logs=2,      # write json every 3rd log event
    )
    alpha0 = 8.0
    beta01 = alpha0 / d
    ard = ARD(use_ard=True, alpha0=alpha0, ema=0.5, update_every=1,
              rho_min=0.0, rho_max=1e18, beta0=beta01)

    out_dir = "/home/goring/mean_field_langevin/MCMC_composite/results_d35k4hm_paper_final_conv2_long_fin_ard8.0_fix/1e-2"
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
