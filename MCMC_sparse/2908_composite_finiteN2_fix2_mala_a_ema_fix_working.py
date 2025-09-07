# rs_cavity_explicit_aw_ard.py
# RS self-consistent cavity with explicit (a,w), per-particle MALA/SGLD,
# and Automatic Relevance Determination (ARD) prior on w (diagonal precisions ρ_j).
#
# This version keeps the ORIGINAL MALA you had and adds NaN/Inf robustness only:
# - dtype switch (float32/float64)
# - reject non-finite proposals
# - sanitize states (reinit NaN/Inf particles)
# - safe grad clipping (stateless)
# - optional parameter clamps (abs-value or per-particle L2)
#
# Field map: <f> = (N^{1-γ}/B) Φ(W) a
#
# Cavity energy at fixed residual r = y - <f>:
#   E_b(w,a | r,ρ) =
#       (1/(2σ_a^2)) a^2
#     + (1/2) Σ_j ρ_j w_j^2
#     + (1/(2κ^2 P)) Σ_μ ( r_μ - (a/N^γ) φ(w^T x_μ) )^2
#
# Notes:
# * The global term - (B/2) Σ_j log ρ_j is constant w.r.t. (w,a) when ρ is fixed
#   and cancels from Metropolis ratios; we omit it inside per-particle energies.
# * Conditioned on ρ, particles remain independent; gradients stay factorized.

import os, time, math, json, random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

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
    if S.numel() == 0:  # empty-set parity is constant 1
        return torch.ones(X_pm1.shape[0], device=X_pm1.device, dtype=X_pm1.dtype)
    return X_pm1[:, S].prod(dim=1).to(X_pm1.dtype)

def parse_sets(spec: str) -> List[List[int]]:
    import re
    blocks = re.findall(r"\{([^}]*)\}", spec)
    out = []
    for s in blocks:
        toks = [t.strip() for t in s.split(",") if t.strip()!=""]
        out.append(sorted(map(int, toks)))
    if not out: raise ValueError("bad teacher spec")
    return out

def generate_parity(P: int, d: int, sets: List[torch.Tensor], device, dtype):
    g = torch.Generator(device=device).manual_seed(0)
    X = (torch.randint(0,2,(P,d),generator=g,device=device,dtype=torch.int8).to(dtype) * 2.0 - 1.0)
    Ccols = [parity_character(X, S) for S in sets]
    C = torch.stack(Ccols, dim=1) if Ccols else torch.zeros(P,0,device=device, dtype=dtype)
    y = C.sum(dim=1, keepdim=True)
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

    # safety knobs
    grad_clip_norm: Optional[float] = None  # clip joint (w,a) gradient L2 to this value
    use_float64: bool = False               # do everything in float64 (slower but more stable)
    kill_nan_particles: bool = True         # reinit particles whose state becomes non-finite
    max_abs_w: Optional[float] = None       # clamp |w| per-coordinate if set (e.g., 1e3)
    max_abs_a: Optional[float] = None       # clamp |a| if set
    max_l2_w: Optional[float] = None        # clamp per-particle ||w_b||_2 if set
    max_l2_a: Optional[float] = None        # clamp per-particle |a_b| if set
    nan_reinit_std_scale: float = 1.0       # scale for prior std when reinitializing
    log_bad_counts: bool = True             # log counts of bad/rehab particles

# ----------------------------- core ------------------------------

class RSCavityExplicit:
    def __init__(self, mdl: Model, algo: Algo, kappa: float, device: torch.device,
                 teacher_sets: Optional[List[torch.Tensor]] = None,
                 ard: Optional[ARD] = None):
        self.mdl, self.algo, self.kappa = mdl, algo, float(kappa)
        self.device = device
        self.sets = teacher_sets or []
        self.ard = ard or ARD()
        self.dtype = torch.float64 if algo.use_float64 else torch.float32

        # particles
        self.W = torch.randn(mdl.B, mdl.d, device=device, dtype=self.dtype) * (mdl.sigma_w / math.sqrt(mdl.d))
        self.a = torch.randn(mdl.B, 1, device=device, dtype=self.dtype) * mdl.sigma_a

        # ARD precisions ρ_j (start isotropic to match old prior mass d/σ_w^2)
        rho0 = (mdl.d / (mdl.sigma_w**2))
        self.rho = torch.full((mdl.d,), rho0, device=device, dtype=self.dtype)
        if self.ard.beta0 is None:
            self.beta0 = self.ard.alpha0 / float(rho0)
        else:
            self.beta0 = float(self.ard.beta0)

        # mixed precision toggles (only relevant for float32)
        try:
            torch.set_float32_matmul_precision('high')
            torch.backends.cuda.matmul.allow_tf32 = not algo.use_float64
            torch.backends.cudnn.allow_tf32 = not algo.use_float64
        except Exception:
            pass

    # ---------- helpers: sanitize / reinit / clipping ----------

    @torch.no_grad()
    def _reinit_particles(self, mask: torch.Tensor):
        """Reinitialize selected particles from the prior (same variance)."""
        if not mask.any():
            return 0
        n = int(mask.sum().item())
        sw = self.mdl.sigma_w * self.algo.nan_reinit_std_scale
        sa = self.mdl.sigma_a * self.algo.nan_reinit_std_scale
        self.W[mask] = torch.randn(n, self.mdl.d, device=self.device, dtype=self.dtype) * (sw / math.sqrt(self.mdl.d))
        self.a[mask] = torch.randn(n, 1, device=self.device, dtype=self.dtype) * sa
        return n

    @torch.no_grad()
    def _clamp_params(self):
        """Optional clamps to prevent runaway values."""
        if self.algo.max_abs_w is not None:
            self.W.clamp_(-self.algo.max_abs_w, self.algo.max_abs_w)
        if self.algo.max_abs_a is not None:
            self.a.clamp_(-self.algo.max_abs_a, self.algo.max_abs_a)
        if self.algo.max_l2_w is not None:
            # per-particle L2 clamp for w
            norms = torch.norm(self.W, dim=1, keepdim=True) + 1e-12
            scale = torch.clamp(self.algo.max_l2_w / norms, max=1.0)
            self.W.mul_(scale)
        if self.algo.max_l2_a is not None:
            norms = torch.sqrt((self.a*self.a).sum(dim=1, keepdim=True)) + 1e-12
            scale = torch.clamp(self.algo.max_l2_a / norms, max=1.0)
            self.a.mul_(scale)

    # ---------------- core math ----------------

    @torch.no_grad()
    def _field_from_particles(self, X: torch.Tensor) -> torch.Tensor:
        z = X @ self.W.t().contiguous()                     # (P,B)
        Phi = activation(z, self.mdl.act)                   # (P,B)
        scale = (self.mdl.N ** (1.0 - self.mdl.gamma)) / float(self.mdl.B)
        return (scale * Phi @ self.a).to(self.dtype)        # (P,1)

    def _energy_per_particle(self, Phi: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        P = r.shape[0]
        siga = self.mdl.sigma_a

        prior_w_b = 0.5 * (self.rho.view(1, -1) * (self.W * self.W)).sum(dim=1)
        prior_a_b = 0.5 * (1.0 / (siga**2)) * (self.a[:,0]**2)

        C1 = (Phi.t() @ r).view(-1)
        C2 = (Phi * Phi).sum(dim=0)
        a_flat = self.a[:,0]
        N_gamma = self.mdl.N ** self.mdl.gamma
        data_b = ( - (a_flat / N_gamma) * C1 + 0.5 * (a_flat*a_flat / (N_gamma**2)) * C2 ) / (self.kappa**2 * P)

        return (prior_w_b + prior_a_b + data_b).to(self.dtype)

    @torch.no_grad()
    def _grads_cavity(self, X: torch.Tensor, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        P = X.shape[0]
        z = X @ self.W.t().contiguous()
        Phi = activation(z, self.mdl.act)
        dPhi = act_prime(z, self.mdl.act)

        # a-gradient
        term1 = (1.0 / (self.mdl.sigma_a**2)) * self.a[:,0]
        N_gamma = self.mdl.N ** self.mdl.gamma
        term2 = - (Phi.t() @ r).view(-1) / (self.kappa**2 * P * N_gamma)
        term3 = ((Phi*Phi).sum(dim=0) / (self.kappa**2 * P * (N_gamma**2))) * self.a[:,0]
        grad_a = (term1 + term2 + term3).view(-1,1)

        # w-gradient
        M = (r.view(-1, 1) - (self.a.view(1, -1) / N_gamma) * Phi) * dPhi * (self.a.view(1, -1) / N_gamma)  # (P,B)
        G = -(1.0/(self.kappa**2 * P)) * M.t().matmul(X)                                                   # (B,d)
        grad_w = G + self.W * self.rho.view(1, -1)

        # robust, stateless grad clipping (no autograd needed)
        if self.algo.grad_clip_norm is not None:
            # joint norm across (w,a) per-particle
            gw2 = (grad_w * grad_w).sum(dim=1, keepdim=True)
            ga2 = (grad_a * grad_a).sum(dim=1, keepdim=True)
            gn = torch.sqrt(gw2 + ga2) + 1e-12
            scale = torch.clamp(self.algo.grad_clip_norm / gn, max=1.0)
            grad_w = grad_w * scale
            grad_a = grad_a * scale

        return grad_w.to(self.dtype), grad_a.to(self.dtype)

    @torch.no_grad()
    def _mala_inner(self, X: torch.Tensor, r: torch.Tensor, eta: float) -> float:
        """
        Original MALA step over all particles (independently) at fixed residual r,
        with NaN/Inf-robust safeguards. Returns mean accept rate.
        """
        # grads & current energy
        gw, ga = self._grads_cavity(X, r)

        # random noise
        xi_w = torch.randn_like(self.W, dtype=self.dtype)
        xi_a = torch.randn_like(self.a, dtype=self.dtype)

        # propose
        Wp = (self.W - eta * gw + math.sqrt(2.0*eta) * xi_w)
        ap = (self.a - eta * ga + math.sqrt(2.0*eta) * xi_a)

        # reject non-finite proposals early (keeps state unchanged)
        prop_finite = torch.isfinite(Wp).all(dim=1) & torch.isfinite(ap).all(dim=1)

        # energy at current
        z = X @ self.W.t().contiguous()
        Phi = activation(z, self.mdl.act)
        E_curr = self._energy_per_particle(Phi, r)                          # (B,)

        # evaluate proposal only where finite; elsewhere keep current
        W_saved, a_saved = self.W, self.a
        self.W = torch.where(prop_finite.view(-1,1), Wp, self.W)
        self.a = torch.where(prop_finite.view(-1,1), ap, self.a)

        gw_p, ga_p = self._grads_cavity(X, r)
        zp = X @ self.W.t().contiguous()
        Phip = activation(zp, self.mdl.act)
        E_prop = self._energy_per_particle(Phip, r)

        # restore current state for MH update
        self.W, self.a = W_saved, a_saved

        # log proposal densities (joint (W,a)), ORIGINAL isotropic metric
        mw = self.W - eta * gw; ma = self.a - eta * ga
        mpw = Wp - eta * gw_p; mpa = ap - eta * ga_p
        def sqsum_rows(A): return (A*A).sum(dim=tuple(range(1,A.ndim)))
        log_q_prop_given_curr = - (sqsum_rows(Wp - mw) + sqsum_rows(ap - ma)) / (4.0*eta)
        log_q_curr_given_prop = - (sqsum_rows(self.W - mpw) + sqsum_rows(self.a - mpa)) / (4.0*eta)

        # MH log-acceptance
        log_acc = (-E_prop + E_curr) + (log_q_curr_given_prop - log_q_prop_given_curr)

        # anything non-finite -> reject
        ok = torch.isfinite(log_acc) & prop_finite
        log_acc = torch.where(ok, log_acc, torch.full_like(log_acc, -float("inf")))

        u = torch.rand_like(log_acc)
        accept = (torch.log(u) < log_acc)

        # apply accepts
        n_acc = int(accept.sum().item())
        if n_acc > 0:
            self.W[accept] = Wp[accept]
            self.a[accept] = ap[accept]

        # sanitize states (very rare if guards worked)
        n_bad = 0
        if self.algo.kill_nan_particles:
            bad_now = ~torch.isfinite(self.W).all(dim=1) | ~torch.isfinite(self.a).all(dim=1)
            if bad_now.any():
                n_bad += int(bad_now.sum().item())
                n_bad -= self._reinit_particles(bad_now)  # returns how many reinit (== bad count)

        # optional parameter clamps
        self._clamp_params()

        # optionally log counts (returned to outer loop via attribute)
        self._last_bad = n_bad
        self._last_bad_prop = int((~prop_finite).sum().item())

        return n_acc / float(self.mdl.B)

    @torch.no_grad()
    def _sgld_inner(self, X: torch.Tensor, r: torch.Tensor, eta: float):
        gw, ga = self._grads_cavity(X, r)
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
        ss = 0.5 * (self.W * self.W).sum(dim=0)  # (d,)
        beta_post = torch.tensor(self.beta0, device=self.device, dtype=self.dtype) + ss
        rho_hat = alpha_post / torch.clamp(beta_post, min=torch.tensor(1e-24, dtype=self.dtype, device=self.device))
        rho_hat = torch.clamp(rho_hat, min=self.ard.rho_min, max=self.ard.rho_max)
        self.rho.mul_(1.0 - self.ard.ema).add_(rho_hat, alpha=self.ard.ema)

    @torch.no_grad()
    def _eval_heldout(self, d: int, P_eval: int, chunk: int) -> Dict[str, float]:
        device = self.device
        M = len(self.sets)
        sum_f2 = torch.zeros(1, device=device, dtype=self.dtype)
        sum_Ct_f = torch.zeros(M, device=device, dtype=self.dtype) if M>0 else None
        sum_G = torch.zeros(M,M, device=device, dtype=self.dtype) if M>0 else None

        scale = (self.mdl.N ** (1.0 - self.mdl.gamma)) / float(self.mdl.B)
        a = self.a.detach()

        g = torch.Generator(device=device).manual_seed(1234567)
        for start in range(0, P_eval, chunk):
            n = min(chunk, P_eval-start)
            Xc = (torch.randint(0,2,(n,d),generator=g,device=device,dtype=torch.int8).to(self.dtype) * 2.0 - 1.0)
            z = Xc @ self.W.t().contiguous()
            Phi = activation(z, self.mdl.act)
            f = (scale * Phi @ a)[:,0]
            sum_f2 += (f*f).sum()
            if M>0:
                Ccols = [parity_character(Xc, S) for S in self.sets]
                C = torch.stack(Ccols, dim=1)
                sum_Ct_f += C.t().matmul(f)
                sum_G += C.t().matmul(C)
            del Xc, z, Phi, f
            if M>0: del C

        invP = 1.0/float(P_eval)
        f2_bar = float((sum_f2*invP).item())
        out = dict(half_mse_empirical=0.5*f2_bar, half_mse_total_ms=0.5*f2_bar,
                   half_mse_modes=0.0, half_noise=0.5*f2_bar, m_S=[])
        if M==0: return out

        v = sum_Ct_f * invP
        G = sum_G * invP
        ones = torch.ones(M, device=device, dtype=self.dtype)
        m_S = v
        mTm = float((m_S*m_S).sum().item())
        mTGm = float(m_S.view(1,-1).matmul(G).matmul(m_S.view(-1,1)).item())
        noise = f2_bar - 2.0*mTm + mTGm
        out.update(
            m_S=m_S.detach().cpu().tolist(),
            half_mse_modes=0.5*float(((1.0-m_S)**2).sum().item()),
            half_noise=0.5*float(noise),
            half_mse_total_ms=0.5*float(((1.0-m_S)**2).sum().item()) + 0.5*float(noise),
            half_mse_empirical=0.5*(f2_bar - 2.0*float(ones.dot(v).item())
                                    + float(ones.view(1,-1).matmul(G).matmul(ones.view(-1,1)).item()))
        )
        return out

    @torch.no_grad()
    def run(self, X: torch.Tensor, y: torch.Tensor, out_dir: str, tag: str=""):
        os.makedirs(out_dir, exist_ok=True)
        # cast training data to working dtype
        X = X.to(self.dtype)
        y = y.to(self.dtype)

        P = X.shape[0]
        f_mean = torch.zeros(P,1, device=self.device, dtype=self.dtype)

        hist = {
            "iter": [], "train_mse": [], "accept": [],
            "half_mse_modes": [], "half_noise": [],
            "half_mse_total_ms": [], "half_mse_empirical": [], "m_S": [],
            "rho_min": [], "rho_max": [], "elapsed_s": [],
            "bad_prop": [], "bad_reset": [], "dtype": str(self.dtype)
        }
        t0 = time.time()

        for it in range(1, self.algo.outer_steps+1):
            # fixed residual
            r = (y - f_mean)

            # inner sampler
            acc = 0.0
            bad_prop_sum = 0
            bad_reset_sum = 0
            for _ in range(self.algo.inner_mala_steps):
                if self.algo.use_mala:
                    a_rate = self._mala_inner(X, r, self.algo.step_size)
                    acc += a_rate
                    if self.algo.log_bad_counts:
                        bad_prop_sum += getattr(self, "_last_bad_prop", 0)
                        bad_reset_sum += getattr(self, "_last_bad", 0)
                else:
                    self._sgld_inner(X, r, self.algo.step_size)
            if self.algo.use_mala:
                acc /= max(1, self.algo.inner_mala_steps)

            # ARD update (slow, stable)
            if self.ard.use_ard and (it % self.ard.update_every == 0):
                self._update_rho_ard()

            # field self-consistency
            f_new = self._field_from_particles(X)
            if self.algo.cg_like_update:
                f_mean = (1.0 - self.algo.field_blend) * f_mean + self.algo.field_blend * f_new
            else:
                f_mean = f_new

            # logging
            if it % self.algo.log_every == 0 or it==1 or it==self.algo.outer_steps:
                train_mse = float(((y - f_mean)**2).mean().item())
                ev = self._eval_heldout(self.mdl.d, self.algo.P_eval, self.algo.batch_eval)
                rhomin = float(self.rho.min().item())
                rhomax = float(self.rho.max().item())

                hist["iter"].append(it)
                hist["train_mse"].append(train_mse)
                hist["accept"].append(acc if self.algo.use_mala else 0.0)
                hist["half_mse_modes"].append(ev["half_mse_modes"])
                hist["half_noise"].append(ev["half_noise"])
                hist["half_mse_total_ms"].append(ev["half_mse_total_ms"])
                hist["half_mse_empirical"].append(ev["half_mse_empirical"])
                hist["m_S"].append(ev["m_S"])
                hist["rho_min"].append(rhomin); hist["rho_max"].append(rhomax)
                hist["elapsed_s"].append(round(time.time()-t0,2))
                hist["bad_prop"].append(bad_prop_sum)
                hist["bad_reset"].append(bad_reset_sum)

                print(json.dumps({
                    "iter": it, "train_mse": train_mse, "accept": acc if self.algo.use_mala else 0.0,
                    "half_mse_modes": ev["half_mse_modes"], "half_noise": ev["half_noise"],
                    "half_mse_total_ms": ev["half_mse_total_ms"], "half_mse_empirical": ev["half_mse_empirical"],
                    "m_S": ev["m_S"], "B": self.mdl.B, "N": self.mdl.N, "gamma": self.mdl.gamma,
                    "kappa": self.kappa, "rho_min": rhomin, "rho_max": rhomax,
                    "bad_prop": bad_prop_sum, "bad_reset": bad_reset_sum,
                    "dtype": "float64" if self.algo.use_float64 else "float32",
                    "elapsed_s": round(time.time()-t0,2)
                }))

        # save
        tag = tag or time.strftime("%Y%m%d_%H%M%S")
        out = {"summary": {
                    "train_mse_last": hist["train_mse"][-1],
                    "accept_last": hist["accept"][-1] if hist["accept"] else None,
                    "P_eval": self.algo.P_eval
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
               }}
        path = os.path.join(out_dir, f"rs_cavity_explicit_aw_ard_{tag}_Ptr{P}_Peval{self.algo.P_eval}_kap{self.kappa:.3e}_N{self.mdl.N}_B{self.mdl.B}_g{self.mdl.gamma}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {path}")
        return out

# ----------------------------- main ------------------------------

if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # teacher parity {0,1,2,3} on d=35
    d = 35
    spec = "{0,1,2,3}"
    sets = [torch.tensor(s, device=device, dtype=torch.long) for s in parse_sets(spec)]

    # data
    P_train = 1000
    # choose dtype once
    use_float64 = False  # set True if you want double precision
    dtype = torch.float64 if use_float64 else torch.float32
    X, y = generate_parity(P_train, d, sets, device, dtype)

    # hyperparams (H100 friendly-ish defaults)
    mdl = Model(d=d, B=1024, N=512, gamma=0.5, sigma_a=1.0, sigma_w=1.0, act="relu")
    algo = Algo(
        outer_steps=50_000, inner_mala_steps=100, step_size=5e-7, use_mala=True,
        log_every=10, cg_like_update=False, field_blend=0.8,
        P_eval=50_000, batch_eval=50_000,

        # ---- NaN/Inf robustness switches ----
        use_float64=use_float64,      # flip to True for double precision
        grad_clip_norm=None,          # stateless per-particle L2 clip on (w,a) grads
        kill_nan_particles=True,      # reinit any NaN/Inf particle from prior
        max_abs_w=None,               # e.g. 1e3 to hard-clamp |w|
        max_abs_a=None,               # e.g. 1e3 to hard-clamp |a|
        max_l2_w=None,                # e.g. 1e3 to clamp per-particle ||w_b||
        max_l2_a=None,                # e.g. 1e3 to clamp per-particle |a_b|
        nan_reinit_std_scale=1.0,
        log_bad_counts=True,
    )
    kappa = 7.5e-3

    # ARD config (weak prior; EMA for stability)
    ard = ARD(use_ard=True, alpha0=1e-2, ema=0.25, update_every=1,
              rho_min=0.0, rho_max=1e18, beta0=None)

    out_dir = "/home/goring/mean_field_langevin/MCMC_composite/resutls_mf1/test"
    solver = RSCavityExplicit(mdl, algo, kappa, device, teacher_sets=sets, ard=ard)
    solver.run(X, y, out_dir, tag=f"P{P_train}_kap{kappa:.3e}")
