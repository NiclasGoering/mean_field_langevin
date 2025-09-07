# rs_cavity_explicit_aw.py
# RS self-consistent cavity with explicit (a,w) and per-particle MALA.
# Field update: <f> = (N^{1-γ}/B) Φ a  (Monte Carlo estimate of N^{1-γ} E[a φ])
#
# Single-neuron cavity energy at fixed residual r = y - <f>:
#   Baseline (separate L2 priors):
#     E_b(w,a) = d/(2σ_w^2)||w||^2 + (1/(2σ_a^2)) a^2 + (1/(2κ^2 P)) Σμ (rμ - (a/N^γ) φ(w^T xμ))^2
#
# === Why A (path-norm) is the right lens for full-batch GD ===
# 1) Scale symmetry is real and important.
#    For 1-homogeneous φ (ReLU): φ(c·w^T x) = c·φ(w^T x). A unit’s contribution a·φ(w^T x)
#    is invariant under the rescaling w ← t w, a ← a/t. The network has rays of equivalent
#    parameters (“degeneracy”); gradient flow can move along them without changing the function.
#
# 2) Separate L2 priors break that symmetry.
#    With d/(2σ_w^2)||w||^2 + (1/(2σ_a^2)) a^2, two parameterizations of the SAME function
#    (related by t) get different penalties. The model — not the function — breaks the symmetry.
#
# 3) Eliminating the scale recovers a scale-invariant penalty: the path-norm.
#    Minimizing the quadratic penalty over t>0 while keeping the functional contribution fixed yields
#        min_t { (d/(2σ_w^2)) t^2 ||ŵ||^2 + (1/(2σ_a^2)) a^2 / t^2 }  =  λ · |a| · ||ŵ||
#    with λ = sqrt(d / (σ_w^2 σ_a^2)). If ||ŵ||≈1 (or projecting w to the unit sphere),
#    this becomes λ·|a| — an ℓ1-type (variation / path-norm) penalty whose extreme points are sparse.
#    In the mean-field limit, minimizing loss + path-norm over measures picks atomic solutions:
#    a handful of “winner” features carry mass (±O(1) a’s), the rest stay near zero.
#
# This file implements the path-norm explicitly as  E_path = λ Σ_b |a_b| ||w_b||.
# We use smooth |·| and ||·|| with a tiny ε for stable gradients.
#
# H100-friendly: bf16 autocast, TF32 matmuls, big vectorized matmuls, streaming eval.

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
    if kind == "relu": return (z > 0).to(torch.float32)
    if kind == "tanh": return 1.0 - torch.tanh(z) ** 2
    raise ValueError(f"Unknown activation: {kind}")

def parity_character(X_pm1: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    if S.numel() == 0:  # empty-set parity is constant 1
        return torch.ones(X_pm1.shape[0], device=X_pm1.device, dtype=torch.float32)
    return X_pm1[:, S].prod(dim=1).to(torch.float32)

def parse_sets(spec: str) -> List[List[int]]:
    import re
    blocks = re.findall(r"\{([^}]*)\}", spec)
    out = []
    for s in blocks:
        toks = [t.strip() for t in s.split(",") if t.strip()!=""]
        out.append(sorted(map(int, toks)))
    if not out: raise ValueError("bad teacher spec")
    return out

def generate_parity(P: int, d: int, sets: List[torch.Tensor], device):
    g = torch.Generator(device=device).manual_seed(0)
    X = (torch.randint(0,2,(P,d),generator=g,device=device,dtype=torch.int8).float()*2.0-1.0)
    Ccols = [parity_character(X, S) for S in sets]
    C = torch.stack(Ccols, dim=1) if Ccols else torch.zeros(P,0,device=device)
    y = C.sum(dim=1, keepdim=True)
    return X, y

# ----------------------------- config -----------------------------

@dataclass
class Model:
    d: int = 35
    B: int = 16384         # number of particles
    N: int = 512           # network N in f = N^{-γ} Σ a φ
    gamma: float = 0.5
    sigma_a: float = 1.0
    sigma_w: float = 1.0
    act: str = "relu"

    # ---- Path-norm controls (A) ----
    use_path_norm: bool = True                 # if True, use λ Σ |a| ||w|| instead of separate L2 priors
    path_eps: float = 1e-12                    # smoothing ε for |a| and ||w||
    path_lambda_scale: float = 1.0             # overall multiplier on λ = sqrt(d/(σ_w^2 σ_a^2))
    keep_l2_w: float = 0.0                     # add small extra (keep_l2_w)*d/(2σ_w^2)||w||^2
    keep_l2_a: float = 0.0                     # add small extra (keep_l2_a)/(2σ_a^2)a^2
    project_w_unit_ball: bool = False          # optional: project w to unit ball after steps (OFF for MALA)

@dataclass
class Algo:
    outer_steps: int = 2000              # RS fixed-point iterations
    inner_mala_steps: int = 1            # MALA steps per outer iter
    step_size: float = 1e-5              # MALA/SGLD step size
    use_mala: bool = True                # False -> SGLD (no accept/reject)
    log_every: int = 10
    cg_like_update: bool = False         # optional: blend field with old (<f> ← (1-α) old + α new)
    field_blend: float = 1.0             # 1.0 -> full overwrite, <1.0 -> underrelax
    batch_eval: int = 262_144            # eval chunk
    P_eval: int = 50_000

# ----------------------------- core -------------------------------

class RSCavityExplicit:
    def __init__(self, mdl: Model, algo: Algo, kappa: float, device: torch.device,
                 teacher_sets: Optional[List[torch.Tensor]] = None):
        self.mdl, self.algo, self.kappa = mdl, algo, float(kappa)
        self.device = device
        self.sets = teacher_sets or []

        # params (particles)
        self.W = torch.randn(mdl.B, mdl.d, device=device) * (mdl.sigma_w / math.sqrt(mdl.d))
        self.a = torch.randn(mdl.B, 1, device=device) * mdl.sigma_a

        # path-norm λ
        self.lam_path = self.mdl.path_lambda_scale * math.sqrt(
            self.mdl.d / (self.mdl.sigma_w**2 * self.mdl.sigma_a**2)
        )

        # mixed precision + TF32
        try:
            torch.set_float32_matmul_precision('high')
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    @torch.no_grad()
    def _field_from_particles(self, X: torch.Tensor) -> torch.Tensor:
        """<f> = (N^{1-γ}/B) Φ(W) a on the training points."""
        z = X @ self.W.t().contiguous()                     # (P,B)
        Phi = activation(z, self.mdl.act).to(torch.float32) # (P,B)
        scale = (self.mdl.N ** (1.0 - self.mdl.gamma)) / float(self.mdl.B)
        return (scale * Phi @ self.a).to(torch.float32)     # (P,1)

    def _energy_per_particle(self, Phi: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        E_b for all b (vector length B), with Φ given for current W and r = y-<f>.
        """
        P = r.shape[0]
        d, sigw, siga = self.mdl.d, self.mdl.sigma_w, self.mdl.sigma_a

        # ---- Prior / regularizer ----
        if self.mdl.use_path_norm:
            eps = self.mdl.path_eps
            w_norm = torch.sqrt((self.W * self.W).sum(dim=1) + eps)     # (B,)
            abs_a = torch.sqrt(self.a[:,0] * self.a[:,0] + eps)         # (B,)
            prior_b = self.lam_path * (abs_a * w_norm)                   # (B,)

            # Optional tiny extra L2 for damping (both are scaled by keep_l2_* in [0,1])
            if self.mdl.keep_l2_w > 0.0:
                prior_b = prior_b + 0.5 * self.mdl.keep_l2_w * (d/(sigw**2)) * (self.W * self.W).sum(dim=1)
            if self.mdl.keep_l2_a > 0.0:
                prior_b = prior_b + 0.5 * self.mdl.keep_l2_a * (1.0/(siga**2)) * (self.a[:,0]**2)
        else:
            # original separate L2 priors
            prior_b = 0.5 * (d/(sigw**2)) * (self.W * self.W).sum(dim=1) \
                    + 0.5 * (1.0/(siga**2)) * (self.a[:,0]**2)          # (B,)

        # ---- Data term (separable expansion) ----
        # data per b: ( - (a_b/N^γ) <φ_b, r> + 0.5 (a_b^2/N^{2γ}) <φ_b^2> ) / (κ^2 P)
        C1 = (Phi.t() @ r).view(-1)                     # (B,)
        C2 = (Phi * Phi).mean(dim=0) * P                # (B,)
        a_flat = self.a[:,0]
        N_gamma = self.mdl.N ** self.mdl.gamma
        data_b = ( - (a_flat / N_gamma) * C1 + 0.5 * (a_flat*a_flat / (N_gamma**2)) * C2 ) / (self.kappa**2 * P)

        return prior_b + data_b

    @torch.no_grad()
    def _grads_cavity(self, X: torch.Tensor, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized grads of E_b at fixed residual r.
        ∇_a E = prior + data terms
        ∇_w E_b = prior + data terms
        """
        P = X.shape[0]
        z = X @ self.W.t().contiguous()                # (P,B)
        Phi = activation(z, self.mdl.act).to(torch.float32)
        dPhi = act_prime(z, self.mdl.act).to(torch.float32)

        N_gamma = self.mdl.N ** self.mdl.gamma

        # ---- a-gradient: data terms ----
        term2 = - (Phi.t() @ r).view(-1) / (self.kappa**2 * P * N_gamma)
        term3 = ((Phi*Phi).sum(dim=0) / (self.kappa**2 * P * (N_gamma**2))) * self.a[:,0]

        # ---- a-gradient: prior / regularizer ----
        if self.mdl.use_path_norm:
            eps = self.mdl.path_eps
            w_norm = torch.sqrt((self.W * self.W).sum(dim=1) + eps)     # (B,)
            abs_a = torch.sqrt(self.a[:,0] * self.a[:,0] + eps)         # (B,)
            # d/da [λ |a| ||w||]  with smooth |a|≈sqrt(a^2+ε):
            term1 = self.lam_path * (self.a[:,0] / abs_a) * w_norm      # (B,)
            if self.mdl.keep_l2_a > 0.0:
                term1 = term1 + self.mdl.keep_l2_a * (1.0/(self.mdl.sigma_a**2)) * self.a[:,0]
        else:
            term1 = (1.0 / (self.mdl.sigma_a**2)) * self.a[:,0]

        grad_a = (term1 + term2 + term3).view(-1,1)                      # (B,1)

        # ---- w-gradient: data terms ----
        # M_{μb} = (rμ - (a_b/N^γ) φ_{μb}) * φ'(z_{μb}) * (a_b/N^γ)
        M = (r.view(-1, 1) - (self.a.view(1, -1) / N_gamma) * Phi) * dPhi * (self.a.view(1, -1) / N_gamma)
        G = -(1.0/(self.kappa**2 * P)) * M.t().matmul(X)                 # (B,d)

        # ---- w-gradient: prior / regularizer ----
        if self.mdl.use_path_norm:
            eps = self.mdl.path_eps
            w_norm = torch.sqrt((self.W * self.W).sum(dim=1) + eps)     # (B,)
            abs_a = torch.sqrt(self.a[:,0] * self.a[:,0] + eps)         # (B,)
            # d/dw [λ |a| ||w||] ≈ λ |a| * w / ||w||
            gw_prior = self.lam_path * (abs_a / w_norm).view(-1,1) * self.W
            if self.mdl.keep_l2_w > 0.0:
                gw_prior = gw_prior + self.mdl.keep_l2_w * (self.mdl.d/(self.mdl.sigma_w**2)) * self.W
        else:
            gw_prior = (self.mdl.d/(self.mdl.sigma_w**2)) * self.W

        grad_w = G + gw_prior

        return grad_w.to(torch.float32), grad_a.to(torch.float32)

    @torch.no_grad()
    def _maybe_project_w(self):
        if self.mdl.project_w_unit_ball:
            n = self.W.norm(dim=1, keepdim=True).clamp_min(1.0)
            self.W /= n

    @torch.no_grad()
    def _mala_inner(self, X: torch.Tensor, r: torch.Tensor, eta: float) -> float:
        """
        One MALA step over all particles (independently) at fixed residual r.
        Returns mean accept rate.
        NOTE: If project_w_unit_ball=True, projection after accept will change the
        stationary law (breaks detailed balance). Keep it False for MALA.
        """
        # current grads and energies
        gw, ga = self._grads_cavity(X, r)
        z = X @ self.W.t().contiguous()
        Phi = activation(z, self.mdl.act).to(torch.float32)
        E_curr = self._energy_per_particle(Phi, r)                          # (B,)

        # propose
        xi_w = torch.randn_like(self.W); xi_a = torch.randn_like(self.a)
        Wp = (self.W - eta * gw + math.sqrt(2.0*eta) * xi_w).requires_grad_(False)
        ap = (self.a - eta * ga + math.sqrt(2.0*eta) * xi_a).requires_grad_(False)

        # grads and energies at proposal
        W_saved, a_saved = self.W, self.a
        self.W, self.a = Wp, ap
        gw_p, ga_p = self._grads_cavity(X, r)
        zp = X @ self.W.t().contiguous()
        Phip = activation(zp, self.mdl.act).to(torch.float32)
        E_prop = self._energy_per_particle(Phip, r)
        self.W, self.a = W_saved, a_saved

        # log proposal densities per particle for joint (W,a)
        mw = self.W - eta * gw; ma = self.a - eta * ga
        mpw = Wp - eta * gw_p; mpa = ap - eta * ga_p
        def sqsum_rows(A): return (A*A).sum(dim=tuple(range(1,A.ndim)))
        log_q_prop_given_curr = - (sqsum_rows(Wp - mw) + sqsum_rows(ap - ma)) / (4.0*eta)   # (B,)
        log_q_curr_given_prop = - (sqsum_rows(self.W - mpw) + sqsum_rows(self.a - mpa)) / (4.0*eta)

        log_acc = (-E_prop + E_curr) + (log_q_curr_given_prop - log_q_prop_given_curr)      # (B,)
        u = torch.rand_like(log_acc)
        accept = (torch.log(u) < log_acc)

        # apply accepts per particle
        n_acc = int(accept.sum().item())
        self.W[accept] = Wp[accept]
        self.a[accept] = ap[accept]

        # (optional) projection: keep OFF for MALA
        self._maybe_project_w()

        return n_acc / float(self.mdl.B)

    @torch.no_grad()
    def _sgld_inner(self, X: torch.Tensor, r: torch.Tensor, eta: float):
        gw, ga = self._grads_cavity(X, r)
        self.W.add_(gw, alpha=-eta)
        self.a.add_(ga, alpha=-eta)
        self.W.add_(torch.randn_like(self.W), alpha=math.sqrt(2.0*eta))
        self.a.add_(torch.randn_like(self.a), alpha=math.sqrt(2.0*eta))
        self._maybe_project_w()

    @torch.no_grad()
    def _eval_heldout(self, d: int, P_eval: int, chunk: int) -> Dict[str, float]:
        """Compute m_S, noise, and half-MSE on a large held-out stream."""
        device = self.device
        M = len(self.sets)
        sum_f2 = torch.zeros(1, device=device, dtype=torch.float32)
        sum_Ct_f = torch.zeros(M, device=device, dtype=torch.float32) if M>0 else None
        sum_G = torch.zeros(M,M, device=device, dtype=torch.float32) if M>0 else None

        scale = (self.mdl.N ** (1.0 - self.mdl.gamma)) / float(self.mdl.B)
        a = self.a.detach()

        g = torch.Generator(device=device).manual_seed(1234567)
        for start in range(0, P_eval, chunk):
            n = min(chunk, P_eval-start)
            Xc = (torch.randint(0,2,(n,d),generator=g,device=device,dtype=torch.int8).float()*2.0-1.0)
            z = Xc @ self.W.t().contiguous()
            Phi = activation(z, self.mdl.act).to(torch.float32)
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
        ones = torch.ones(M, device=device, dtype=torch.float32)
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
        P = X.shape[0]
        # init field
        f_mean = torch.zeros(P,1, device=self.device, dtype=torch.float32)

        hist = {
            "iter": [], "train_mse": [], "accept": [],
            "half_mse_modes": [], "half_noise": [],
            "half_mse_total_ms": [], "half_mse_empirical": [], "m_S": [],
            "elapsed_s": []
        }
        t0 = time.time()

        for it in range(1, self.algo.outer_steps+1):
            # fixed residual for cavity step
            r = (y - f_mean).to(torch.float32)

            # inner sampler at fixed r
            acc = 0.0
            for _ in range(self.algo.inner_mala_steps):
                if self.algo.use_mala:
                    acc += self._mala_inner(X, r, self.algo.step_size)
                else:
                    self._sgld_inner(X, r, self.algo.step_size)
            acc /= max(1, self.algo.inner_mala_steps)

            # update field from particles (self-consistency)
            f_new = self._field_from_particles(X)
            if self.algo.cg_like_update:
                f_mean = (1.0 - self.algo.field_blend) * f_mean + self.algo.field_blend * f_new
            else:
                f_mean = f_new

            # logging
            if it % self.algo.log_every == 0 or it==1 or it==self.algo.outer_steps:
                train_mse = float(((y - f_mean)**2).mean().item())
                ev = self._eval_heldout(self.mdl.d, self.algo.P_eval, self.algo.batch_eval)
                hist["iter"].append(it)
                hist["train_mse"].append(train_mse)
                hist["accept"].append(acc)
                hist["half_mse_modes"].append(ev["half_mse_modes"])
                hist["half_noise"].append(ev["half_noise"])
                hist["half_mse_total_ms"].append(ev["half_mse_total_ms"])
                hist["half_mse_empirical"].append(ev["half_mse_empirical"])
                hist["m_S"].append(ev["m_S"])
                hist["elapsed_s"].append(round(time.time()-t0,2))
                print(json.dumps({
                    "iter": it, "train_mse": train_mse, "accept": acc,
                    "half_mse_modes": ev["half_mse_modes"], "half_noise": ev["half_noise"],
                    "half_mse_total_ms": ev["half_mse_total_ms"], "half_mse_empirical": ev["half_mse_empirical"],
                    "m_S": ev["m_S"], "B": self.mdl.B, "N": self.mdl.N, "gamma": self.mdl.gamma,
                    "kappa": self.kappa, "use_path_norm": self.mdl.use_path_norm,
                    "lambda_path": self.lam_path, "elapsed_s": round(time.time()-t0,2)
                }))

        # save
        tag = tag or time.strftime("%Y%m%d_%H%M%S")
        out = {"summary": {
                    "train_mse_last": hist["train_mse"][-1],
                    "accept_last": hist["accept"][-1],
                    "P_eval": self.algo.P_eval
                },
               "traj": hist,
               "config": {"model": self.mdl.__dict__, "algo": self.algo.__dict__, "kappa": self.kappa}}
        path = os.path.join(out_dir, f"rs_cavity_explicit_aw_{tag}_Ptr{P}_Peval{self.algo.P_eval}_kap{self.kappa:.3e}_N{self.mdl.N}_B{self.mdl.B}_g{self.mdl.gamma}.json")
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
    X, y = generate_parity(P_train, d, sets, device)

    # hyperparams (H100 friendly)
    mdl = Model(
        d=d, B=2048*2, N=512, gamma=0.5, sigma_a=1.0, sigma_w=1.0, act="relu",
        # ---- Path-norm defaults ----
        use_path_norm=True,              # turn off to revert to separate L2 priors
        path_eps=1e-12,
        path_lambda_scale=0.8,           # try (0.5 .. 2.0) to tune sparsity strength
        keep_l2_w=0.0,                   # small (e.g. 0.05) if you want extra damping
        keep_l2_a=0.0,
        project_w_unit_ball=False        # keep False for MALA
    )
    algo = Algo(
        outer_steps=50000, inner_mala_steps=400, step_size=1e-6, use_mala=True,
        log_every=4, cg_like_update=True, field_blend=0.1,
        P_eval=50_000, batch_eval=50_000
    )
    kappa = 7.5e-3

    out_dir = "/home/goring/mean_field_langevin/MCMC_composite/resutls_mf1/test"
    solver = RSCavityExplicit(mdl, algo, kappa, device, teacher_sets=sets)
    solver.run(X, y, out_dir, tag=f"P{P_train}_kap{kappa:.3e}")
