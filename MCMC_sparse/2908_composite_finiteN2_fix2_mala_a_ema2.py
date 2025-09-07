# rs_cavity_explicit_aw_ard2.py
# RS self-consistent cavity with explicit (a,w), per-particle MALA/SGLD,
# and two-parameter ARD prior on w:
# a single precision ρ_sig for indices in S (signal coords), and ρ_noi for the rest.
#
# Field map: <f> = (N^{1-γ}/B) Φ(W) a
#
# Cavity energy at fixed residual r = y - <f>:
#   E_b(w,a | r, ρ_sig, ρ_noi) =
#       (1/(2σ_a^2)) a^2
#     + (1/2) Σ_j ρ_j w_j^2   ,  where ρ_j = ρ_sig if j∈S else ρ_noi
#     + (1/(2κ^2 P)) Σ_μ ( r_μ - (a/N^γ) φ(w^T x_μ) )^2
#
# The global - (B/2)[ |S| log ρ_sig + (d-|S|) log ρ_noi ] is constant in (w,a) for fixed ρ
# and dropped inside per-particle energies (not needed for proposals).

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
    if S.numel() == 0:
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

def union_mask(d: int, sets: List[torch.Tensor], device) -> torch.Tensor:
    """Binary mask (d,) with 1 on union of teacher sets."""
    m = torch.zeros(d, device=device, dtype=torch.float32)
    if len(sets) == 0:
        return m
    idx = torch.unique(torch.cat([s.view(-1) for s in sets]))  # unique sorted
    m[idx] = 1.0
    return m

# ----------------------------- config -----------------------------

@dataclass
class Model:
    d: int = 35
    B: int = 16384
    N: int = 512
    gamma: float = 0.5
    sigma_a: float = 1.0
    sigma_w: float = 1.0
    act: str = "relu"

@dataclass
class ARD2:
    use_ard: bool = True
    alpha0: float = 1e-2        # weak Gamma prior shape
    ema: float = 0.25           # EMA for ρ updates
    update_every: int = 1
    rho_min: float = 1e-6
    rho_max: float = 1e6
    beta0: Optional[float] = None  # if None: set so E[ρ]=d/σ_w^2 initially

@dataclass
class Algo:
    outer_steps: int = 2000
    inner_mala_steps: int = 1
    step_size: float = 1e-5
    use_mala: bool = True
    log_every: int = 10
    cg_like_update: bool = False
    field_blend: float = 1.0
    batch_eval: int = 262_144
    P_eval: int = 50_000
    grad_clip_norm: Optional[float] = None  # manual clipping of raw grads

# ----------------------------- core ------------------------------

class RSCavityExplicit:
    def __init__(self, mdl: Model, algo: Algo, kappa: float, device: torch.device,
                 teacher_sets: Optional[List[torch.Tensor]] = None,
                 ard2: Optional[ARD2] = None):
        self.mdl, self.algo, self.kappa = mdl, algo, float(kappa)
        self.device = device
        self.sets = teacher_sets or []
        self.ard2 = ard2 or ARD2()

        # particles
        self.W = torch.randn(mdl.B, mdl.d, device=device) * (mdl.sigma_w / math.sqrt(mdl.d))
        self.a = torch.randn(mdl.B, 1, device=device) * mdl.sigma_a

        # two-group mask
        self.mask_sig = union_mask(mdl.d, self.sets, device)  # (d,) in {0,1}
        self.mask_noi = 1.0 - self.mask_sig
        self.n_sig = int(self.mask_sig.sum().item())
        self.n_noi = mdl.d - self.n_sig

        # initial precisions (both start isotropic)
        rho0 = (mdl.d / (mdl.sigma_w**2))
        self.rho_sig = torch.tensor(float(rho0), device=device, dtype=torch.float32)
        self.rho_noi = torch.tensor(float(rho0), device=device, dtype=torch.float32)

        # choose beta0 so E[ρ]=rho0
        if self.ard2.beta0 is None:
            self.beta0 = self.ard2.alpha0 / float(rho0)
        else:
            self.beta0 = float(self.ard2.beta0)

        # mixed precision + TF32
        try:
            torch.set_float32_matmul_precision('high')
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    @torch.no_grad()
    def _rho_vec(self) -> torch.Tensor:
        """Vector of ρ_j assembled from group scalars."""
        return (self.rho_sig - self.rho_noi) * self.mask_sig + self.rho_noi

    @torch.no_grad()
    def _field_from_particles(self, X: torch.Tensor) -> torch.Tensor:
        """<f> = (N^{1-γ}/B) Φ(W) a on the training points."""
        z = X @ self.W.t().contiguous()                     # (P,B)
        Phi = activation(z, self.mdl.act).to(torch.float32) # (P,B)
        scale = (self.mdl.N ** (1.0 - self.mdl.gamma)) / float(self.mdl.B)
        return (scale * Phi @ self.a).to(torch.float32)     # (P,1)

    def _energy_per_particle(self, Phi: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        E_b for all b (length B), with Φ given for current W and r = y-<f>.
        Prior term uses two-ρ ARD: 0.5 * Σ_j ρ_j w_{b,j}^2.
        """
        P = r.shape[0]
        siga = self.mdl.sigma_a

        rho_vec = self._rho_vec().view(1, -1)                      # (1,d)
        prior_w_b = 0.5 * (rho_vec * (self.W * self.W)).sum(dim=1) # (B,)
        prior_a_b = 0.5 * (1.0 / (siga**2)) * (self.a[:,0]**2)     # (B,)

        # separable data term with N^γ scaling
        C1 = (Phi.t() @ r).view(-1)                    # (B,)
        C2 = (Phi * Phi).sum(dim=0)                    # (B,)
        a_flat = self.a[:,0]
        N_gamma = self.mdl.N ** self.mdl.gamma
        data_b = ( - (a_flat / N_gamma) * C1 + 0.5 * (a_flat*a_flat / (N_gamma**2)) * C2 ) / (self.kappa**2 * P)

        return prior_w_b + prior_a_b + data_b

    @torch.no_grad()
    def _manual_grad_clip(self, gw: torch.Tensor, ga: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Clip raw gradients by global norm (no autograd .grad needed)."""
        if self.algo.grad_clip_norm is None:
            return gw, ga
        maxn = float(self.algo.grad_clip_norm)
        gn = torch.sqrt((gw*gw).sum() + (ga*ga).sum())
        if torch.isfinite(gn) and gn.item() > maxn:
            scale = maxn / (gn + 1e-12)
            gw = gw * scale
            ga = ga * scale
        return gw, ga

    @torch.no_grad()
    def _grads_cavity(self, X: torch.Tensor, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized grads of E_b at fixed residual r under two-ρ ARD prior.
        ∇_a E = (1/σ_a^2) a  - (1/(κ^2 P N^γ)) Φ^T r  + (1/(κ^2 P N^{2γ})) diag(Φ^T Φ) ⊙ a
        ∇_w E_b = (ρ_vec ⊙ w_b)  - (1/(κ^2 P)) Σ_μ (r_μ - (a_b/N^γ) φ_{μb}) (a_b/N^γ) φ'(z_{μb}) x_μ
        """
        P = X.shape[0]
        z = X @ self.W.t().contiguous()                # (P,B)
        Phi = activation(z, self.mdl.act).to(torch.float32)
        dPhi = act_prime(z, self.mdl.act).to(torch.float32)

        # a-gradient
        term1 = (1.0 / (self.mdl.sigma_a**2)) * self.a[:,0]
        N_gamma = self.mdl.N ** self.mdl.gamma
        term2 = - (Phi.t() @ r).view(-1) / (self.kappa**2 * P * N_gamma)
        term3 = ((Phi*Phi).sum(dim=0) / (self.kappa**2 * P * (N_gamma**2))) * self.a[:,0]
        grad_a = (term1 + term2 + term3).view(-1,1)                                                  # (B,1)

        # w-gradient: data part + prior part
        M = (r.view(-1, 1) - (self.a.view(1, -1) / N_gamma) * Phi) * dPhi * (self.a.view(1, -1) / N_gamma)  # (P,B)
        G = -(1.0/(self.kappa**2 * P)) * M.t().matmul(X)                                                   # (B,d)
        rho_vec = self._rho_vec().view(1, -1)
        grad_w = G + self.W * rho_vec

        # optional raw-grad clipping
        grad_w, grad_a = self._manual_grad_clip(grad_w, grad_a)
        return grad_w.to(torch.float32), grad_a.to(torch.float32)

    @torch.no_grad()
    def _mala_inner(self, X: torch.Tensor, r: torch.Tensor, eta: float) -> float:
        gw, ga = self._grads_cavity(X, r)
        z = X @ self.W.t().contiguous()
        Phi = activation(z, self.mdl.act).to(torch.float32)
        E_curr = self._energy_per_particle(Phi, r)                          # (B,)

        xi_w = torch.randn_like(self.W); xi_a = torch.randn_like(self.a)
        Wp = (self.W - eta * gw + math.sqrt(2.0*eta) * xi_w).requires_grad_(False)
        ap = (self.a - eta * ga + math.sqrt(2.0*eta) * xi_a).requires_grad_(False)

        W_saved, a_saved = self.W, self.a
        self.W, self.a = Wp, ap
        gw_p, ga_p = self._grads_cavity(X, r)
        zp = X @ self.W.t().contiguous()
        Phip = activation(zp, self.mdl.act).to(torch.float32)
        E_prop = self._energy_per_particle(Phip, r)
        self.W, self.a = W_saved, a_saved

        mw = self.W - eta * gw; ma = self.a - eta * ga
        mpw = Wp - eta * gw_p; mpa = ap - eta * ga_p
        def sqsum_rows(A): return (A*A).sum(dim=tuple(range(1,A.ndim)))
        log_q_prop_given_curr = - (sqsum_rows(Wp - mw) + sqsum_rows(ap - ma)) / (4.0*eta)
        log_q_curr_given_prop = - (sqsum_rows(self.W - mpw) + sqsum_rows(self.a - mpa)) / (4.0*eta)

        log_acc = (-E_prop + E_curr) + (log_q_curr_given_prop - log_q_prop_given_curr)
        u = torch.rand_like(log_acc)
        accept = (torch.log(u) < log_acc)

        n_acc = int(accept.sum().item())
        self.W[accept] = Wp[accept]
        self.a[accept] = ap[accept]
        return n_acc / float(self.mdl.B)

    @torch.no_grad()
    def _sgld_inner(self, X: torch.Tensor, r: torch.Tensor, eta: float):
        gw, ga = self._grads_cavity(X, r)
        self.W.add_(gw, alpha=-eta)
        self.a.add_(ga, alpha=-eta)
        self.W.add_(torch.randn_like(self.W), alpha=math.sqrt(2.0*eta))
        self.a.add_(torch.randn_like(self.a), alpha=math.sqrt(2.0*eta))

    @torch.no_grad()
    def _update_rhos_two_group(self):
        """
        Group-ARD update:
          ρ_sig <- EMA( (α0 + B*|S|/2) / (β0 + 0.5 Σ_{j∈S} Σ_b w_{b,j}^2) )
          ρ_noi <- EMA( (α0 + B*(d-|S|)/2) / (β0 + 0.5 Σ_{j∉S} Σ_b w_{b,j}^2) )
        """
        if not self.ard2.use_ard:
            return
        B = self.mdl.B
        alpha_sig = self.ard2.alpha0 + 0.5 * B * max(self.n_sig, 0)
        alpha_noi = self.ard2.alpha0 + 0.5 * B * max(self.n_noi, 0)

        # sufficient stats
        if self.n_sig > 0:
            ss_sig = 0.5 * (self.W[:, self.mask_sig.bool()].pow(2)).sum()
        else:
            ss_sig = torch.tensor(0.0, device=self.device)
        if self.n_noi > 0:
            ss_noi = 0.5 * (self.W[:, self.mask_noi.bool()].pow(2)).sum()
        else:
            ss_noi = torch.tensor(0.0, device=self.device)

        beta_sig = self.beta0 + ss_sig
        beta_noi = self.beta0 + ss_noi

        rho_sig_hat = alpha_sig / torch.clamp(beta_sig, min=1e-12)
        rho_noi_hat = alpha_noi / torch.clamp(beta_noi, min=1e-12)

        # EMA + clamp
        ema = self.ard2.ema
        self.rho_sig = (1.0 - ema) * self.rho_sig + ema * rho_sig_hat
        self.rho_noi = (1.0 - ema) * self.rho_noi + ema * rho_noi_hat
        self.rho_sig = torch.clamp(self.rho_sig, min=self.ard2.rho_min, max=self.ard2.rho_max)
        self.rho_noi = torch.clamp(self.rho_noi, min=self.ard2.rho_min, max=self.ard2.rho_max)

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
        f_mean = torch.zeros(P,1, device=self.device, dtype=torch.float32)

        hist = {
            "iter": [], "train_mse": [], "accept": [],
            "half_mse_modes": [], "half_noise": [],
            "half_mse_total_ms": [], "half_mse_empirical": [], "m_S": [],
            "rho_sig": [], "rho_noi": [], "elapsed_s": []
        }
        t0 = time.time()

        for it in range(1, self.algo.outer_steps+1):
            # fixed residual
            r = (y - f_mean).to(torch.float32)

            # inner sampler
            acc = 0.0
            for _ in range(self.algo.inner_mala_steps):
                if self.algo.use_mala:
                    acc += self._mala_inner(X, r, self.algo.step_size)
                else:
                    self._sgld_inner(X, r, self.algo.step_size)
            if self.algo.use_mala:
                acc /= max(1, self.algo.inner_mala_steps)

            # ARD two-group update
            if self.ard2.use_ard and (it % self.ard2.update_every == 0):
                self._update_rhos_two_group()

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
                hist["iter"].append(it)
                hist["train_mse"].append(train_mse)
                hist["accept"].append(acc if self.algo.use_mala else 0.0)
                hist["half_mse_modes"].append(ev["half_mse_modes"])
                hist["half_noise"].append(ev["half_noise"])
                hist["half_mse_total_ms"].append(ev["half_mse_total_ms"])
                hist["half_mse_empirical"].append(ev["half_mse_empirical"])
                hist["m_S"].append(ev["m_S"])
                hist["rho_sig"].append(float(self.rho_sig.item()))
                hist["rho_noi"].append(float(self.rho_noi.item()))
                hist["elapsed_s"].append(round(time.time()-t0,2))
                print(json.dumps({
                    "iter": it, "train_mse": train_mse, "accept": acc if self.algo.use_mala else 0.0,
                    "half_mse_modes": ev["half_mse_modes"], "half_noise": ev["half_noise"],
                    "half_mse_total_ms": ev["half_mse_total_ms"], "half_mse_empirical": ev["half_mse_empirical"],
                    "m_S": ev["m_S"], "B": self.mdl.B, "N": self.mdl.N, "gamma": self.mdl.gamma,
                    "kappa": self.kappa, "rho_sig": float(self.rho_sig.item()),
                    "rho_noi": float(self.rho_noi.item()),
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
                    "ard2": {
                        "alpha0": self.ard2.alpha0, "beta0": self.beta0,
                        "ema": self.ard2.ema, "update_every": self.ard2.update_every,
                        "rho_min": self.ard2.rho_min, "rho_max": self.ard2.rho_max,
                        "use_ard": self.ard2.use_ard
                    }
               }}
        path = os.path.join(out_dir, f"rs_cavity_explicit_aw_ard2_{tag}_Ptr{P}_Peval{self.algo.P_eval}_kap{self.kappa:.3e}_N{self.mdl.N}_B{self.mdl.B}_g{self.mdl.gamma}.json")
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

    # hyperparams
    mdl = Model(d=d, B=128, N=512, gamma=0.5, sigma_a=1.0, sigma_w=1.0, act="relu")
    algo = Algo(
        outer_steps=50000, inner_mala_steps=600, step_size=5e-7, use_mala=False,
        log_every=10, cg_like_update=False, field_blend=0.5,
        P_eval=50_000, batch_eval=50_000,
        grad_clip_norm=None
    )
    kappa = 7.5e-3

    # two-group ARD config
    ard2 = ARD2(use_ard=True, alpha0=1e-2, ema=0.25, update_every=1,
                rho_min=1e-6, rho_max=1e6, beta0=None)

    out_dir = "/home/goring/mean_field_langevin/MCMC_composite/resutls_mf1/test"
    solver = RSCavityExplicit(mdl, algo, kappa, device, teacher_sets=sets, ard2=ard2)
    solver.run(X, y, out_dir, tag=f"P{P_train}_kap{kappa:.3e}")
