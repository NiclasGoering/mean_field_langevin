# rs_cavity_explicit_aw.py
# RS self-consistent cavity with explicit (a,w) and per-particle MALA.
# Field update: <f> = (N^{1-γ}/B) Φ a  (Monte-Carlo estimate of N^{1-γ} E[a φ])
#
# Single-neuron cavity energy at fixed residual r = y - <f>:
#   E_b(w,a) = d/(2σ_w^2)||w||^2 + N^γ/(2σ_a^2) a^2 + (1/(2κ^2 P)) Σμ (rμ - a φ(w^T xμ))^2
#
# This keeps 'a' explicit, so κ→0 drives a handful of amplitudes large (no κ-cancellation).
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

@dataclass
class Algo:
    outer_steps: int = 2000              # RS fixed-point iterations
    inner_mala_steps: int = 1            # MALA steps per outer iter
    step_size: float = 1e-5              # MALA/SGLD step size
    use_mala: bool = True                # False -> SGLD (no accept/reject)
    log_every: int = 10
    cg_like_update: bool = False         # optional: blend field with old (<f> ← 0.5 old + 0.5 new)
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

        # mixed precision + TF32
        try:
            torch.set_float32_matmul_precision('high')
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    @torch.no_grad()
    def _field_from_particles(self, X: torch.Tensor) -> torch.Tensor:
        """<f> = (N^{1-γ}/B) Φ(W) a  on the training points."""
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
        k2 = self.kappa * self.kappa

        prior_w_b = 0.5 * (d/(sigw**2)) * (self.W * self.W).sum(dim=1)         # (B,)
        #prior_a_b = 0.5 * ((self.mdl.N**self.mdl.gamma)/(siga**2)) * (self.a[:,0]**2)  # (B,)
        prior_a_b = 0.5 * (1.0 / (siga**2)) * (self.a[:,0]**2)  # (B,)

        # data term per particle: (1/(2κ^2 P)) ||r - Φ_b a_b||^2
        Ra = r - Phi @ self.a                                                  # (P,1)
        # expand contribution per b: ||r - Σ_b a_b φ_b||^2 = ||r||^2 - 2 a_b <φ_b,r> + a_b^2 <φ_b^2>
        # but we compute directly (fast enough) via columnwise squares:
        # Compute per-column squared residual contributions:
        # For acceptance/proposal we only need full per-particle energies, so we can use:
        # data_tot = (1/(2 κ^2 P)) * ||r - Φ a||^2, which is *global*, not separable.
        # Instead compute per-b using expansion (stable and separable):
        r2 = (r*r).mean() * (P)  # scale back by P later; constant across b -> cancels in MALA; we can drop it
        # Separable part:
        C1 = (Phi.t() @ r).view(-1)                     # (B,)
        C2 = (Phi * Phi).mean(dim=0) * P                # (B,)
        a_flat = self.a[:,0]

        # AFTER
        N_gamma = self.mdl.N ** self.mdl.gamma
        data_b = ( - (a_flat / N_gamma) * C1 + 0.5 * (a_flat*a_flat / (N_gamma**2)) * C2 ) / (self.kappa**2 * P)
        #data_b = ( - a_flat * C1 + 0.5 * (a_flat*a_flat) * C2 ) / (self.kappa**2 * P)
        # (constant r^2/(2κ^2 P) dropped since it cancels in MH ratio)
        return prior_w_b + prior_a_b + data_b

    @torch.no_grad()
    def _grads_cavity(self, X: torch.Tensor, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized grads of E_b at fixed residual r.
        ∇_a E = (N^γ/σ_a^2) a  - (1/(κ^2 P)) Φ^T r  + (1/(κ^2 P)) diag(Φ^T Φ) ⊙ a
        ∇_w E_b = (d/σ_w^2) w_b  - (a_b/(κ^2 P)) Σμ (rμ - a_b φ_{μb}) φ'(z_{μb}) x_μ
        """
        P = X.shape[0]
        z = X @ self.W.t().contiguous()                # (P,B)
        Phi = activation(z, self.mdl.act).to(torch.float32)
        dPhi = act_prime(z, self.mdl.act).to(torch.float32)

        # a-gradient
        #term1 = (self.mdl.N ** self.mdl.gamma) / (self.mdl.sigma_a**2) * self.a[:,0]                 # (B,)
        term1 = (1.0 / (self.mdl.sigma_a**2)) * self.a[:,0] # CORRECT GRADIENT
        #term2 = - (Phi.t() @ r).view(-1) / (self.kappa**2 * P)                                       # (B,)
        #term3 = ((Phi*Phi).sum(dim=0) / (self.kappa**2 * P)) * self.a[:,0]                           # (B,)

        # AFTER
        N_gamma = self.mdl.N ** self.mdl.gamma
        term2 = - (Phi.t() @ r).view(-1) / (self.kappa**2 * P * N_gamma)
        term3 = ((Phi*Phi).sum(dim=0) / (self.kappa**2 * P * (N_gamma**2))) * self.a[:,0]
        grad_a = (term1 + term2 + term3).view(-1,1)                                                  # (B,1)

        # w-gradient
        Ra = r - Phi @ self.a                                                                        # (P,1)
        # M_{μb} = (rμ - a_b φ_{μb}) * φ'(z_{μb}) * a_b
        #M = (Ra @ torch.ones(1, self.mdl.B, device=self.device))   # (P,B) broadcast r
        #M = (M - Phi * self.a.view(1,-1)) * dPhi * self.a.view(1,-1)                                 # (P,B)
        #G = -(1.0/(self.kappa**2 * P)) * M.t().matmul(X)                                             # (B,d)
        #grad_w = G + (self.mdl.d/(self.mdl.sigma_w**2)) * self.W                                     # (B,d)
        # AFTER
        N_gamma = self.mdl.N ** self.mdl.gamma
        # In the original expression, a_b is used twice. We replace it with a_b / N_gamma
        # The term is proportional to (residual) * a_b/N_gamma * dPhi
        # The residual is (r - a_b/N_gamma * phi)
        M = (r.view(-1, 1) - (self.a.view(1, -1) / N_gamma) * Phi) * dPhi * (self.a.view(1, -1) / N_gamma)
        G = -(1.0/(self.kappa**2 * P)) * M.t().matmul(X)
        grad_w = G + (self.mdl.d/(self.mdl.sigma_w**2)) * self.W

        return grad_w.to(torch.float32), grad_a.to(torch.float32)

    @torch.no_grad()
    def _mala_inner(self, X: torch.Tensor, r: torch.Tensor, eta: float) -> float:
        """
        One MALA step over all particles (independently) at fixed residual r.
        Returns mean accept rate.
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
        # (reuse routines with (Wp, ap) by temporarily swapping)
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
        return n_acc / float(self.mdl.B)

    @torch.no_grad()
    def _sgld_inner(self, X: torch.Tensor, r: torch.Tensor, eta: float):
        gw, ga = self._grads_cavity(X, r)
        self.W.add_(gw, alpha=-eta)
        self.a.add_(ga, alpha=-eta)
        self.W.add_(torch.randn_like(self.W), alpha=math.sqrt(2.0*eta))
        self.a.add_(torch.randn_like(self.a), alpha=math.sqrt(2.0*eta))

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
        bf16 = torch.bfloat16

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
                    "kappa": self.kappa, "elapsed_s": round(time.time()-t0,2)
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
    P_train = 20000
    X, y = generate_parity(P_train, d, sets, device)

    # hyperparams (H100 friendly)
    mdl = Model(d=d, B=128, N=512, gamma=0.5, sigma_a=1.0, sigma_w=1.0, act="relu")
    algo = Algo(
        outer_steps=50000, inner_mala_steps=600, step_size=5e-7, use_mala=False,
        log_every=10, cg_like_update=False, field_blend=0.5,
        P_eval=50_000, batch_eval=50000
    )
    kappa = 7.5e-3

    out_dir = "/home/goring/mean_field_langevin/MCMC_composite/resutls_mf1/test"
    solver = RSCavityExplicit(mdl, algo, kappa, device, teacher_sets=sets)
    solver.run(X, y, out_dir, tag=f"P{P_train}_kap{kappa:.3e}")
