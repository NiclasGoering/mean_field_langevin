# rs_cavity_explicit_aw_corrected.py
# RS self-consistent cavity with explicit (a,w).
# Self-consistency: <f> = (N^{1-γ}/B) Φ a  (Monte-Carlo estimate of N^{1-γ} E[a φ])
#
# Single-neuron global energy (no cavity expansion):
#   E(w,a) = d/(2σ_w^2)||w||^2 + (1/(2σ_a^2)) a^2 + (1/(2κ^2 P)) || y - s Φ a ||^2
# with s = N^{1-γ}/B.
#
# Correct global gradients (no leave-one-out subtraction):
#   ∇_a E = (1/σ_a^2) a  - (s/(κ^2 P)) Φ^T r
#   ∇_w E_b = (d/σ_w^2) w_b  - (s a_b/(κ^2 P)) Σ_μ r_μ φ'(z_{μb}) x_μ
#
# This version uses SGLD (temperature 1) for the inner sampler by default.
# If you want MALA: implement per-particle ΔE and proposal means using r' = r - s(a'φ' - aφ)
# sequentially (chunked) to keep the MH ratio exact.

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

# ----------------------------- config -----------------------------

@dataclass
class Model:
    d: int = 35
    B: int = 16384       # number of particles
    N: int = 512         # network N in f = N^{-γ} Σ a φ
    gamma: float = 0.5
    sigma_a: float = 1.0 # prior std for a (each a ~ N(0, σ_a^2))
    sigma_w: float = 1.0 # prior std scale for w entries: each w_j ~ N(0, σ_w^2/d)
    act: str = "relu"

@dataclass
class Algo:
    outer_steps: int = 2000              # RS fixed-point iterations
    inner_mala_steps: int = 1            # inner steps per outer iter
    step_size: float = 1e-5              # SGLD/Langevin step size
    use_mala: bool = False               # default to SGLD (safer & correct)
    log_every: int = 10
    cg_like_update: bool = False         # optional underrelax
    field_blend: float = 1.0             # 1.0 -> full overwrite
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
        # Each entry variance is σ_w^2/d  -> std = σ_w/√d
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
    def _scale(self) -> float:
        # s = N^{1-γ} / B
        return (self.mdl.N ** (1.0 - self.mdl.gamma)) / float(self.mdl.B)

    @torch.no_grad()
    def _field_from_particles(self, X: torch.Tensor) -> torch.Tensor:
        """<f> = s Φ(W) a on the training points (global scaling)."""
        z = X @ self.W.t().contiguous()                     # (P,B)
        Phi = activation(z, self.mdl.act).to(torch.float32) # (P,B)
        s = self._scale()
        return (s * Phi @ self.a).to(torch.float32)         # (P,1)

    @torch.no_grad()
    def _grads_global(self, X: torch.Tensor, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Correct global gradients of E at fixed global residual r = y - s Φ a.
        ∇_a E = (1/σ_a^2) a  - (s/(κ^2 P)) Φ^T r
        ∇_w E_b = (d/σ_w^2) w_b  - (s a_b/(κ^2 P)) Σμ rμ φ'(z_{μb}) x_μ
        """
        P = X.shape[0]
        z = X @ self.W.t().contiguous()                # (P,B)
        Phi = activation(z, self.mdl.act).to(torch.float32)
        dPhi = act_prime(z, self.mdl.act).to(torch.float32)

        s = self._scale()

        # grad wrt a
        term1 = (1.0 / (self.mdl.sigma_a**2)) * self.a[:,0]                 # (B,)
        term2 = - s * (Phi.t() @ r).view(-1) / (self.kappa**2 * P)          # (B,)
        grad_a = (term1 + term2).view(-1,1)                                 # (B,1)

        # grad wrt w
        # (r * dPhi) is (P,B). Multiply each column by s * a_b, then X^T
        M = (r.view(-1,1) * dPhi) * (self.a.view(1,-1) * s)                 # (P,B)
        G = -(1.0/(self.kappa**2 * P)) * M.t().matmul(X)                    # (B,d)
        grad_w = G + (self.mdl.d/(self.mdl.sigma_w**2)) * self.W            # (B,d)

        return grad_w.to(torch.float32), grad_a.to(torch.float32)

    @torch.no_grad()
    def _mala_inner(self, X: torch.Tensor, r: torch.Tensor, eta: float) -> float:
        """
        Placeholder-safe MALA: for now, use SGLD step (ULA) and report accept=1.0.
        Implementing exact per-particle MALA requires sequential (chunked) MH with
        ΔE computed from dF = s(a'φ' - aφ). See comments below if you want to extend.
        """
        # SGLD/ULA step (temperature 1 for this energy)
        gw, ga = self._grads_global(X, r)
        self.W.add_(gw, alpha=-eta)
        self.a.add_(ga, alpha=-eta)
        self.W.add_(torch.randn_like(self.W), alpha=math.sqrt(2.0*eta))
        self.a.add_(torch.randn_like(self.a), alpha=math.sqrt(2.0*eta))
        return 1.0

        # -------------------------- sketch for exact MALA (if you want it) --------------------------
        # 1) compute gw,ga at current (global r)
        # 2) propose Wp = W - η gw + sqrt(2η) ξ_w, ap analogously
        # 3) process particles in chunks:
        #    - compute φ_old = Φ[:, b], φ_new via X @ Wp_chunk^T, dF_b = s(a'_b φ_new - a_b φ_old)
        #    - ΔE_data_b = ( - <r, dF_b> + 0.5 ||dF_b||^2 )/(κ^2 P)
        #    - ΔE_prior_b = 0.5( (a'_b)^2 - a_b^2 )/σ_a^2 + 0.5 d (||w'_b||^2 - ||w_b||^2)/σ_w^2
        #    - build proposal means m_b = (a_b - η ga_b, w_b - η gw_b) and m'_b using r' = r - dF_b
        #    - log_q terms: -||θ'_b - m_b||^2/(4η) + ||θ_b - m'_b||^2/(4η)
        #    - accept per b using log_acc = -(ΔE_b) + (log_q_rev - log_q_fwd); if accepted, update r <- r - dF_b
        # 4) write accepted Wp,ap back.

    @torch.no_grad()
    def _sgld_inner(self, X: torch.Tensor, r: torch.Tensor, eta: float):
        gw, ga = self._grads_global(X, r)
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

        s = self._scale()
        a = self.a.detach()

        g = torch.Generator(device=device).manual_seed(1234567)
        for start in range(0, P_eval, chunk):
            n = min(chunk, P_eval-start)
            Xc = (torch.randint(0,2,(n,d),generator=g,device=device,dtype=torch.int8).float()*2.0-1.0)
            z = Xc @ self.W.t().contiguous()
            Phi = activation(z, self.mdl.act).to(torch.float32)
            f = (s * Phi @ a)[:,0]
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
            # global residual (consistent)
            r = (y - f_mean).to(torch.float32)

            # inner sampler at fixed r
            acc = 0.0
            for _ in range(self.algo.inner_mala_steps):
                if self.algo.use_mala:
                    acc += self._mala_inner(X, r, self.algo.step_size)
                else:
                    self._sgld_inner(X, r, self.algo.step_size)
                    acc += 1.0
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
    P_train = 2133*5  # match your GD example; set 1000 if you want
    X, y = generate_parity(P_train, d, sets, device)

    # hyperparams
    mdl = Model(d=d, B=2048, N=512, gamma=0.5, sigma_a=1.0, sigma_w=1.0, act="relu")
    algo = Algo(
        outer_steps=50000,      # you can reduce while testing
        inner_mala_steps=400,
        step_size=2e-6,
        use_mala=False,         # SGLD default (correct for the global energy)
        log_every=10,
        cg_like_update=False,
        field_blend=1.0,
        P_eval=50_000,
        batch_eval=50_000
    )
    kappa = 7.5e-3

    out_dir = "/home/goring/mean_field_langevin/MCMC_composite/resutls_mf1/test_corrected"
    solver = RSCavityExplicit(mdl, algo, kappa, device, teacher_sets=sets)
    solver.run(X, y, out_dir, tag=f"P{P_train}_kap{kappa:.3e}")
