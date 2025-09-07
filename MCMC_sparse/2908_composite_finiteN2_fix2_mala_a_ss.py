# rs_cavity_explicit_aw.py
# RS self-consistent cavity with explicit (a,w) and per-particle MALA/SGLD.
# Field update: <f> = (N^{1-γ}/B) Φ a  (Monte-Carlo estimate of N^{1-γ} E[a φ])
#
# Single-neuron cavity energy at fixed residual r = y - <f>:
#   E_b(w,a) = d/(2σ_w^2)||w||^2 + 0.5 * λ_b * a_b^2 + (1/(2κ^2 P)) Σμ (rμ - (a_b/N^γ) φ(w^T xμ))^2
# where λ_b is the (possibly gated) prior precision on a_b:
#   - no gating:        λ_b = 1/σ_a^2
#   - soft spike–slab:  λ_b = q_b/σ_slab^2 + (1-q_b)/σ_sp^2,   q_b = q(z_b=1)
#   - hard top-K:       λ_b = 1/σ_slab^2 for top-K by saliency, else 1/σ_sp^2
#
# H100-friendly: TF32 matmuls, vectorized ops.

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
    inner_mala_steps: int = 1            # MALA/SGLD steps per outer iter
    step_size: float = 1e-5              # MALA/SGLD step size
    use_mala: bool = True                # False -> SGLD (no accept/reject)
    log_every: int = 10
    cg_like_update: bool = False
    field_blend: float = 1.0             # 1.0 -> full overwrite, <1.0 -> under-relax
    batch_eval: int = 262_144
    P_eval: int = 50_000

@dataclass
class SpikeSlabConfig:
    mode: str = "hard"                   # "none", "soft", or "hard"
    # --- hard top-K ---
    K: Optional[int] = None              # if None -> K = max(1, B//8)
    saliency: str = "phi_r_whiten"       # "phi_r_whiten", "phi_r", or "abs_a"
    # --- soft spike–slab (EM gate on prior) ---
    pi: float = 0.1                      # prior slab probability
    sigma_sp: float = 1e-3               # spike std (small)
    sigma_slab: float = 1.0              # slab std  (large)
    temperature: float = 1.0             # gate sharpening (<=1 sharper)

# ----------------------------- core -------------------------------

class RSCavityExplicit:
    def __init__(self, mdl: Model, algo: Algo, kappa: float, device: torch.device,
                 teacher_sets: Optional[List[torch.Tensor]] = None,
                 spike_slab: Optional[SpikeSlabConfig] = None):
        self.mdl, self.algo, self.kappa = mdl, algo, float(kappa)
        self.device = device
        self.sets = teacher_sets or []
        self.ss = spike_slab or SpikeSlabConfig(mode="none")

        # params (particles)
        self.W = torch.randn(mdl.B, mdl.d, device=device) * (mdl.sigma_w / math.sqrt(mdl.d))
        self.a = torch.randn(mdl.B, 1, device=device) * mdl.sigma_a

        # per-particle prior precision on a (λ_b). Start at plain Gaussian.
        self._lambda_a = torch.full((mdl.B,), 1.0/(mdl.sigma_a**2), device=device, dtype=torch.float32)
        # for soft gates, keep q(z=1)
        self._qz = torch.full((mdl.B,), float(self.ss.pi), device=device, dtype=torch.float32)

        # mixed precision / TF32
        try:
            torch.set_float32_matmul_precision('high')
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    # ---------- spike–slab / top-K gating on a ----------

    @torch.no_grad()
    def _update_lambda_a(self, X: torch.Tensor, r: torch.Tensor):
        """
        Update per-particle prior precision λ_b on a_b according to selected gating mode.
        Hard top-K uses saliency; soft spike–slab uses EM gate q(z_b=1).
        """
        mode = (self.ss.mode or "none").lower()
        if mode == "none":
            self._lambda_a.fill_(1.0/(self.mdl.sigma_a**2))
            return

        if mode == "soft":
            # q(z=1) posterior under mixture prior given current a_b (independent of r)
            a2 = (self.a[:,0] * self.a[:,0]).to(torch.float32)
            eps = 1e-12
            pi = float(self.ss.pi)
            sig_sp2 = float(self.ss.sigma_sp**2)
            sig_slab2 = float(self.ss.sigma_slab**2)
            # logit = log(pi/(1-pi)) + 0.5 * a^2 * (1/σ_sp^2 - 1/σ_slab^2)
            logit = math.log(pi + eps) - math.log(max(1.0-pi, eps)) + 0.5 * a2 * (1.0/sig_sp2 - 1.0/sig_slab2)
            if self.ss.temperature != 1.0:
                logit = logit / max(self.ss.temperature, 1e-6)
            qz = torch.sigmoid(logit)
            self._qz.copy_(qz)
            # expected precision E_z[1/σ_z^2] = q/σ_slab^2 + (1-q)/σ_sp^2
            lam = qz*(1.0/sig_slab2) + (1.0-qz)*(1.0/sig_sp2)
            self._lambda_a.copy_(lam.to(torch.float32))
            return

        if mode == "hard":
            # choose top-K by saliency; active -> slab precision, inactive -> spike precision
            B = self.mdl.B
            K = self.ss.K if (self.ss.K is not None) else max(1, B//8)
            sig_sp2 = float(self.ss.sigma_sp**2)
            sig_slab2 = float(self.ss.sigma_slab**2)
            sal = self._saliency(X, r)  # (B,)
            # top-K mask
            K = min(K, B)
            topk = torch.topk(sal, k=K, sorted=False)
            mask = torch.zeros(B, device=self.device, dtype=torch.bool)
            mask[topk.indices] = True
            lam = torch.where(mask, torch.tensor(1.0/sig_slab2, device=self.device),
                              torch.tensor(1.0/sig_sp2, device=self.device))
            self._lambda_a.copy_(lam.to(torch.float32))
            # for logging / curiosity, set qz ~ {0,1}
            self._qz.copy_(mask.to(torch.float32))
            return

        raise ValueError(f"Unknown spike–slab mode: {self.ss.mode}")

    @torch.no_grad()
    def _saliency(self, X: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Saliency per particle for hard top-K selection.
        Options:
          - 'phi_r_whiten': |<φ_b, r>| / sqrt(<φ_b^2>)
          - 'phi_r':        |<φ_b, r>|
          - 'abs_a':        |a_b|
        """
        choice = (self.ss.saliency or "phi_r_whiten").lower()
        if choice == "abs_a":
            return self.a[:,0].abs().to(torch.float32)

        # need φ and correlations
        z = X @ self.W.t().contiguous()                 # (P,B)
        Phi = activation(z, self.mdl.act).to(torch.float32)
        C1 = (Phi.t() @ r).view(-1)                    # (B,)
        if choice == "phi_r":
            return C1.abs()
        # whitened by diag(Φ^T Φ)
        C2 = (Phi * Phi).sum(dim=0) + 1e-8             # (B,)
        return (C1.abs() / C2.sqrt()).to(torch.float32)

    # ---------- model mechanics ----------

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
        Uses current per-particle precision self._lambda_a on a.
        """
        P = r.shape[0]
        d, sigw = self.mdl.d, self.mdl.sigma_w

        prior_w_b = 0.5 * (d/(sigw**2)) * (self.W * self.W).sum(dim=1)         # (B,)
        prior_a_b = 0.5 * self._lambda_a * (self.a[:,0]**2)                    # (B,)

        # Separable data term per b (constant ||r||^2 dropped)
        C1 = (Phi.t() @ r).view(-1)                     # (B,)
        C2 = (Phi * Phi).sum(dim=0)                     # (B,)
        a_flat = self.a[:,0]
        N_gamma = self.mdl.N ** self.mdl.gamma
        data_b = ( - (a_flat / N_gamma) * C1 + 0.5 * (a_flat*a_flat / (N_gamma**2)) * C2 ) / (self.kappa**2 * P)

        return (prior_w_b + prior_a_b + data_b).to(torch.float32)

    @torch.no_grad()
    def _grads_cavity(self, X: torch.Tensor, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized grads of E_b at fixed residual r, using current λ_b on a_b:
        ∇_a E_b = λ_b a_b  - (1/(κ^2 P N^γ)) <φ_b,r>  + (1/(κ^2 P N^{2γ})) <φ_b^2> a_b
        ∇_w E_b = (d/σ_w^2) w_b  - (1/(κ^2 P)) Σμ (rμ - (a_b/N^γ) φ_{μb}) (a_b/N^γ) φ'(z_{μb}) x_μ
        """
        P = X.shape[0]
        z = X @ self.W.t().contiguous()                # (P,B)
        Phi = activation(z, self.mdl.act).to(torch.float32)
        dPhi = act_prime(z, self.mdl.act).to(torch.float32)

        # a-gradient
        N_gamma = self.mdl.N ** self.mdl.gamma
        term1 = self._lambda_a * self.a[:,0]                                              # (B,)
        term2 = - (Phi.t() @ r).view(-1) / (self.kappa**2 * P * N_gamma)                  # (B,)
        term3 = ((Phi*Phi).sum(dim=0) / (self.kappa**2 * P * (N_gamma**2))) * self.a[:,0] # (B,)
        grad_a = (term1 + term2 + term3).view(-1,1)                                       # (B,1)

        # w-gradient
        # M_{μb} = (rμ - (a_b/N^γ) φ_{μb}) * (a_b/N^γ) * φ'(z_{μb})
        M = (r.view(-1, 1) - (self.a.view(1, -1) / N_gamma) * Phi) * dPhi * (self.a.view(1, -1) / N_gamma)
        G = -(1.0/(self.kappa**2 * P)) * M.t().matmul(X)                                   # (B,d)
        grad_w = G + (self.mdl.d/(self.mdl.sigma_w**2)) * self.W

        return grad_w.to(torch.float32), grad_a.to(torch.float32)

    @torch.no_grad()
    def _mala_inner(self, X: torch.Tensor, r: torch.Tensor, eta: float) -> float:
        """
        One MALA step over all particles (independently) at fixed residual r.
        Returns mean accept rate.
        """
        gw, ga = self._grads_cavity(X, r)

        # current energies (need Φ)
        z = X @ self.W.t().contiguous()
        Phi = activation(z, self.mdl.act).to(torch.float32)
        E_curr = self._energy_per_particle(Phi, r)                          # (B,)

        # propose
        xi_w = torch.randn_like(self.W); xi_a = torch.randn_like(self.a)
        Wp = (self.W - eta * gw + math.sqrt(2.0*eta) * xi_w).requires_grad_(False)
        ap = (self.a - eta * ga + math.sqrt(2.0*eta) * xi_a).requires_grad_(False)

        # grads and energies at proposal (swap-in trick)
        W_saved, a_saved = self.W, self.a
        self.W, self.a = Wp, ap
        gw_p, ga_p = self._grads_cavity(X, r)
        zp = X @ self.W.t().contiguous()
        Phip = activation(zp, self.mdl.act).to(torch.float32)
        E_prop = self._energy_per_particle(Phip, r)
        self.W, self.a = W_saved, a_saved

        # log proposal densities for joint (W,a)
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
            "elapsed_s": [], "active_frac": []
        }
        t0 = time.time()

        for it in range(1, self.algo.outer_steps+1):
            # fixed residual for cavity step
            r = (y - f_mean).to(torch.float32)

            # --- update spike–slab / top-K gates on a (sets λ_b) ---
            self._update_lambda_a(X, r)

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
                active_frac = float(self._qz.mean().item()) if self.ss.mode != "none" else 1.0
                hist["iter"].append(it)
                hist["train_mse"].append(train_mse)
                hist["accept"].append(acc)
                hist["half_mse_modes"].append(ev["half_mse_modes"])
                hist["half_noise"].append(ev["half_noise"])
                hist["half_mse_total_ms"].append(ev["half_mse_total_ms"])
                hist["half_mse_empirical"].append(ev["half_mse_empirical"])
                hist["m_S"].append(ev["m_S"])
                hist["elapsed_s"].append(round(time.time()-t0,2))
                hist["active_frac"].append(active_frac)
                print(json.dumps({
                    "iter": it, "train_mse": train_mse, "accept": acc,
                    "half_mse_modes": ev["half_mse_modes"], "half_noise": ev["half_noise"],
                    "half_mse_total_ms": ev["half_mse_total_ms"], "half_mse_empirical": ev["half_mse_empirical"],
                    "m_S": ev["m_S"], "active_frac": active_frac,
                    "B": self.mdl.B, "N": self.mdl.N, "gamma": self.mdl.gamma,
                    "kappa": self.kappa, "elapsed_s": round(time.time()-t0,2),
                    "gate_mode": self.ss.mode
                }))

        # save
        tag = tag or time.strftime("%Y%m%d_%H%M%S")
        out = {"summary": {
                    "train_mse_last": hist["train_mse"][-1],
                    "accept_last": hist["accept"][-1],
                    "P_eval": self.algo.P_eval
                },
               "traj": hist,
               "config": {"model": self.mdl.__dict__, "algo": self.algo.__dict__,
                          "kappa": self.kappa, "spike_slab": self.ss.__dict__}}
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
    P_train = 500
    X, y = generate_parity(P_train, d, sets, device)

    # hyperparams
    mdl = Model(d=d, B=512, N=512, gamma=0.5, sigma_a=1.0, sigma_w=1.0, act="relu")
    algo = Algo(
        outer_steps=50000, inner_mala_steps=600, step_size=5e-7, use_mala=False,
        log_every=10, cg_like_update=False, field_blend=0.5,
        P_eval=50_000, batch_eval=50000
    )
    kappa = 7.5e-3

    # ----- choose gating -----
    # Hard top-K (winner-take-few). K defaults to B//8 if None.
    #ss = SpikeSlabConfig(
    #    mode="hard",
    #    K=None,                       # or set e.g. 16
    #    saliency="phi_r_whiten",      # "phi_r_whiten" | "phi_r" | "abs_a"
    #    sigma_sp=1e-3,
    #    sigma_slab=1.0
    #)
    # Soft spike–slab alternative:
    ss = SpikeSlabConfig(mode="soft", pi=0.1, sigma_sp=1e-3, sigma_slab=1.0, temperature=1.0)

    out_dir = "/home/goring/mean_field_langevin/MCMC_composite/resutls_mf1/test"
    solver = RSCavityExplicit(mdl, algo, kappa, device, teacher_sets=sets, spike_slab=ss)
    solver.run(X, y, out_dir, tag=f"P{P_train}_kap{kappa:.3e}")
