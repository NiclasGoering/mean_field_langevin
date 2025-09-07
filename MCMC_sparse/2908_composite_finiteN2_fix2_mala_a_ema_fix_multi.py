# rs_cavity_explicit_aw_ard_multi_gpu_earlystop_memsafe_fix.py
# Multi-GPU batched RS cavity with explicit (a,w), original MALA, ARD,
# strict per-experiment seeds, EARLY STOP, and memory-safe chunking.
# Fixes: removed [:,:,0] after P-sum; added _update_rho_ard(); replaced einsum digits with letters.

import os, time, math, json, random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

# ============================ utils ============================

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
    if S.numel() == 0:
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

def generate_parity(P: int, d: int, sets: List[torch.Tensor], device, dtype, seed: int):
    g = torch.Generator(device=device).manual_seed(int(seed))
    X = (torch.randint(0,2,(P,d),generator=g,device=device,dtype=torch.int8).to(dtype) * 2.0 - 1.0)
    Ccols = [parity_character(X, S) for S in sets]
    C = torch.stack(Ccols, dim=1) if Ccols else torch.zeros(P,0,device=device, dtype=dtype)
    y = C.sum(dim=1, keepdim=True)
    return X, y

def build_experiment_grid(P_list: List[int], kappa_list: List[float], num_exp: int, base_seed: int):
    C1 = 0x9E3779B1; C2 = 0x85EBCA6B; C3 = 0xC2B2AE35
    exps = []; gid = 0
    for P in P_list:
        for kap in kappa_list:
            for rep in range(num_exp):
                ds_seed   = int(base_seed + gid*C1 + 11)
                init_seed = int(base_seed + gid*C2 + 17)
                noise_seed= int(base_seed + gid*C3 + 23)
                exps.append({
                    "gid": gid,
                    "P": int(P),
                    "kappa": float(kap),
                    "rep": rep,
                    "seeds": {"dataset": ds_seed, "init": init_seed, "noise": noise_seed}
                })
                gid += 1
    return exps

def shard_experiments(exps: List[Dict], num_devices: int, strategy: str = "balance_P"):
    shards = [[] for _ in range(num_devices)]
    if strategy == "round_robin":
        for i, e in enumerate(exps): shards[i % num_devices].append(e)
        return shards
    loads = [0]*num_devices
    for e in sorted(exps, key=lambda z: -z["P"]):
        j = min(range(num_devices), key=lambda k: loads[k])
        shards[j].append(e); loads[j] += e["P"]
    return shards

# ============================ config ============================

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
class ARD:
    use_ard: bool = True
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
    P_eval: int = 50_000
    batch_eval: int = 4096
    B_chunk: int = 2048
    B_chunk_eval: Optional[int] = None
    grad_clip_norm: Optional[float] = None
    use_float64: bool = False
    kill_nan_particles: bool = True
    max_abs_w: Optional[float] = None
    max_abs_a: Optional[float] = None
    max_l2_w: Optional[float] = None
    max_l2_a: Optional[float] = None
    nan_reinit_std_scale: float = 1.0
    log_bad_counts: bool = True
    early_stop_enabled: bool = True
    early_stop_threshold: float = 0.04
    early_stop_patience: int = 2000

# ============================ batched solver (per GPU) ============================

class RSCavityExplicitBatch:
    def __init__(self, mdl: Model, algo: Algo, device: torch.device,
                 X: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, P_counts: torch.Tensor,
                 kappa_vec: torch.Tensor,
                 teacher_sets: Optional[List[torch.Tensor]] = None,
                 ard: Optional[ARD] = None,
                 init_seeds: Optional[List[int]] = None,
                 noise_seeds: Optional[List[int]] = None):
        self.mdl, self.algo = mdl, algo
        self.device = device
        self.sets = teacher_sets or []
        self.ard = ard or ARD()
        self.dtype = torch.float64 if algo.use_float64 else torch.float32

        self.X = X.to(self.dtype)         # (E,P_max,d)
        self.y = y.to(self.dtype)         # (E,P_max,1)
        self.mask = mask.bool()           # (E,P_max)
        self.P_counts = P_counts.to(self.dtype)     # (E,)
        self.kappa = kappa_vec.to(self.dtype)       # (E,)
        self.E = self.X.shape[0]
        self.P_max = self.X.shape[1]

        self._Bck = int(self.algo.B_chunk)
        self._Bck_eval = int(self.algo.B_chunk_eval) if self.algo.B_chunk_eval is not None else int(self.algo.B_chunk)
        self._Pck_eval = int(self.algo.batch_eval)

        self._init_seeds = list(init_seeds) if init_seeds is not None else [12345 + i for i in range(self.E)]
        self._noise_seeds= list(noise_seeds) if noise_seeds is not None else [54321 + i for i in range(self.E)]
        self._rng_step = 0

        E, B, d = self.E, self.mdl.B, self.mdl.d
        self.W = torch.empty(E, B, d, device=device, dtype=self.dtype)
        self.a = torch.empty(E, B, 1, device=device, dtype=self.dtype)
        for e in range(E):
            g = torch.Generator(device=device).manual_seed(int(self._init_seeds[e]))
            self.W[e] = torch.randn((B, d), generator=g, device=device, dtype=self.dtype) * (mdl.sigma_w / math.sqrt(mdl.d))
            self.a[e] = torch.randn((B, 1), generator=g, device=device, dtype=self.dtype) * mdl.sigma_a

        rho0 = (mdl.d / (mdl.sigma_w**2))
        self.rho = torch.full((E, d), rho0, device=device, dtype=self.dtype)
        self.beta0 = float(self.ard.alpha0 / float(rho0)) if self.ard.beta0 is None else float(self.ard.beta0)

        try:
            torch.set_float32_matmul_precision('high')
            torch.backends.cuda.matmul.allow_tf32 = not algo.use_float64
            torch.backends.cudnn.allow_tf32 = not algo.use_float64
        except Exception:
            pass

    @torch.no_grad()
    def _reinit_particles(self, mask_bad: torch.Tensor):
        if not mask_bad.any(): return 0
        n = int(mask_bad.sum().item())
        sw = self.mdl.sigma_w * self.algo.nan_reinit_std_scale
        sa = self.mdl.sigma_a * self.algo.nan_reinit_std_scale
        new_W = torch.randn(n, self.mdl.d, device=self.device, dtype=self.dtype) * (sw / math.sqrt(self.mdl.d))
        new_a = torch.randn(n, 1, device=self.device, dtype=self.dtype) * sa
        self.W[mask_bad] = new_W; self.a[mask_bad] = new_a
        return n

    @torch.no_grad()
    def _clamp_params(self):
        if self.algo.max_abs_w is not None: self.W.clamp_(-self.algo.max_abs_w, self.algo.max_abs_w)
        if self.algo.max_abs_a is not None: self.a.clamp_(-self.algo.max_abs_a, self.algo.max_abs_a)
        if self.algo.max_l2_w is not None:
            norms = torch.norm(self.W, dim=2, keepdim=True) + 1e-12
            scale = torch.clamp(self.algo.max_l2_w / norms, max=1.0)
            self.W.mul_(scale)
        if self.algo.max_l2_a is not None:
            norms = torch.sqrt((self.a*self.a).sum(dim=2, keepdim=True)) + 1e-12
            scale = torch.clamp(self.algo.max_l2_a / norms, max=1.0)
            self.a.mul_(scale)

    # --------------------- memory-safe math (chunked over B) ---------------------

    @torch.no_grad()
    def _field_from_particles(self) -> torch.Tensor:
        E, P, B = self.E, self.P_max, self.mdl.B
        scale = (self.mdl.N ** (1.0 - self.mdl.gamma)) / float(self.mdl.B)
        f = torch.zeros(E, P, 1, device=self.device, dtype=self.dtype)
        for b0 in range(0, B, self._Bck):
            b1 = min(B, b0 + self._Bck)
            Wc = self.W[:, b0:b1, :]
            ac = self.a[:, b0:b1, :]                  # (E, bc, 1)
            z = torch.einsum('epd,ebd->epb', self.X, Wc)
            Phi = activation(z, self.mdl.act)         # (E, P, bc)
            # FIX: no digits in einsum subscripts
            f.add_(scale * torch.einsum('epb,ebc->epc', Phi, ac))  # -> (E, P, 1)
            del Wc, ac, z, Phi
        return f

    @torch.no_grad()
    def _grads_cavity(self, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        E, P, B, d = self.E, self.P_max, self.mdl.B, self.mdl.d
        grad_w = torch.zeros(E, B, d, device=self.device, dtype=self.dtype)
        grad_a = torch.zeros(E, B, 1, device=self.device, dtype=self.dtype)
        m = self.mask.unsqueeze(2)
        N_gamma = self.mdl.N ** self.mdl.gamma
        kap2 = (self.kappa**2).view(-1,1,1)
        P_eff = self.P_counts.view(-1,1,1).to(self.dtype)

        for b0 in range(0, B, self._Bck):
            b1 = min(B, b0 + self._Bck)
            Wc = self.W[:, b0:b1, :]
            ac = self.a[:, b0:b1, :]
            z = torch.einsum('epd,ebd->epb', self.X, Wc)
            Phi = activation(z, self.mdl.act)
            dPhi = act_prime(z, self.mdl.act)

            C1 = (Phi * r * m).sum(dim=1)           # (E, bc)
            C2 = (Phi * Phi * m).sum(dim=1)         # (E, bc)

            term1 = (1.0 / (self.mdl.sigma_a**2)) * ac[:,:,0]
            term2 = - C1 / (kap2.squeeze(-1) * P_eff.squeeze(-1) * N_gamma)
            term3 = (C2 / (kap2.squeeze(-1) * P_eff.squeeze(-1) * (N_gamma**2))) * ac[:,:,0]
            ga_chunk = (term1 + term2 + term3).unsqueeze(2)

            a_over_Ng = ac / N_gamma
            term = (r - Phi * a_over_Ng.transpose(1,2))
            M = term * dPhi * a_over_Ng.transpose(1,2)
            M = M * m
            G = - torch.einsum('epb,epd->ebd', M, self.X) / (kap2 * P_eff)
            gw_chunk = G + Wc * self.rho.unsqueeze(1)

            if self.algo.grad_clip_norm is not None:
                gw2 = (gw_chunk * gw_chunk).sum(dim=2, keepdim=True)
                ga2 = (ga_chunk * ga_chunk).sum(dim=2, keepdim=True)
                gn = torch.sqrt(gw2 + ga2) + 1e-12
                scale = torch.clamp(self.algo.grad_clip_norm / gn, max=1.0)
                gw_chunk = gw_chunk * scale
                ga_chunk = ga_chunk * scale

            grad_w[:, b0:b1, :] = gw_chunk
            grad_a[:, b0:b1, :] = ga_chunk

            del Wc, ac, z, Phi, dPhi, C1, C2, ga_chunk, gw_chunk, M, G, a_over_Ng, term

        return grad_w, grad_a

    @torch.no_grad()
    def _energy_per_particle_from_Wa(self, r: torch.Tensor, W: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        E, P, B, d = self.E, self.P_max, self.mdl.B, self.mdl.d
        prior_w_b = 0.5 * (self.rho.unsqueeze(1) * (W * W)).sum(dim=2)          # (E,B)
        prior_a_b = 0.5 * (1.0 / (self.mdl.sigma_a**2)) * (a[:,:,0]**2)         # (E,B)
        m = self.mask.unsqueeze(2)
        N_gamma = self.mdl.N ** self.mdl.gamma
        P_eff = self.P_counts.view(-1,1).to(self.dtype)
        kap2 = (self.kappa**2).view(-1,1)

        C1_full = torch.zeros(E, B, device=self.device, dtype=self.dtype)
        C2_full = torch.zeros(E, B, device=self.device, dtype=self.dtype)

        for b0 in range(0, B, self._Bck):
            b1 = min(B, b0 + self._Bck)
            Wc = W[:, b0:b1, :]
            z = torch.einsum('epd,ebd->epb', self.X, Wc)
            Phi = activation(z, self.mdl.act)
            C1 = (Phi * r * m).sum(dim=1)           # (E, bc)
            C2 = (Phi * Phi * m).sum(dim=1)         # (E, bc)
            C1_full[:, b0:b1] = C1
            C2_full[:, b0:b1] = C2
            del Wc, z, Phi, C1, C2

        a_flat = a[:,:,0]
        data_b = ( - (a_flat / N_gamma) * C1_full + 0.5 * (a_flat*a_flat / (N_gamma**2)) * C2_full ) / (kap2 * P_eff)
        return (prior_w_b + prior_a_b + data_b).to(self.dtype)

    # -------------------------- ARD updater --------------------------

    @torch.no_grad()
    def _update_rho_ard(self):
        if not self.ard.use_ard: return
        B = self.mdl.B
        alpha_post = self.ard.alpha0 + 0.5 * B
        ss = 0.5 * (self.W * self.W).sum(dim=1)                      # (E,d)
        beta0 = torch.tensor(self.beta0, device=self.device, dtype=self.dtype)
        beta_post = beta0 + ss
        eps = torch.tensor(1e-24, device=self.device, dtype=self.dtype)
        rho_hat = alpha_post / torch.clamp(beta_post, min=eps)       # (E,d)
        rho_hat = torch.clamp(rho_hat, min=self.ard.rho_min, max=self.ard.rho_max)
        self.rho.mul_(1.0 - self.ard.ema).add_(rho_hat, alpha=self.ard.ema)

    # -------------------------- samplers --------------------------

    @torch.no_grad()
    def _mala_inner(self, r: torch.Tensor, eta: float, active_mask: torch.Tensor) -> torch.Tensor:
        gw, ga = self._grads_cavity(r)

        xi_w = torch.empty_like(self.W)
        xi_a = torch.empty_like(self.a)
        for e in range(self.E):
            g = torch.Generator(device=self.device).manual_seed(int(self._noise_seeds[e]) + self._rng_step)
            xi_w[e] = torch.randn((self.mdl.B, self.mdl.d), generator=g, device=self.device, dtype=self.dtype)
            xi_a[e] = torch.randn((self.mdl.B, 1), generator=g, device=self.device, dtype=self.dtype)
        self._rng_step += 1

        s = active_mask.view(-1,1,1).to(self.dtype)
        Wp = (self.W - eta * gw * s + math.sqrt(2.0*eta) * xi_w * s)
        ap = (self.a - eta * ga * s + math.sqrt(2.0*eta) * xi_a * s)

        prop_finite = torch.isfinite(Wp).all(dim=2) & torch.isfinite(ap).all(dim=2)

        E_curr = self._energy_per_particle_from_Wa(r, self.W, self.a)
        W_saved, a_saved = self.W, self.a
        self.W, self.a = Wp, ap
        gw_p, ga_p = self._grads_cavity(r)
        E_prop = self._energy_per_particle_from_Wa(r, self.W, self.a)
        self.W, self.a = W_saved, a_saved

        mw  = self.W - eta * gw * s
        ma  = self.a - eta * ga * s
        mpw = Wp     - eta * gw_p * s
        mpa = ap     - eta * ga_p * s

        def sqsum_last(A): return (A*A).sum(dim=-1)
        log_q_prop_given_curr = - (sqsum_last(Wp - mw) + sqsum_last(ap - ma)) / (4.0*eta)
        log_q_curr_given_prop = - (sqsum_last(self.W - mpw) + sqsum_last(self.a - mpa)) / (4.0*eta)

        log_acc = (-E_prop + E_curr) + (log_q_curr_given_prop - log_q_prop_given_curr)
        ok = torch.isfinite(log_acc) & prop_finite & active_mask.view(-1,1)
        log_acc = torch.where(ok, log_acc, torch.full_like(log_acc, -float("inf")))

        u = torch.rand_like(log_acc)
        accept = (torch.log(u) < log_acc) & active_mask.view(-1,1)

        if accept.any():
            self.W[accept] = Wp[accept]
            self.a[accept] = ap[accept]

        n_bad = torch.zeros(self.E, dtype=torch.long, device=self.device)
        if self.algo.kill_nan_particles:
            bad_now = ( ~torch.isfinite(self.W).all(dim=2) | ~torch.isfinite(self.a).all(dim=2) ) & active_mask.view(-1,1)
            if bad_now.any():
                n_bad = bad_now.sum(dim=1)
                self._reinit_particles(bad_now)

        self._clamp_params()
        self._last_bad = n_bad
        self._last_bad_prop = ((~prop_finite) & active_mask.view(-1,1)).sum(dim=1)
        return accept.float().mean(dim=1)

    @torch.no_grad()
    def _sgld_inner(self, r: torch.Tensor, eta: float, active_mask: torch.Tensor):
        gw, ga = self._grads_cavity(r)
        s = active_mask.view(-1,1,1).to(self.dtype)
        self.W.add_(gw * s, alpha=-eta)
        self.a.add_(ga * s, alpha=-eta)
        xi_w = torch.randn_like(self.W, dtype=self.dtype) * s
        xi_a = torch.randn_like(self.a, dtype=self.dtype) * s
        self.W.add_(xi_w, alpha=math.sqrt(2.0*eta))
        self.a.add_(xi_a, alpha=math.sqrt(2.0*eta))
        self._clamp_params()

    # -------------------------- heldout eval (chunked P & B) --------------------------

    @torch.no_grad()
    def _eval_heldout(self, d: int, P_eval: int, P_chunk: int) -> Dict[str, List[float]]:
        device = self.device
        M = len(self.sets); E = self.E; B = self.mdl.B; Bck = self._Bck_eval

        sum_f2 = torch.zeros(E, device=device, dtype=self.dtype)
        if M > 0:
            sum_Ct_f = torch.zeros(E, M, device=device, dtype=self.dtype)
            sum_G = torch.zeros(M, M, device=device, dtype=self.dtype)
        scale = (self.mdl.N ** (1.0 - self.mdl.gamma)) / float(self.mdl.B)
        a = self.a.detach()

        g = torch.Generator(device=device).manual_seed(1234567)
        for start in range(0, P_eval, P_chunk):
            n = min(P_chunk, P_eval-start)
            Xc = (torch.randint(0,2,(n,d),generator=g,device=device,dtype=torch.int8).to(self.dtype) * 2.0 - 1.0)

            f_chunk = torch.zeros(E, n, 1, device=device, dtype=self.dtype)
            for b0 in range(0, B, Bck):
                b1 = min(B, b0 + Bck)
                Wc = self.W[:, b0:b1, :]
                ac = a[:, b0:b1, :]
                z = torch.einsum('nd,ebd->enb', Xc, Wc)
                Phi = activation(z, self.mdl.act)
                # FIX: no digits in einsum subscripts
                f_chunk.add_(scale * torch.einsum('enb,ebc->enc', Phi, ac))  # -> (E, n, 1)
                del Wc, ac, z, Phi

            f = f_chunk[:,:,0]
            sum_f2 += (f*f).sum(dim=1)

            if M > 0:
                Ccols = [parity_character(Xc, S) for S in self.sets]
                C = torch.stack(Ccols, dim=1)
                sum_Ct_f += f @ C    # (E, n) @ (n, M) -> (E, M)
                sum_G += C.t().matmul(C)
                del C

            del Xc, f_chunk, f

        invP = 1.0/float(P_eval)
        f2_bar = sum_f2 * invP
        out = dict(
            half_mse_empirical=(0.5*f2_bar).tolist(),
            half_mse_total_ms=(0.5*f2_bar).tolist(),
            half_mse_modes=[0.0]*E,
            half_noise=(0.5*f2_bar).tolist(),
            m_S=[[] for _ in range(E)]
        )
        if M==0:
            return out

        v = sum_Ct_f * invP
        G = sum_G * invP
        ones = torch.ones(M, device=device, dtype=self.dtype)
        ones_G_ones = float(ones.view(1,-1).matmul(G).matmul(ones.view(-1,1)).item())

        m_S = v
        half_mse_modes = 0.5*((1.0 - m_S)**2).sum(dim=1)
        ones_v = torch.einsum('m,em->e', ones, v)
        mTm = (m_S*m_S).sum(dim=1)
        mTGm = torch.einsum('em,mn,en->e', m_S, G, m_S)
        noise = f2_bar - 2.0*mTm + mTGm

        out.update(
            m_S=[m_S[i].detach().cpu().tolist() for i in range(self.E)],
            half_mse_modes=half_mse_modes.detach().cpu().tolist(),
            half_noise=(0.5*noise).detach().cpu().tolist(),
            half_mse_total_ms=(half_mse_modes + 0.5*noise).detach().cpu().tolist(),
            half_mse_empirical=(0.5*(f2_bar - 2.0*ones_v + ones_G_ones)).detach().cpu().tolist()
        )
        return out

    # -------------------------- run loop --------------------------

    @torch.no_grad()
    def _save_one(self, out_dir: str, tag: str, e: int, hist: Dict, meta: Dict,
                  early_stopped: bool, stop_iter: Optional[int], stop_mse: Optional[float]):
        P_e = int(meta["P"]); kap_e = float(meta["kappa"]); gid_e = int(meta["gid"])
        seeds_meta = meta["seeds"]
        summary = {
            "train_mse_last": hist["train_mse"][-1] if hist["train_mse"] else None,
            "accept_last": hist["accept"][-1] if hist["accept"] else None,
            "P_eval": self.algo.P_eval,
            "early_stopped": bool(early_stopped),
            "stop_iter": int(stop_iter) if stop_iter is not None else None,
            "stop_train_mse": float(stop_mse) if stop_mse is not None else None,
        }
        out = {
            "summary": summary,
            "traj": hist,
            "config": {
                "model": self.mdl.__dict__, "algo": self.algo.__dict__,
                "kappa": kap_e, "P_train": P_e, "gid": gid_e,
                "seeds": seeds_meta,
                "ard": {
                    "alpha0": self.ard.alpha0, "beta0": self.beta0,
                    "ema": self.ard.ema, "update_every": self.ard.update_every,
                    "rho_min": self.ard.rho_min, "rho_max": self.ard.rho_max,
                    "use_ard": self.ard.use_ard
                }
            }
        }
        tag_final = tag or time.strftime("%Y%m%d_%H%M%S")
        fname = (f"rs_cavity_aw_ard_{tag_final}"
                 f"_gid{gid_e}_Ptr{P_e}_Peval{self.algo.P_eval}"
                 f"_kap{kap_e:.3e}_N{self.mdl.N}_B{self.mdl.B}_g{self.mdl.gamma}.json")
        path = os.path.join(out_dir, fname)
        with open(path, "w") as f: json.dump(out, f, indent=2)
        print(f"[saved] {path}", flush=True)
        return path

    @torch.no_grad()
    def run(self, out_dir: str, tag: str, meta_grid_local: List[Dict]):
        os.makedirs(out_dir, exist_ok=True)
        E, P = self.E, self.P_max
        f_mean = torch.zeros(E, P, 1, device=self.device, dtype=self.dtype)

        hists = []
        for _ in range(E):
            hists.append({
                "iter": [], "train_mse": [], "accept": [],
                "half_mse_modes": [], "half_noise": [],
                "half_mse_total_ms": [], "half_mse_empirical": [], "m_S": [],
                "rho_min": [], "rho_max": [], "elapsed_s": [],
                "bad_prop": [], "bad_reset": [], "dtype": str(self.dtype)
            })

        done = torch.zeros(E, dtype=torch.bool, device=self.device)
        below_ctr = torch.zeros(E, dtype=torch.long, device=self.device)
        stop_iter = [-1]*E; stop_mse  = [None]*E; saved_path = [None]*E

        t0 = time.time()
        for it in range(1, self.algo.outer_steps+1):
            active = ~done
            r = (self.y - f_mean)

            if self.algo.use_mala:
                acc_step_sum = torch.zeros(E, device=self.device, dtype=self.dtype)
                bad_prop_sum = torch.zeros(E, device=self.device, dtype=torch.long)
                bad_reset_sum = torch.zeros(E, device=self.device, dtype=torch.long)
                for _ in range(self.algo.inner_mala_steps):
                    a_rate = self._mala_inner(r, self.algo.step_size, active_mask=active)
                    acc_step_sum += a_rate
                    if self.algo.log_bad_counts:
                        bad_prop_sum += getattr(self, "_last_bad_prop", torch.zeros(E, dtype=torch.long, device=self.device))
                        bad_reset_sum += getattr(self, "_last_bad", torch.zeros(E, dtype=torch.long, device=self.device))
                acc = (acc_step_sum / max(1, self.algo.inner_mala_steps)).detach().cpu().tolist()
            else:
                for _ in range(self.algo.inner_mala_steps):
                    self._sgld_inner(r, self.algo.step_size, active_mask=active)
                acc = [0.0]*E
                bad_prop_sum = torch.zeros(E, device=self.device, dtype=torch.long)
                bad_reset_sum = torch.zeros(E, device=self.device, dtype=torch.long)

            if self.ard.use_ard and (it % self.ard.update_every == 0):
                self._update_rho_ard()

            f_new = self._field_from_particles()
            f_mean = (1.0 - self.algo.field_blend) * f_mean + self.algo.field_blend * f_new if self.algo.cg_like_update else f_new

            if self.algo.early_stop_enabled:
                m = self.mask.unsqueeze(2)
                resid_iter = (self.y - f_mean) * m
                train_mse_vec = (resid_iter*resid_iter).sum(dim=(1,2)) / (self.P_counts.to(self.dtype))
                below = train_mse_vec < self.algo.early_stop_threshold
                inc_mask = active & below; reset_mask = active & (~below)
                below_ctr[inc_mask] += 1; below_ctr[reset_mask] = 0
                newly_done = active & (below_ctr >= self.algo.early_stop_patience)
                if newly_done.any():
                    for e in torch.nonzero(newly_done, as_tuple=False).view(-1).tolist():
                        done[e] = True
                        stop_iter[e] = it
                        stop_mse[e] = float(train_mse_vec[e].item())
                        saved_path[e] = self._save_one(out_dir, tag, e, hists[e], meta_grid_local[e], True, stop_iter[e], stop_mse[e])
                        print(json.dumps({
                            "device": str(self.device), "event": "early_stop", "iter": it,
                            "gid": meta_grid_local[e]["gid"], "P": meta_grid_local[e]["P"],
                            "kappa": meta_grid_local[e]["kappa"], "train_mse_at_stop": stop_mse[e],
                            "patience": self.algo.early_stop_patience, "threshold": self.algo.early_stop_threshold
                        }), flush=True)

            if it % self.algo.log_every == 0:
                m = self.mask.unsqueeze(2)
                resid = (self.y - f_mean) * m
                train_mse = (resid*resid).sum(dim=(1,2)) / (self.P_counts.to(self.dtype))
                ev = self._eval_heldout(self.mdl.d, self.algo.P_eval, self.algo.batch_eval)
                rhomin = self.rho.min(dim=1).values.detach().cpu().tolist()
                rhomax = self.rho.max(dim=1).values.detach().cpu().tolist()
                now_s = round(time.time()-t0,2)

                for e in range(E):
                    hists[e]["iter"].append(it)
                    hists[e]["train_mse"].append(float(train_mse[e].item()))
                    hists[e]["accept"].append(float(acc[e]) if not done[e] else hists[e]["accept"][-1] if hists[e]["accept"] else 0.0)
                    hists[e]["half_mse_modes"].append(float(ev["half_mse_modes"][e]))
                    hists[e]["half_noise"].append(float(ev["half_noise"][e]))
                    hists[e]["half_mse_total_ms"].append(float(ev["half_mse_total_ms"][e]))
                    hists[e]["half_mse_empirical"].append(float(ev["half_mse_empirical"][e]))
                    hists[e]["m_S"].append(ev["m_S"][e])
                    hists[e]["rho_min"].append(rhomin[e]); hists[e]["rho_max"].append(rhomax[e])
                    hists[e]["elapsed_s"].append(now_s)
                    hists[e]["bad_prop"].append(int(bad_prop_sum[e].item()))
                    hists[e]["bad_reset"].append(int(bad_reset_sum[e].item()))

                printable = []
                for e in range(E):
                    meta = meta_grid_local[e]
                    printable.append({
                        "iter": it, "gid": meta["gid"], "P": meta["P"], "kappa": meta["kappa"],
                        "train_mse": hists[e]["train_mse"][-1], "accept": hists[e]["accept"][-1],
                        "rho_min": rhomin[e], "rho_max": rhomax[e],
                        "bad_prop": int(bad_prop_sum[e].item()), "bad_reset": int(bad_reset_sum[e].item()),
                        "elapsed_s": now_s, "dtype": "float64" if self.algo.use_float64 else "float32",
                        "done": bool(done[e].item()), "below_ctr": int(below_ctr[e].item())
                    })
                print(json.dumps({"device": str(self.device), "log_step": it, "experiments": printable}), flush=True)

            if done.all():
                print(json.dumps({"device": str(self.device), "event": "all_done", "iter": it}), flush=True)
                break

        for e in range(E):
            if saved_path[e] is None:
                _ = self._save_one(out_dir, tag, e, hists[e], meta_grid_local[e],
                                   bool(done[e].item()),
                                   stop_iter[e] if done[e].item() else None,
                                   stop_mse[e] if done[e].item() else None)

# ============================ worker & launcher ============================

def worker_process(dev_id: int,
                   exps_local: List[Dict],
                   mdl_dict: Dict,
                   algo_dict: Dict,
                   ard_dict: Dict,
                   out_dir: str,
                   teacher_sets_spec: str,
                   dtype_float64: bool):
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.set_device(dev_id)
    device = torch.device(f"cuda:{dev_id}" if torch.cuda.is_available() else "cpu")

    d = mdl_dict["d"]
    sets_idx = parse_sets(teacher_sets_spec)
    sets = [torch.tensor(s, device=device, dtype=torch.long) for s in sets_idx]

    use_float64 = bool(dtype_float64)
    dtype = torch.float64 if use_float64 else torch.float32

    E_loc = len(exps_local)
    if E_loc == 0:
        return

    P_max = max(e["P"] for e in exps_local)
    X_all = torch.zeros(E_loc, P_max, d, device=device, dtype=dtype)
    y_all = torch.zeros(E_loc, P_max, 1, device=device, dtype=dtype)
    mask  = torch.zeros(E_loc, P_max, device=device, dtype=torch.bool)
    P_counts = torch.zeros(E_loc, device=device, dtype=torch.long)
    kappas = torch.zeros(E_loc, device=device, dtype=dtype)
    init_seeds = []; noise_seeds = []

    for idx, meta in enumerate(exps_local):
        P_e = meta["P"]; kap = meta["kappa"]; ds_seed = meta["seeds"]["dataset"]
        init_seeds.append(meta["seeds"]["init"]); noise_seeds.append(meta["seeds"]["noise"])
        Xe, ye = generate_parity(P_e, d, sets, device, dtype, seed=ds_seed)
        X_all[idx, :P_e, :] = Xe; y_all[idx, :P_e, :] = ye
        mask[idx, :P_e] = True; P_counts[idx] = P_e; kappas[idx] = kap

    mdl = Model(**mdl_dict); algo = Algo(**algo_dict); ard  = ARD(**ard_dict)

    solver = RSCavityExplicitBatch(
        mdl, algo, device,
        X_all, y_all, mask, P_counts, kappas,
        teacher_sets=sets, ard=ard,
        init_seeds=init_seeds, noise_seeds=noise_seeds
    )
    tag = f"GPU{dev_id}_E{E_loc}"
    solver.run(out_dir=out_dir, tag=tag, meta_grid_local=exps_local)

def run_grid_multi_gpu(
    P_train_list: List[int],
    kappa_list: List[float],
    num_exp: int,
    base_seed: int,
    out_dir: str,
    mdl: Model,
    algo: Algo,
    ard: ARD,
    teacher_sets_spec: str = "{0,1,2,3}",
    shard_strategy: str = "balance_P",
):
    os.makedirs(out_dir, exist_ok=True)
    exps = build_experiment_grid(P_train_list, kappa_list, num_exp, base_seed)
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    shards = shard_experiments(exps, num_devices, strategy=shard_strategy)

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

# ============================ main ============================

if __name__ == "__main__":
    set_seed(42)

    teacher_sets_spec = "{0,1,2,3}"
    d = 35



    P_train_list =[10, 100, 500, 750, 1000, 2133, 3666, 5000, 7500,20000,10000]
    kappa_list   =[7.5e-3]  #[7.5e-3, 1e-2, 1e-1, 1e-3, 5e-3, 7.5e-2, 2.5e-2, 5e-2, 5e-4]
    num_exp = 3
    base_seed = 123456

    use_float64 = False
    mdl = Model(d=d, B=128*8, N=512, gamma=0.5, sigma_a=1.0, sigma_w=1.0, act="relu")
    algo = Algo(
        outer_steps=25_000, inner_mala_steps=200, step_size=4e-7, use_mala=True,
        log_every=10, cg_like_update=False, field_blend=0.8,
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
        early_stop_threshold=0.04,
        early_stop_patience=2000,
    )
    ard = ARD(use_ard=True, alpha0=1e-2, ema=0.25, update_every=1,
              rho_min=1e-12, rho_max=1e12, beta0=None)

    out_dir = "/home/goring/mean_field_langevin/MCMC_sparse/results/d35_k4_grid_75e-3_3008_a01e-2"

    run_grid_multi_gpu(
        P_train_list=P_train_list,
        kappa_list=kappa_list,
        num_exp=num_exp,
        base_seed=base_seed,
        out_dir=out_dir,
        mdl=mdl,
        algo=algo,
        ard=ard,
        teacher_sets_spec=teacher_sets_spec,
        shard_strategy="balance_P",
    )
