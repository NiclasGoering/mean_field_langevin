# schur_quenched_finiteP.py
import os, json, time, math, random
from dataclasses import dataclass
from itertools import combinations
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch._dynamo as dynamo

# ----------------------------- Global perf knobs (Hopper) -----------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ----------------------------- Utils -----------------------------
def check_gpu():
    if not torch.cuda.is_available():
        print("CUDA not available. Running on CPU.")
        return []
    n = torch.cuda.device_count()
    info = []
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        info.append({"index": i, "name": name, "capability": cap, "mem_GB": round(total_mem, 2)})
    print("GPUs:", info)
    return list(range(n))

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def enumerate_combos(d: int, c: int) -> torch.Tensor:
    rows = list(combinations(range(d), c))
    if len(rows) == 0:
        return torch.empty((0, c), dtype=torch.long)
    return torch.tensor(rows, dtype=torch.long)

def make_parity_indices(d: int, k: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    S = torch.randperm(d, generator=g)[:k]
    return S.sort().values

def activation(z: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "relu": return torch.relu(z)
    if kind == "tanh": return torch.tanh(z)
    raise ValueError(f"Unknown activation: {kind}")

# ----------------------------- Build T/H mode lists -----------------------------
def build_modes_T_H(
    d: int,
    teacher_S: torch.Tensor,
    H_card_small: int,
    H_per_card_high: int,
    seed: int = 1,
    T_extra: List[Tuple[int]] = None
) -> Tuple[List[Tuple[int]], List[Tuple[int]]]:
    """
    Return (T_list, H_list) of tuples of indices (sorted).
    Teacher always first entry of T (tuple(teacher_S)).
    H: all subsets up to H_card_small (excluding teacher),
       plus H_per_card_high random from higher cardinalities.
    """
    rng = np.random.default_rng(seed)
    teacher_tuple = tuple(teacher_S.tolist())

    T_list: List[Tuple[int]] = [teacher_tuple]
    if T_extra:
        for t in T_extra:
            t = tuple(sorted(t))
            if t != teacher_tuple and t not in T_list:
                T_list.append(t)

    H_list: List[Tuple[int]] = []
    # small cards
    for c in range(1, min(H_card_small, d) + 1):
        all_c = enumerate_combos(d, c)
        keep = [tuple(row.tolist()) for row in all_c if tuple(row.tolist()) != teacher_tuple]
        H_list.extend(keep)

    # higher cards (sample)
    for c in range(H_card_small + 1, d + 1):
        all_c = enumerate_combos(d, c)
        if all_c.numel() == 0:
            continue
        keep = [tuple(row.tolist()) for row in all_c if tuple(row.tolist()) != teacher_tuple]
        if len(keep) == 0:
            continue
        k_take = min(H_per_card_high, len(keep))
        idx = rng.choice(len(keep), size=k_take, replace=False)
        H_list.extend([keep[i] for i in idx])

    return T_list, H_list

def modes_to_padded_idx(modes: List[Tuple[int]]) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Convert list of tuples to (idx_padded, mask, k_max).
    """
    if len(modes) == 0:
        return torch.zeros((0,1), dtype=torch.long), torch.zeros((0,1), dtype=torch.bool), 1
    k_max = max(len(t) for t in modes)
    M = len(modes)
    idx = torch.zeros((M, k_max), dtype=torch.long)
    mask = torch.zeros((M, k_max), dtype=torch.bool)
    for i,t in enumerate(modes):
        L = len(t)
        if L>0:
            idx[i,:L] = torch.tensor(t, dtype=torch.long)
            mask[i,:L] = True
    return idx, mask, k_max

# ----------------------------- Parity signs for selected modes -----------------------------
def chi_for_modes(
    X: torch.Tensor,           # (P,d) float ±1
    idx_padded: torch.Tensor,  # (M,k_max)
    mask: torch.Tensor         # (M,k_max)
) -> torch.Tensor:
    """
    Return C in int8: (P,M) with entries ±1 for parity χ_A(x_i).
    """
    device = X.device
    P, d = X.shape
    M, kmax = idx_padded.shape
    if M == 0:
        return torch.empty((P,0), dtype=torch.int8, device=device)
    # gather the relevant bits: (P, M, k_max)
    idx = idx_padded.to(device)         # (M,kmax)
    msk = mask.to(device)               # (M,kmax)
    # Expand X to (P, M, d)
    Xexp = X.unsqueeze(1).expand(P, M, d)
    idxexp = idx.unsqueeze(0).expand(P, M, kmax)
    feats = torch.gather(Xexp, 2, idxexp)          # (P,M,kmax)
    feats = torch.where(msk.unsqueeze(0), feats, torch.ones_like(feats))
    out = feats.prod(dim=2)                         # (P,M) float ±1
    return out.to(torch.int8)

# ----------------------------- Data classes -----------------------------
@dataclass
class ModelParams:
    d: int = 25
    N: int = 1024
    k: int = 4
    sigma_a: float = 1.0
    sigma_w: float = 1.0
    gamma: float = 1.0
    act: str = "relu"

@dataclass
class MCMCParams:
    B: int = 32768        # number of chains
    steps: int = 300
    step_size: float = 2e-3
    step_decay: float = 0.999
    grad_clip: float = 1e15
    clamp_w: float = 10.0
    langevin_sqrt2: bool = True

@dataclass
class SolveParams:
    ridge_lambda: float = 1e-3
    pcg_tol_outer: float = 1e-5
    pcg_max_outer: int = 400
    pcg_tol_inner: float = 5e-4
    pcg_max_inner: int = 200
    print_every: int = 10
    saem_a0: float = 1.0
    saem_t0: float = 150.0
    saem_damping: float = 3.0

# ------

@dynamo.disable
def sgld_sample_w(
    w: torch.Tensor,      # (B,d)
    X: torch.Tensor,      # (P,d) ±1 float
    C_all: torch.Tensor,  # (P,M_all) ±1 float/int8 for all tracked modes (T ∪ H)
    y_vec: torch.Tensor,  # (P,) ±1 float  (teacher labels per-sample)
    m_full: torch.Tensor, # (M_all,) current mean-field coefficients for all modes
    kappa: float,
    mdl: ModelParams,
    mcmc: MCMCParams,
    *,
    sgld_scale: str = "P",             # "P" -> uses P*Σ in α, "1" -> uses Σ
    autocast_enabled: Optional[bool] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One full SGLD run. Returns:
      - new w (B,d)
      - phi (P,B) activations at the final iterate (to build alpha)
      - D_oracle (B,)         -- κ² * α_variant(w) = Acoef*κ² + scale*Σ, used by oracle α_i = E[phi_i^2 / D_oracle_b]
      - D_internal (B,)       -- α_variant(w) = Acoef + scale*Σ/κ², used inside SGLD gradient/logdet
    """
    device = w.device
    P = X.shape[0]

    # Prior scales
    gw2   = mdl.sigma_w / mdl.d                     # variance(w_j) = σ_w / d
    Acoef = (mdl.N ** mdl.gamma) / mdl.sigma_a      # α(w) base term

    step = mcmc.step_size

    # Per-sample mean field f_mean = (C_all @ m_full)
    f_mean = (C_all.float() @ m_full.view(-1, 1)).squeeze(1)   # (P,)

    s = (sgld_scale or "P").strip().lower()
    if s not in {"p","1"}:
        raise ValueError(f"sgld_scale must be 'P' or '1', got {sgld_scale!r}")
    scale = float(P) if s == "p" else 1.0

    if autocast_enabled is None:
        autocast_enabled = w.is_cuda

    for _ in range(mcmc.steps):
        w = w.detach().requires_grad_(True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            z = torch.matmul(X, w.t().contiguous())           # (P,B)
            phi = activation(z, mdl.act)
        phi_f = phi.float()

        # JY = <phi, y>_P ;  Jm = <phi, f_mean>_P
        JY = torch.matmul(phi_f.t(), y_vec.float()) / float(P)     # (B,)
        Jm = torch.matmul(phi_f.t(), f_mean)         / float(P)     # (B,)

        Sigma = (phi_f * phi_f).mean(dim=0)                         # (B,)

        # Internal α_variant(w) and corresponding log-likelihood pieces
        D_internal = (Acoef + scale * Sigma / (kappa**2)).clamp_min(1e-9)   # (B,)
        data_quad  = (scale**2) * (0.5) * (JY - Jm).pow(2) / ((kappa**4) * D_internal)
        log_det    = 0.5 * torch.log(D_internal)

        # Gaussian prior appears with a minus in log p
        prior_term = 0.5 * (w*w).sum(dim=1) / gw2
        logp = -prior_term + data_quad - log_det

        grad = torch.autograd.grad(logp.sum(), w, retain_graph=False, create_graph=False)[0]
        grad = torch.where(torch.isfinite(grad), grad, torch.zeros_like(grad))
        if mcmc.grad_clip and mcmc.grad_clip > 0:
            gn = grad.norm(dim=1, keepdim=True).clamp_min(1e-12)
            grad = grad * (mcmc.grad_clip/gn).clamp(max=1.0)

        noise = torch.randn_like(w)
        if mcmc.langevin_sqrt2:
            w = w - step * grad + noise * math.sqrt(2.0*step)
        else:
            w = w - 0.5*step*grad + noise * math.sqrt(step)
        if mcmc.clamp_w:
            w = torch.clamp(w, -mcmc.clamp_w, mcmc.clamp_w)
        step *= mcmc.step_decay

    # Build denominators for the oracle from the final Sigma:
    #   D_oracle = κ² * α_variant(w) = Acoef*κ² + scale*Sigma
    #   D_internal (for diagnostics) = α_variant(w) = Acoef + scale*Sigma/κ²
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        z = torch.matmul(X, w.t().contiguous())
        phi = activation(z, mdl.act)
    phi_f = phi.float()
    Sigma = (phi_f * phi_f).mean(dim=0)

    D_oracle   = (Acoef * (kappa**2) + scale * Sigma).clamp_min(1e-9)
    D_internal = (Acoef + scale * Sigma / (kappa**2)).clamp_min(1e-9)

    return w.detach(), phi.detach(), D_oracle.detach(), D_internal.detach()


# ----------------------------- B/b oracles and matvecs -----------------------------
class BbOracle:
    """
    y = B v and b, with scale in {"P","P2"} (case-insensitive):
      "P" :  B = (1/P)   C^T diag(alpha) C,  b = (1/P)   C^T (alpha ⊙ sS)
      "P2":  B = (1/P^2) C^T diag(alpha) C,  b = (1/P^2) C^T (alpha ⊙ sS)
    alpha_i = mean_b ( u_i^2 / D_oracle_b ).
    """
    def __init__(self, C_pm: torch.Tensor, sS: torch.Tensor, device: torch.device,
                 scale: str = "P"):
        self.device = device
        self.C = C_pm.to(device)                 # (P,M) int8
        self.P, self.M = self.C.shape
        self.Cf = self.C.float()
        self.Ct = self.Cf.t().contiguous()       # (M,P)
        self.sS = sS.to(device).float()
        self.alpha = torch.ones(self.P, device=device)

        s = (scale or "P").strip().lower()
        if s not in {"p", "p2"}:
            raise ValueError(f"BbOracle: unknown scale '{scale}', expected 'P' or 'P2'.")
        self.scale = s
        self._update_diag_scalar()

    def _denom(self) -> float:
        return float(self.P) if self.scale == "p" else float(self.P**2)

    def _update_diag_scalar(self):
        self.sum_alpha = float(self.alpha.sum().item())
        self.diag_B_scalar = self.sum_alpha / self._denom()

    def update_alpha(self, phi: torch.Tensor, D_oracle: torch.Tensor):
        """
        D_oracle must be κ² * α_variant(w) = Acoef*κ² + scale*Σ (same variant as SGLD).
        alpha_i = E_b[ phi_i(b)^2 / D_oracle_b ].
        """
        phi_f = phi.float()
        B = phi_f.shape[1]
        invD = (1.0 / D_oracle.float()).view(1, B)
        a = (phi_f*phi_f) * invD
        self.alpha = a.mean(dim=1).contiguous()     # (P,)
        self._update_diag_scalar()

    def matvec_B(self, v_full: torch.Tensor) -> torch.Tensor:
        s = torch.matmul(self.Cf, v_full)                                 # (P,)
        y = torch.matmul(self.Ct, self.alpha * s) / self._denom()         # (M,)
        return y

    def vec_b(self) -> torch.Tensor:
        return torch.matmul(self.Ct, self.alpha * self.sS) / self._denom()

# ----------------------------- PCG utilities -----------------------------
def pcg(apply_A, rhs, diag_A, tol=1e-5, maxit=400):
    x = torch.zeros_like(rhs)
    r = rhs - apply_A(x)
    Minv = 1.0 / diag_A.clamp_min(1e-30)
    z = Minv * r
    p = z.clone()
    rz_old = (r*z).sum()
    rhsn = rhs.norm().clamp_min(1e-30)
    it = 0
    relres = float((r.norm()/rhsn).item())
    while it < maxit and relres > tol:
        Ap = apply_A(p)
        alpha = rz_old / (p*Ap).sum().clamp_min(1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        z = Minv * r
        rz_new = (r*z).sum()
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new
        relres = float((r.norm()/rhsn).item())
        it += 1
    return x, relres, it

# ----------------------------- Schur solver over H -----------------------------
class SchurSolver:
    """
    Outer CG on T using Schur complement;
    Inner CG on H to apply [(1+λ)I_H + N B_HH]^{-1}.
    Everything built from a single BbOracle (C, alpha).
    """
    def __init__(self, oracle: BbOracle, T_idx: torch.Tensor, H_idx: torch.Tensor,
                 N: int, ridge_lambda: float, tol_outer: float, max_outer: int,
                 tol_inner: float, max_inner: int):
        self.ora = oracle
        self.T_idx = T_idx
        self.H_idx = H_idx
        self.N = float(N)
        self.lam = float(ridge_lambda)
        self.tol_outer = tol_outer
        self.max_outer = max_outer
        self.tol_inner = tol_inner
        self.max_inner = max_inner

        self.M_T = T_idx.numel()
        self.M_H = H_idx.numel()
        self.device = oracle.C.device

        # diag(B) is constant across coords: alpha.sum()/denom
        self.diagB_scalar = oracle.diag_B_scalar
        self.diag_AH = (1.0 + self.lam) + self.N * self.diagB_scalar

        self.M_all = oracle.M

    def _pad_T(self, vT: torch.Tensor) -> torch.Tensor:
        v = torch.zeros(self.M_all, device=self.device, dtype=vT.dtype)
        v[self.T_idx] = vT
        return v

    def _pad_H(self, vH: torch.Tensor) -> torch.Tensor:
        v = torch.zeros(self.M_all, device=self.device, dtype=vH.dtype)
        v[self.H_idx] = vH
        return v

    def apply_B_TT(self, vT: torch.Tensor) -> torch.Tensor:
        y = self.ora.matvec_B(self._pad_T(vT))
        return y[self.T_idx]

    def apply_B_HT(self, vT: torch.Tensor) -> torch.Tensor:
        y = self.ora.matvec_B(self._pad_T(vT))
        return y[self.H_idx]

    def apply_B_TH(self, vH: torch.Tensor) -> torch.Tensor:
        y = self.ora.matvec_B(self._pad_H(vH))
        return y[self.T_idx]

    def apply_B_HH(self, vH: torch.Tensor) -> torch.Tensor:
        y = self.ora.matvec_B(self._pad_H(vH))
        return y[self.H_idx]

    def solve_AH(self, rhs_H: torch.Tensor) -> Tuple[torch.Tensor, float, int]:
        def apply_AH(vH):
            return (1.0 + self.lam) * vH + self.N * self.apply_B_HH(vH)
        diag = torch.full_like(rhs_H, self.diag_AH)
        z, relres, it = pcg(apply_AH, rhs_H, diag, tol=self.tol_inner, maxit=self.max_inner)
        return z, relres, it

    def build_rhs_T(self, b_full: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        b_T = b_full[self.T_idx]
        b_H = b_full[self.H_idx]
        z_H, relres_in, it_in = self.solve_AH(b_H)
        corr_T = self.apply_B_TH(z_H)
        rhs = self.N * b_T - (self.N**2) * corr_T
        stats = {"inner_rhs_relres": relres_in, "inner_rhs_iters": it_in}
        return rhs, stats

    def apply_Schur(self, vT: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        BTv = self.apply_B_TT(vT)
        r_H = self.apply_B_HT(vT)
        z_H, relres_in, it_in = self.solve_AH(r_H)
        corr_T = self.apply_B_TH(z_H)
        out = (1.0 + self.lam) * vT + self.N * BTv - (self.N**2) * corr_T
        return out, {"inner_apply_relres": relres_in, "inner_apply_iters": it_in}

    def solve_outer(self, rhs_T: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        def apply_AT(vT):
            out, info = self.apply_Schur(vT)
            self.last_inner_info = info
            return out
        diag = torch.full_like(rhs_T, (1.0 + self.lam) + self.N * self.diagB_scalar)
        xT, relres, it = pcg(apply_AT, rhs_T, diag, tol=self.tol_outer, maxit=self.max_outer)
        return xT, {"outer_relres": relres, "outer_iters": it, **getattr(self, "last_inner_info", {})}

    def reconstruct_mH(self, mT: torch.Tensor, b_full: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        b_H = b_full[self.H_idx]
        r_H = b_H - self.apply_B_HT(mT)
        rhs_H = self.N * r_H
        mH, relres, it = self.solve_AH(rhs_H)
        return mH, {"rec_relres": relres, "rec_iters": it}

# ----------------------------- PROBE MODE (randomized H per iteration) -----------------------------
def sample_random_modes(d: int, count: int, exclude: set, rng: np.random.Generator) -> List[Tuple[int]]:
    """
    Draw 'count' random parity subsets (any cardinality 1..d) excluding given tuples.
    Cardinality is sampled roughly geometric toward small k (p(k) ∝ 2^{-k}).
    """
    modes: List[Tuple[int]] = []
    weights = np.array([2.0**(-k) for k in range(1, d+1)], dtype=float)
    weights /= weights.sum()

    ks = rng.choice(np.arange(1, d+1), size=4*count, replace=True, p=weights)
    ptr = 0
    while len(modes) < count and ptr < len(ks):
        k = int(ks[ptr]); ptr += 1
        if k > d:
            continue
        S = tuple(sorted(rng.choice(d, size=k, replace=False).tolist()))
        if S not in exclude:
            modes.append(S)
            exclude.add(S)
    return modes

def build_modes_T_H_probes(
    d: int,
    teacher_S: torch.Tensor,
    H_card_small: int,
    R_probes_high: int,
    seed: int = 1234,
    T_extra: Optional[List[Tuple[int]]] = None
) -> Tuple[List[Tuple[int]], List[Tuple[int]]]:
    """
    T: [teacher] (+ optional extras)
    H: all subsets up to H_card_small (excluding T), PLUS R_probes_high random modes across higher k.
    """
    rng = np.random.default_rng(seed)
    teacher_tuple = tuple(teacher_S.tolist())
    T_list: List[Tuple[int]] = [teacher_tuple]
    if T_extra:
        for t in T_extra:
            t = tuple(sorted(t))
            if t != teacher_tuple and t not in T_list:
                T_list.append(t)

    # small-k exactly
    H_small: List[Tuple[int]] = []
    for c in range(1, min(H_card_small, d) + 1):
        all_c = enumerate_combos(d, c)
        keep = [tuple(row.tolist()) for row in all_c if tuple(row.tolist()) != teacher_tuple]
        H_small.extend(keep)

    exclude = set(T_list) | set(H_small)
    H_rand = sample_random_modes(d, R_probes_high, exclude, rng)
    H_list = H_small + H_rand
    return T_list, H_list

# ----------------------------- Main experiment (single run) -----------------------------
def run_schur_quenched(
    d: int = 25, P: int = 2000, k: int = 4, kappa: float = 1e-3,
    H_card_small: int = 2, H_per_card_high: int = 50, T_extra: List[Tuple[int]] = None,
    mdl: ModelParams = None, mcmc: MCMCParams = None, sol: SolveParams = None,
    steps: int = 300, save_dir: str = "./results_schur_quenched", run_tag: str = "",
    *,
    oracle_scale: str = "P",
    H_mode: str = "enumerate",
    R_probes_high: int = 4096,
    resample_probes_each_iter: bool = True,
    probes_seed: int = 1234,
    sgld_scale: str = "P"
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mdl = mdl or ModelParams(d=d, N=1024, k=k, sigma_a=1.0, sigma_w=1.0, gamma=1.0, act="relu")
    mcmc = mcmc or MCMCParams(B=32768, steps=400, step_size=2e-3, step_decay=0.999, grad_clip=1e15, clamp_w=10.0)
    sol = sol or SolveParams(ridge_lambda=1e-3, pcg_tol_outer=1e-5, pcg_max_outer=400,
                             pcg_tol_inner=5e-4, pcg_max_inner=200, print_every=10,
                             saem_a0=1.0, saem_t0=150.0, saem_damping=3.0)

    os.makedirs(save_dir, exist_ok=True)

    # Dataset (quenched)
    g = torch.Generator(device=device).manual_seed(0)
    X = (torch.randint(0,2,(P,d), generator=g, device=device, dtype=torch.int8).float()*2.0 - 1.0)
    teacher_S = make_parity_indices(d, k, seed=0)

    # ----- Build T/H once (rebuilt inside loop only if probes enabled) -----
    if H_mode == "probes":
        T_list, H_list = build_modes_T_H_probes(d, teacher_S, H_card_small, R_probes_high,
                                                seed=probes_seed, T_extra=T_extra)
    else:
        T_list, H_list = build_modes_T_H(d, teacher_S, H_card_small, H_per_card_high,
                                         seed=1, T_extra=T_extra)
    assert tuple(teacher_S.tolist()) == T_list[0]

    # Teacher labels sS from teacher mode
    T_idx_pad, T_mask, _ = modes_to_padded_idx(T_list)
    C_T = chi_for_modes(X, T_idx_pad, T_mask).float()      # (P, M_T)
    sS  = C_T[:, 0].contiguous().float()                   # (P,)

    # Build C_all and indices NOW (so they exist before the loop)
    all_modes = T_list + H_list
    all_idx_pad, all_mask, _ = modes_to_padded_idx(all_modes)
    C_all = chi_for_modes(X, all_idx_pad, all_mask)        # (P, M_all) int8
    T_idx = torch.arange(0, len(T_list), device=device, dtype=torch.long)
    H_idx = torch.arange(len(T_list), len(all_modes), device=device, dtype=torch.long)

    # Initialize parameters
    w = torch.randn(mcmc.B, d, device=device) * math.sqrt(mdl.sigma_w/mdl.d)
    m_T = torch.zeros(len(T_list), device=device, dtype=torch.float32)
    m_T[0] = 0.5
    m_H = torch.zeros(len(H_list), device=device, dtype=torch.float32)   # ensure defined

    print({"device": str(device), "P": P, "B": mcmc.B, "M_T": len(T_list), "M_H": len(H_list), "H_mode": H_mode})
    traj = {"mS": [], "m_noise_norm2": [], "m_noise_rms": [], "err01_clt": [], "mse": [], "time_s": [],
            "diagB": [], "sum_alpha": []}
    t0 = time.time()
    local_seed = probes_seed

    for it in range(1, steps+1):

        # --------- Optionally rebuild H (probes) BEFORE E-step ----------
        if H_mode == "probes" and (it == 1 or resample_probes_each_iter):
            local_seed = (local_seed * 1315423911 + 2654435761) & 0xFFFFFFFF
            T_list, H_list = build_modes_T_H_probes(d, teacher_S, H_card_small, R_probes_high,
                                                    seed=int(local_seed), T_extra=T_extra)
            all_modes = T_list + H_list
            all_idx_pad, all_mask, _ = modes_to_padded_idx(all_modes)
            C_all = chi_for_modes(X, all_idx_pad, all_mask)              # redefine (P, M_all)
            T_idx = torch.arange(0, len(T_list), device=device, dtype=torch.long)
            H_idx = torch.arange(len(T_list), len(all_modes), device=device, dtype=torch.long)
            m_H = torch.zeros(len(H_list), device=device, dtype=torch.float32)  # reset to match new H

        # --------- Build full mean-field coeffs and run SGLD (E-step) ----------
        m_full = torch.zeros(C_all.shape[1], device=device)
        m_full[T_idx] = m_T
        m_full[H_idx] = m_H

        w, phi, D_oracle, D_internal = sgld_sample_w(
            w, X,
            C_all=C_all.float(),   # SGLD wants float
            y_vec=sS,
            m_full=m_full,
            kappa=kappa,
            mdl=mdl,
            mcmc=mcmc,
            sgld_scale=sgld_scale
        )

        # --------- Oracle and Schur solve ----------
        oracle = BbOracle(C_all, sS, device=device, scale=oracle_scale)
        oracle.update_alpha(phi, D_oracle)  # IMPORTANT: use κ²·α_variant(w)
        b_full = oracle.vec_b()

        schur = SchurSolver(
            oracle, T_idx=T_idx, H_idx=H_idx, N=mdl.N, ridge_lambda=sol.ridge_lambda,
            tol_outer=sol.pcg_tol_outer, max_outer=sol.pcg_max_outer,
            tol_inner=sol.pcg_tol_inner, max_inner=sol.pcg_max_inner
        )

        rhs_T, rhs_stats = schur.build_rhs_T(b_full)
        mT_star, stats = schur.solve_outer(rhs_T)

        # SAEM smoothing on m_T
        a_t = sol.saem_a0 / (it + sol.saem_t0)
        m_T = (1 - sol.saem_damping*a_t) * m_T + sol.saem_damping*a_t * mT_star
        m_T[0] = m_T[0].clamp_(0.0, 1.0)

        # Reconstruct m_H (diagnostics + next-iter mean field)
        m_H, rec_stats = schur.reconstruct_mH(m_T, b_full)

        # ---- diagnostics ----
        noise_norm2 = float((m_H*m_H).sum().item())
        noise_rms = float((m_H*m_H).mean().sqrt().item())
        mS = float(m_T[0].item())
        mse = 0.5*(1.0 - mS)**2 + noise_norm2
        denom = max(noise_norm2, 1e-30)
        clt = float(torch.distributions.Normal(0,1).cdf(torch.tensor(-mS/math.sqrt(denom))).item())

        traj["mS"].append(mS); traj["m_noise_norm2"].append(noise_norm2); traj["m_noise_rms"].append(noise_rms)
        traj["mse"].append(mse); traj["err01_clt"].append(clt); traj["time_s"].append(time.time()-t0)
        traj["diagB"].append(oracle.diag_B_scalar); traj["sum_alpha"].append(oracle.sum_alpha)

        if it % sol.print_every == 1 or it == steps:
            msg = {
                "iter": it, "P": P, "kappa": kappa,
                "m_S": mS, "mse": mse, "err01_clt": clt,
                "noise_rms": noise_rms,
                "pcg_outer_relres": stats["outer_relres"], "pcg_outer_iters": stats["outer_iters"],
                "inner_apply_relres": stats.get("inner_apply_relres", None),
                "inner_apply_iters": stats.get("inner_apply_iters", None),
                "inner_rhs_relres": rhs_stats["inner_rhs_relres"], "inner_rhs_iters": rhs_stats["inner_rhs_iters"],
                "rec_relres": rec_stats["rec_relres"], "rec_iters": rec_stats["rec_iters"],
                "diagB_scalar": oracle.diag_B_scalar, "sum_alpha": oracle.sum_alpha,
                "M_T": len(T_list), "M_H": len(H_list),
                "H_mode": H_mode, "scale": oracle_scale, "sgld_scale": sgld_scale,
                "time_s": round(traj["time_s"][-1], 2)
            }
            print(json.dumps(msg))

    # Save snapshot
    out = {
        "summary": {
            "P": P, "kappa": kappa, "M_T": len(T_list), "M_H": len(H_list),
            "mS_last": traj["mS"][-1], "mse_last": traj["mse"][-1],
            "err01_clt_last": traj["err01_clt"][-1],
            "diagB_last": traj["diagB"][-1], "sum_alpha_last": traj["sum_alpha"][-1],
            "H_mode": H_mode, "scale": oracle_scale, "sgld_scale": sgld_scale
        },
        "traj": traj,
        "config": {
            "d": d, "k": k, "H_card_small": H_card_small,
            "H_per_card_high": H_per_card_high, "R_probes_high": R_probes_high,
            "T_extra": T_extra, "mdl": vars(mdl), "mcmc": vars(mcmc), "sol": vars(sol),
            "H_mode": H_mode, "oracle_scale": oracle_scale, "sgld_scale": sgld_scale,
            "resample_probes_each_iter": resample_probes_each_iter, "probes_seed": probes_seed
        }
    }
    tag = run_tag or time.strftime("%Y%m%d_%H%M%S")
    fname = f"schur_quenched_{tag}_kap{float(kappa):.3e}_P{P}_M{len(T_list)+len(H_list)}.json"
    path = os.path.join(save_dir, fname)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {path}")
    return out, path

# ----------------------------- Sweep runner (P, kappa iterable) -----------------------------
def run_sweep(
    P_list: List[int],
    kappa_list: List[float],
    *,
    d: int = 25,
    k: int = 4,
    H_card_small: int = 2,
    H_per_card_high: int = 50,
    T_extra: List[Tuple[int]] = None,
    mdl: ModelParams = None,
    mcmc: MCMCParams = None,
    sol: SolveParams = None,
    steps: int = 300,
    save_dir: str = "./results_schur_quenched",
    run_tag: str = "",
    oracle_scale: str = "P",
    H_mode: str = "enumerate",
    R_probes_high: int = 4096,
    resample_probes_each_iter: bool = True,
    probes_seed: int = 1234,
    sgld_scale: str = "P"
):
    os.makedirs(save_dir, exist_ok=True)
    index = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_tag": run_tag or time.strftime("%Y%m%d_%H%M%S"),
        "grid": {"P_list": P_list, "kappa_list": kappa_list},
        "runs": []
    }
    for kap in kappa_list:
        for P in P_list:
            res, path = run_schur_quenched(
                d=d, P=P, k=k, kappa=kap,
                H_card_small=H_card_small, H_per_card_high=H_per_card_high, T_extra=T_extra,
                mdl=mdl, mcmc=mcmc, sol=sol, steps=steps,
                save_dir=save_dir, run_tag=index["run_tag"],
                oracle_scale=oracle_scale, H_mode=H_mode,
                R_probes_high=R_probes_high,
                resample_probes_each_iter=resample_probes_each_iter,
                probes_seed=probes_seed,
                sgld_scale=sgld_scale
            )
            s = res["summary"]
            index["runs"].append({
                "P": P, "kappa": kap,
                "M_T": s["M_T"], "M_H": s["M_H"],
                "mS_last": s["mS_last"], "mse_last": s["mse_last"], "err01_clt_last": s["err01_clt_last"],
                "file": os.path.basename(path)
            })

    idx_path = os.path.join(save_dir, f"index_{index['run_tag']}.json")
    with open(idx_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"[index saved] {idx_path}")
    return index

# ----------------------------- Entrypoint -----------------------------
if __name__ == "__main__":
    set_seed(42)
    _ = check_gpu()

    # ======== EDIT THESE TO CHOOSE YOUR GRID ========
    P_list     = [2000]      # e.g. [100, 500, 1000, 2000]
    kappa_list = [1e-3]      # e.g. [1e-3, 1e-2]
    steps = 1200
    save_dir = "./results_schur_quenched"
    run_tag  = ""

    # H split:
    H_card_small     = 4              # exact small-k in both modes
    H_per_card_high  = 50            # only used when H_mode="enumerate"

    # PROBE MODE SETTINGS
    H_mode = "enumerate"                 # "enumerate" or "probes"
    R_probes_high = 1024              # number of random high-k probes each iteration
    resample_probes_each_iter = True
    probes_seed = 1234

    # Model/MCMC/Solver knobs
    d = 25
    k = 4
    mdl  = ModelParams(d=d, N=1024, k=k, sigma_a=1.0, sigma_w=1.0, gamma=1.0, act="relu")
    mcmc = MCMCParams(B=32768*4, steps=steps, step_size=1e-4, step_decay=0.999,
                      grad_clip=1e15, clamp_w=100000.0)
    sol  = SolveParams(ridge_lambda=1e-3, pcg_tol_outer=1e-5, pcg_max_outer=400,
                       pcg_tol_inner=5e-4, pcg_max_inner=200, print_every=10,
                       saem_a0=1.0, saem_t0=150.0, saem_damping=3.0)

    # NORMALIZATION (Schur side)
    oracle_scale = "P"                # "P" or "P2" (strictly validated)

    # Choose which SGLD variant to use:
    #   "P" -> α_variant(w) = Acoef + (P Σ)/κ²  and data term scaled with P²
    #   "1" -> α_variant(w) = Acoef + (Σ)/κ²    and data term unscaled
    sgld_scale = "1"                  # change to "1" for the other variant

    # ================================================
    run_sweep(
        P_list=P_list,
        kappa_list=kappa_list,
        d=d, k=k,
        H_card_small=H_card_small, H_per_card_high=H_per_card_high,
        T_extra=None,
        mdl=mdl, mcmc=mcmc, sol=sol,
        steps=steps,
        save_dir=save_dir,
        run_tag=run_tag,
        oracle_scale=oracle_scale,
        H_mode=H_mode,
        R_probes_high=R_probes_high,
        resample_probes_each_iter=resample_probes_each_iter,
        probes_seed=probes_seed,
        sgld_scale=sgld_scale
    )
