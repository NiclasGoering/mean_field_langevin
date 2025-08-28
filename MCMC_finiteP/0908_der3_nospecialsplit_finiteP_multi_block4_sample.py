# schur_fullkernel_finiteP_gold.py
import os, json, time, math, random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch._dynamo as dynamo

# ----------------------------- Global perf knobs -----------------------------
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

def make_parity_indices(d: int, k: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    S = torch.randperm(d, generator=g)[:k]
    return S.sort().values

def activation(z: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "relu": return torch.relu(z)
    if kind == "tanh": return torch.tanh(z)
    raise ValueError(f"Unknown activation: {kind}")

# ----------------------------- Parity features for a set of modes -----------------------------
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
    idx = idx_padded.to(device)         # (M,kmax)
    msk = mask.to(device)               # (M,kmax)
    Xexp = X.unsqueeze(1).expand(P, M, d)
    idxexp = idx.unsqueeze(0).expand(P, M, kmax)
    feats = torch.gather(Xexp, 2, idxexp)          # (P,M,kmax)
    feats = torch.where(msk.unsqueeze(0), feats, torch.ones_like(feats))
    out = feats.prod(dim=2)                         # (P,M) float ±1
    return out.to(torch.int8)

# ----------------------------- GOLD: i.i.d. sampling of H modes (uniform over subsets) -----------------------------
def sample_modes_iid_uniform_subsets(
    d: int,
    M_H: int,
    teacher_S: torch.Tensor,
    seed: int = 12345,
    exclude_empty: bool = True,
    require_unique: bool = True,
) -> List[Tuple[int]]:
    """
    Gold-standard unbiased sampling of parity modes: draw subsets S ⊆ [d] i.i.d.
    with coordinates included independently with prob 1/2. This is equivalent to
    uniform over all 2^d subsets. We typically exclude the empty subset (constant feature).
    The teacher subset is excluded.
    """
    rng = np.random.default_rng(seed)
    teacher_tuple = tuple(teacher_S.tolist())
    modes: List[Tuple[int]] = []
    seen = set([teacher_tuple]) if require_unique else set()

    def draw_one():
        while True:
            mask = rng.integers(0, 2, size=d, endpoint=False)  # 0/1 bits
            if exclude_empty and mask.sum() == 0:
                continue
            S = tuple(np.nonzero(mask)[0].tolist())
            if require_unique:
                if S in seen:
                    continue
                seen.add(S)
            if S == teacher_tuple:
                continue
            return S

    for _ in range(M_H):
        modes.append(draw_one())
    return modes

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
    B: int = 8192          # chains (reduced)
    steps: int = 50        # SGLD steps per outer iter (reduced)
    step_size: float = 2e-4
    step_decay: float = 0.999
    grad_clip: float = 1e15
    clamp_w: float = 10.0
    langevin_sqrt2: bool = True
    autocast: Optional[bool] = None   # None -> auto

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

# ----------------------------- SGLD sampler (correct sign & scaling) -----------------------------
@dynamo.disable
def sgld_sample_w(
    w: torch.Tensor,      # (B,d)
    X: torch.Tensor,      # (P,d) ±1 float
    C_all: torch.Tensor,  # (P,M_all) float (pass pre-cast)
    y_vec: torch.Tensor,  # (P,) ±1 float
    m_full: torch.Tensor, # (M_all,)
    kappa: float,
    mdl: ModelParams,
    mcmc: MCMCParams
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One SGLD run targeting the finite-P effective action with a integrated out.
    Returns:
      - new w (B,d)
      - phi (P,B)
      - D_oracle (B,) = κ²*α(w) = Acoef*κ² + Σ_hat(w)
      - D_internal (B,) = α(w)   = Acoef + Σ_hat(w)/κ²   (diagnostic)
    """
    device = w.device
    P = X.shape[0]

    # Prior scales (keep mean-field factors exactly)
    gw2   = mdl.sigma_w / mdl.d                     # variance(w_j) = σ_w / d
    Acoef = (mdl.N ** mdl.gamma) / mdl.sigma_a      # α base term

    # Precompute quantities constant within this SGLD call
    Cf   = C_all.to(torch.float32)                  # (P,M)
    y_f  = y_vec.to(torch.float32).view(-1)         # (P,)
    m_f  = m_full.to(torch.float32).view(-1)        # (M,)
    f_mean = (Cf @ m_f).view(-1)                    # (P,)  <-- Hoisted out of loop

    step = mcmc.step_size
    autocast_enabled = mcmc.autocast if (mcmc.autocast is not None) else w.is_cuda

    for _ in range(mcmc.steps):
        w = w.detach().requires_grad_(True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            z = X @ w.t().contiguous()             # (P,B)
            phi = activation(z, mdl.act)

        g = phi.to(torch.float32)                  # fp32 view for reductions
        Sigma = (g*g).mean(dim=0)                  # (B,)
        JY = (g.t() @ y_f) / float(P)              # (B,)
        Jm = (g.t() @ f_mean) / float(P)           # (B,)

        # Correct finite-P α(w)
        D_internal = (Acoef + Sigma / (kappa**2)).clamp_min(1e-9)   # α(w)

        # Energy U(w) = prior + 0.5 log α - 0.5 * (JY-Jm)^2 / (κ^4 α)
        prior_term = 0.5 * (w*w).sum(dim=1) / gw2
        log_det    = 0.5 * torch.log(D_internal)
        data_quad  = (-0.5) * (JY - Jm).pow(2) / ((kappa**4) * D_internal)
        U = prior_term + log_det + data_quad

        # CORRECT SGLD SIGN: w <- w - step*∇U + sqrt(2 step) ξ
        grad = torch.autograd.grad(U.sum(), w, retain_graph=False, create_graph=False)[0]
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

    # Final recompute for outputs
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        z = X @ w.t().contiguous()
        phi = activation(z, mdl.act).detach()
    g = phi.to(torch.float32)
    Sigma = (g*g).mean(dim=0)

    D_oracle   = (Acoef * (kappa**2) + Sigma).clamp_min(1e-9)     # (B,)
    D_internal = (Acoef + Sigma / (kappa**2)).clamp_min(1e-9)     # (B,)
    return w.detach(), phi.detach(), D_oracle.detach(), D_internal.detach()

# ----------------------------- Full-kernel oracle (implicit K matvec) -----------------------------
class FullKernelOracle:
    """
    Implements (I + N B) m = N b using an implicit full kernel:

      Khat v  := (1/B) * sum_b  φ^(b) * [ (φ^(b)^T v) / D_oracle_b ]   ∈ R^P
      B v_M   := (N/P^2) * C^T [ Khat (C v_M) ]
      b       := (N/P^2) * C^T [ Khat y ]

    No diagonal approximation; no need to form P×P matrices.
    """
    def __init__(self, C_pm: torch.Tensor, y: torch.Tensor, N: int):
        self.device = C_pm.device
        self.C = C_pm.to(self.device)             # (P,M) int8 or float
        self.Cf = self.C.float()
        self.Ct = self.Cf.t().contiguous()
        self.P, self.M = self.C.shape
        self.y = y.to(self.device).float()
        self.N = float(N)

        # Buffers filled by update_samples(...)
        self.phi = None          # (P,B) float
        self.D_oracle = None     # (B,)  float

        # Precomputed for diag preconditioner
        self.diag_B_scalar = None

    def update_samples(self, phi: torch.Tensor, D_oracle: torch.Tensor):
        self.phi = phi.to(self.device).float()           # (P,B)
        self.D_oracle = D_oracle.to(self.device).float() # (B,)
        B = self.phi.shape[1]
        invD = (1.0 / self.D_oracle).view(1, B)          # (1,B)
        # For diag preconditioner: sum_μ w_μ with w_μ = mean_b[ φ_μb^2 / D_b ]
        w_mu = (self.phi*self.phi) * invD
        sum_alpha = w_mu.mean(dim=1).sum().item()        # mean over b, sum over μ
        self.diag_B_scalar = (self.N / (self.P**2)) * sum_alpha

    def _Khat(self, vP: torch.Tensor) -> torch.Tensor:
        """
        vP ∈ R^P -> (1/B) Σ_b φ_b * ((φ_b^T vP)/D_b) ∈ R^P
        """
        assert self.phi is not None and self.D_oracle is not None, "call update_samples first"
        vP = vP.to(self.device).float()
        g = torch.matmul(self.phi.t().contiguous(), vP)           # (B,)
        weights = g / self.D_oracle                               # (B,)
        out = torch.matmul(self.phi, weights) / float(self.phi.shape[1])  # (P,)
        return out

    def matvec_B(self, v_full: torch.Tensor) -> torch.Tensor:
        """
        v_full ∈ R^M -> (N/P^2) * C^T [ Khat (C v_full) ] ∈ R^M
        """
        u = torch.matmul(self.Cf, v_full.to(self.device).float())         # (P,)
        Ku = self._Khat(u)                                                # (P,)
        y = torch.matmul(self.Ct, Ku) * (self.N / (self.P**2))            # (M,)
        return y

    def vec_b(self) -> torch.Tensor:
        """
        b = (N/P^2) * C^T [ Khat y ] ∈ R^M
        """
        Ky = self._Khat(self.y)                                           # (P,)
        b = torch.matmul(self.Ct, Ky) * (self.N / (self.P**2))            # (M,)
        return b

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
        denom = (p*Ap).sum().clamp_min(1e-30)
        alpha = rz_old / denom
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

# ----------------------------- Schur solver over H (full kernel) -----------------------------
class SchurSolver:
    """
    Outer CG on T using Schur complement;
    Inner CG on H to apply [(1+λ)I_H + N B_HH]^{-1}.
    Everything built from a FullKernelOracle (implicit K matvecs).
    """
    def __init__(self, oracle: FullKernelOracle, T_idx: torch.Tensor, H_idx: torch.Tensor,
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

    def solve_AH(self, rhs_H):
        def apply_AH(vH):
            return (1.0 + self.lam) * vH + self.apply_B_HH(vH)   # drop explicit N (already in oracle)
        diag = torch.full_like(rhs_H, (1.0 + self.lam) + self.ora.diag_B_scalar)
        z, relres, it = pcg(apply_AH, rhs_H, diag, tol=self.tol_inner, maxit=self.max_inner)
        return z, relres, it

    def build_rhs_T(self, b_full):
        b_T = b_full[self.T_idx]
        b_H = b_full[self.H_idx]
        z_H, relres_in, it_in = self.solve_AH(b_H)                # (I+B_HH)^{-1} b_H
        corr_T = self.apply_B_TH(z_H)
        rhs = b_T - corr_T                                        # drop N factors
        return rhs, {"inner_rhs_relres": relres_in, "inner_rhs_iters": it_in}

    def apply_Schur(self, vT):
        BTv = self.apply_B_TT(vT)
        r_H = self.apply_B_HT(vT)
        z_H, relres_in, it_in = self.solve_AH(r_H)                 # (I+B_HH)^{-1} B_HT vT
        corr_T = self.apply_B_TH(z_H)
        out = (1.0 + self.lam) * vT + BTv - corr_T                 # drop N factors
        return out, {"inner_apply_relres": relres_in, "inner_apply_iters": it_in}

    def reconstruct_mH(self, mT, b_full):
        b_H = b_full[self.H_idx]
        r_H = b_H - self.apply_B_HT(mT)
        mH, relres, it = self.solve_AH(r_H)                        # drop N factor
        return mH, {"rec_relres": relres, "rec_iters": it}

    def solve_outer(self, rhs_T: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        def apply_AT(vT):
            out, info = self.apply_Schur(vT)
            self.last_inner_info = info
            return out
        diag = torch.full_like(rhs_T, (1.0 + self.lam) + self.ora.diag_B_scalar)
        xT, relres, it = pcg(apply_AT, rhs_T, diag, tol=self.tol_outer, maxit=self.max_outer)
        return xT, {"outer_relres": relres, "outer_iters": it, **getattr(self, "last_inner_info", {})}

# ----------------------------- GOLD m_H susceptibility diagnostic -----------------------------
def gold_mH_susceptibility(M_feat: int, teacher_index: int, samples: int = 8192, device: str = "cpu",
                           loss: str = "logistic") -> Dict[str, float]:
    """
    Monte-Carlo check of susceptibility m_H = E[(u^T h)^2] * psi'(0) under i.i.d. dense Rademacher h.
    Here u is unit basis vector at teacher_index. For isotropic Rademacher, E[(u^T h)^2]=1.
    """
    u = torch.zeros(M_feat, device=device)
    u[teacher_index] = 1.0  # unit vector along teacher axis

    H = torch.empty(samples, M_feat, device=device).bernoulli_(0.5).mul_(2).sub_(1)  # ±1 i.i.d.
    b = H @ u  # (samples,)
    Eb2 = (b * b).mean().item()

    if loss == "square":
        psi_p0 = 1.0
    elif loss == "logistic":
        psi_p0 = 0.25
    else:
        raise ValueError("Specify loss as 'square' or 'logistic' to set psi'(0).")
    mH = Eb2 * psi_p0
    return {"E_b2": Eb2, "m_H": mH, "psi_prime_0": psi_p0}

# ----------------------------- Main experiment (single run) -----------------------------
def run_schur_fullkernel(
    d: int = 25, P: int = 2000, k: int = 4, kappa: float = 1e-3,
    M_H: int = 12000,                      # GOLD: number of i.i.d. H modes to sample
    T_extra: List[Tuple[int]] = None,
    mdl: ModelParams = None, mcmc: MCMCParams = None, sol: SolveParams = None,
    outer_steps: int = 300, save_dir: str = "./results_schur_fullkernel", run_tag: str = "",
    *,
    seed_modes: int = 12345,               # RNG for H sampling
    loss_for_mH_diag: str = "logistic"
):
    """
    Gold-standard run:
      - No 'enumerate' or 'probes' modes.
      - H is sampled i.i.d. from the correct prior: uniform over all parity subsets (excluding empty set & teacher).
      - All aggregates are normalized as MEANS (not sums).
      - Includes a Monte-Carlo susceptibility diagnostic m_H under isotropic Rademacher codes in feature space.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mdl = mdl or ModelParams(d=d, N=1024, k=k, sigma_a=1.0, sigma_w=1.0, gamma=1.0, act="relu")
    mcmc = mcmc or MCMCParams(B=8192, steps=50, step_size=2e-4, step_decay=0.999, grad_clip=1e15, clamp_w=10.0)
    sol = sol or SolveParams(ridge_lambda=1e-3, pcg_tol_outer=1e-5, pcg_max_outer=400,
                             pcg_tol_inner=5e-4, pcg_max_inner=200, print_every=10,
                             saem_a0=1.0, saem_t0=150.0, saem_damping=3.0)

    os.makedirs(save_dir, exist_ok=True)

    # Dataset (quenched, dense Rademacher ±1)
    g = torch.Generator(device=device).manual_seed(0)
    X = (torch.randint(0,2,(P,d), generator=g, device=device, dtype=torch.int8).float()*2.0 - 1.0)
    teacher_S = make_parity_indices(d, k, seed=0)
    teacher_tuple = tuple(teacher_S.tolist())

    # Build T list (teacher first) and GOLD H list (i.i.d. uniform over subsets)
    T_list: List[Tuple[int]] = [teacher_tuple]
    if T_extra:
        for t in T_extra:
            t_ = tuple(sorted(t))
            if t_ != teacher_tuple and t_ not in T_list:
                T_list.append(t_)

    H_list: List[Tuple[int]] = sample_modes_iid_uniform_subsets(
        d=d, M_H=M_H, teacher_S=teacher_S, seed=seed_modes, exclude_empty=True, require_unique=True
    )

    # Teacher labels from parity of teacher subset
    T_idx_pad, T_mask, _ = modes_to_padded_idx(T_list)
    C_T = chi_for_modes(X, T_idx_pad, T_mask).float()      # (P, M_T)
    sS  = C_T[:, 0].contiguous().float()                   # (P,)

    # Build C_all and indices
    all_modes = T_list + H_list
    all_idx_pad, all_mask, _ = modes_to_padded_idx(all_modes)
    C_all_i8 = chi_for_modes(X, all_idx_pad, all_mask)     # (P, M_all) int8
    T_idx = torch.arange(0, len(T_list), device=device, dtype=torch.long)
    H_idx = torch.arange(len(T_list), len(all_modes), device=device, dtype=torch.long)

    # Initialize parameters
    w = torch.randn(mcmc.B, d, device=device) * math.sqrt(mdl.sigma_w/mdl.d)
    m_T = torch.zeros(len(T_list), device=device, dtype=torch.float32)
    m_T[0] = 0.5
    m_H_vec = torch.zeros(len(H_list), device=device, dtype=torch.float32)

    print({
        "device": str(device), "P": P, "B": mcmc.B,
        "M_T": len(T_list), "M_H": len(H_list),
        "H_sampling": "iid_uniform_subsets(excluding empty & teacher)"
    })

    # Susceptibility diagnostic in feature space (independent of parity subset choices)
    mH_diag = gold_mH_susceptibility(
        M_feat=len(all_modes), teacher_index=0, samples=min(4*len(all_modes), 20000),
        device=str(device).split(":")[0], loss=loss_for_mH_diag
    )
    print({"gold_mH_diag": mH_diag})

    traj = {"mS": [], "m_noise_norm2": [], "m_noise_rms": [], "err01_clt": [], "mse": [], "time_s": [], "diagB": []}
    t0 = time.time()

    for it in range(1, outer_steps+1):
        # Build full mean-field coeffs and run SGLD (E-step)
        m_full = torch.zeros(C_all_i8.shape[1], device=device)
        m_full[T_idx] = m_T
        m_full[H_idx] = m_H_vec
        C_all_f = C_all_i8.float()

        w, phi, D_oracle, D_internal = sgld_sample_w(
            w, X,
            C_all=C_all_f,
            y_vec=sS,
            m_full=m_full,
            kappa=kappa,
            mdl=mdl,
            mcmc=mcmc
        )

        # Full-kernel oracle and Schur solve (M-step)
        oracle = FullKernelOracle(C_all_i8, sS, N=mdl.N)
        oracle.update_samples(phi, D_oracle)
        b_full = oracle.vec_b()

        schur = SchurSolver(
            oracle, T_idx=T_idx, H_idx=H_idx, N=mdl.N, ridge_lambda=sol.ridge_lambda,
            tol_outer=sol.pcg_tol_outer, max_outer=sol.pcg_max_outer,
            tol_inner=sol.pcg_tol_inner, max_inner=sol.pcg_max_inner
        )

        with torch.no_grad():
            eT = torch.ones(1, device=schur.device)    # teacher basis

            # s1 = b_T - B_TH (I+B_HH)^{-1} b_H
            b_T = b_full[schur.T_idx]                  # (1,)
            b_H = b_full[schur.H_idx]                  # (M_H,)
            z_b, _, _ = schur.solve_AH(b_H)            # (I+B_HH)^{-1} b_H
            corr_T_b = schur.apply_B_TH(z_b)
            s1 = b_T - corr_T_b

            # s2 = (1+λ) + B_TT - B_TH (I+B_HH)^{-1} B_HT
            BT_e = schur.apply_B_TT(eT)
            r_H  = schur.apply_B_HT(eT)
            z_r, _, _ = schur.solve_AH(r_H)
            corr_T_r = schur.apply_B_TH(z_r)
            s2 = (1.0 + schur.lam) * eT + BT_e - corr_T_r

            mS_closed = (s1 / s2).item()
            ratio_cancel = float(corr_T_b.abs().item() / b_T.abs().clamp_min(1e-30).item())
            print({
                "b_T": float(b_T.item()),
                "corr_T_from_bH": float(corr_T_b.item()),
                "cancel_ratio": ratio_cancel,
                "den_s2": float(s2.item()),
                "mS_closed": mS_closed
            })

        rhs_T, rhs_stats = schur.build_rhs_T(b_full)
        mT_star, stats = schur.solve_outer(rhs_T)

        # SAEM smoothing on m_T
        a_t = sol.saem_a0 / (it + sol.saem_t0)
        m_T = (1 - sol.saem_damping*a_t) * m_T + sol.saem_damping*a_t * mT_star
        m_T[0] = m_T[0].clamp_(0.0, 1.0)

        # Reconstruct m_H (diagnostics + next-iter mean field)
        m_H_vec, rec_stats = schur.reconstruct_mH(m_T, b_full)

        # ---- diagnostics ----
        noise_norm2 = float((m_H_vec*m_H_vec).sum().item())
        noise_rms = float((m_H_vec*m_H_vec).mean().sqrt().item())
        mS = float(m_T[0].item())
        mse = 0.5*(1.0 - mS)**2 + noise_norm2

        # Fast CLT-ish 0-1 estimate using erf; robust guards
        den = max(noise_norm2, 1e-30)
        sigma = max(math.sqrt(den), 1e-12)
        z = -mS / sigma
        if z > 12.0:
            clt = 1.0
        elif z < -12.0:
            clt = 0.0
        else:
            clt = 0.5*(1.0 + math.erf(z / math.sqrt(2.0)))

        traj["mS"].append(mS); traj["m_noise_norm2"].append(noise_norm2); traj["m_noise_rms"].append(noise_rms)
        traj["mse"].append(mse); traj["err01_clt"].append(clt); traj["time_s"].append(time.time()-t0)
        traj["diagB"].append(oracle.diag_B_scalar)

        if it % sol.print_every == 1 or it == outer_steps:
            msg = {
                "iter": it, "P": P, "kappa": kappa,
                "m_S": mS, "mse": mse, "err01_clt": clt,
                "noise_rms": noise_rms,
                "pcg_outer_relres": stats["outer_relres"], "pcg_outer_iters": stats["outer_iters"],
                "inner_apply_relres": stats.get("inner_apply_relres", None),
                "inner_apply_iters": stats.get("inner_apply_iters", None),
                "inner_rhs_relres": rhs_stats["inner_rhs_relres"], "inner_rhs_iters": rhs_stats["inner_rhs_iters"],
                "rec_relres": rec_stats["rec_relres"], "rec_iters": rec_stats["rec_iters"],
                "diagB_scalar": oracle.diag_B_scalar,
                "M_T": len(T_list), "M_H": len(H_list),
                "time_s": round(traj["time_s"][-1], 2)
            }
            print(json.dumps(msg))

    # Save snapshot
    out = {
        "summary": {
            "P": P, "kappa": kappa, "M_T": len(T_list), "M_H": len(H_list),
            "mS_last": traj["mS"][-1], "mse_last": traj["mse"][-1], "err01_clt_last": traj["err01_clt"][-1],
            "diagB_last": traj["diagB"][-1]
        },
        "traj": traj,
        "config": {
            "d": d, "k": k,
            "M_H": M_H,
            "T_extra": T_extra,
            "mdl": vars(mdl), "mcmc": vars(mcmc), "sol": vars(sol),
            "seed_modes": seed_modes
        }
    }
    tag = run_tag or time.strftime("%Y%m%d_%H%M%S")
    fname = f"schur_fullkernel_gold_{tag}_kap{float(kappa):.3e}_P{P}_M{len(T_list)+len(H_list)}.json"
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
    M_H: int = 12000,
    T_extra: List[Tuple[int]] = None,
    mdl: ModelParams = None,
    mcmc: MCMCParams = None,
    sol: SolveParams = None,
    outer_steps: int = 300,
    save_dir: str = "./results_schur_fullkernel",
    run_tag: str = "",
    seed_modes: int = 12345,
    loss_for_mH_diag: str = "logistic"
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
            res, path = run_schur_fullkernel(
                d=d, P=P, k=k, kappa=kap,
                M_H=M_H, T_extra=T_extra,
                mdl=mdl, mcmc=mcmc, sol=sol, outer_steps=outer_steps,
                save_dir=save_dir, run_tag=index["run_tag"],
                seed_modes=seed_modes,
                loss_for_mH_diag=loss_for_mH_diag
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
    P_list     = [5000]        # e.g. [100, 500, 1000, 2000]
    kappa_list = [1e-3]        # e.g. [1e-3, 1e-2]
    outer_steps = 10000         # outer iterations (decoupled from SGLD steps)
    save_dir = "./results_schur_fullkernel"
    run_tag  = ""

    # GOLD H SAMPLING
    M_H = 40000                # number of i.i.d. H modes (uniform over subsets, exclude empty & teacher)
    seed_modes = 12345

    # Model/MCMC/Solver knobs
    d = 25
    k = 4
    mdl  = ModelParams(d=d, N=1024, k=k, sigma_a=1.0, sigma_w=1.0, gamma=1.0, act="relu")
    mcmc = MCMCParams(B=8192*4, steps=250, step_size=2e-4, step_decay=0.999,
                      grad_clip=1e15, clamp_w=10.0, autocast=True)
    sol  = SolveParams(ridge_lambda=1e-3, pcg_tol_outer=1e-5, pcg_max_outer=400,
                       pcg_tol_inner=5e-4, pcg_max_inner=200, print_every=10,
                       saem_a0=1.0, saem_t0=150.0, saem_damping=3.0)

    # ================================================
    run_sweep(
        P_list=P_list,
        kappa_list=kappa_list,
        d=d, k=k,
        M_H=M_H,
        T_extra=None,
        mdl=mdl, mcmc=mcmc, sol=sol,
        outer_steps=outer_steps,
        save_dir=save_dir,
        run_tag=run_tag,
        seed_modes=seed_modes,
        loss_for_mH_diag="logistic"
    )
