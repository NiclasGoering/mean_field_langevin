# schur_quenched_finiteP.py
# schur_quenched_finiteP.py
import os, json, time, math, random
from dataclasses import dataclass
from itertools import combinations
from typing import List, Tuple, Dict

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
    if len(rows) == 0: return torch.empty((0, c), dtype=torch.long)
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

    T_list = [teacher_tuple]
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
        if all_c.numel() == 0: continue
        keep = [tuple(row.tolist()) for row in all_c if tuple(row.tolist()) != teacher_tuple]
        if len(keep) == 0: continue
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
    idx = idx_padded.to(device)         # (M,k_max)
    msk = mask.to(device)               # (M,k_max)
    # Expand X to (P, M, d)
    Xexp = X.unsqueeze(1).expand(P, M, d)
    idxexp = idx.unsqueeze(0).expand(P, M, kmax)
    feats = torch.gather(Xexp, 2, idxexp)          # (P,M,kmax)
    # set ones where mask False to multiplicative identity
    feats = torch.where(msk.unsqueeze(0), feats, torch.ones_like(feats))
    out = feats.prod(dim=2)                         # (P,M) float ±1
    return (out.to(torch.int8))

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
    grad_clip: float = 1e4
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

# ----------------------------- SGLD (E-step) -----------------------------
@dynamo.disable
def sgld_sample_w(
    w: torch.Tensor,      # (B,d)
    X: torch.Tensor,      # (P,d) ±1 float
    C_T: torch.Tensor,    # (P,M_T) ±1 float for tracked T (default M_T=1 teacher)
    sS: torch.Tensor,     # (P,) ±1 float (teacher parity per-sample)
    m_T: torch.Tensor,    # (M_T,)
    kappa: float,
    mdl: ModelParams,
    mcmc: MCMCParams
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One full SGLD run. Returns:
      - new w (B,d)
      - phi (P,B) activations at the final iterate (to build alpha)
      - D (B,) final
    Uses the exact finite-P posterior with the correct factor -(P/2) log D.
    """
    device = w.device
    P = X.shape[0]
    gw2 = (mdl.sigma_w ** 2) / mdl.d            # prior var per weight component
    Acoef = (mdl.N ** mdl.gamma) / (mdl.sigma_a ** 2)  # 1/ga2

    step = mcmc.step_size
    m_T_col = m_T.view(-1,1)  # (M_T,1)

    for _ in range(mcmc.steps):
        w = w.detach().requires_grad_(True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=w.is_cuda):
            # z: (P,B)
            z = torch.matmul(X, w.t().contiguous())
            phi = activation(z, mdl.act)
        # build J terms in fp32
        phi_f = phi.float()                                 # (P,B)
        # J_Y = (1/P) Phi^T sS  -> (B,)
        JY = torch.matmul(phi_f.t(), sS.float()) / float(P)

        # J_T(m_T) = (1/P) Phi^T (C_T m_T) -> (B,)
        CmT     = torch.matmul(C_T.float(), m_T_col).squeeze(-1)  # (P,)
        Jm      = torch.matmul(phi_f.t(), CmT) / float(P)         # (B,)
        # D(b) = Acoef kappa^2 + (1/P) sum_i u_i^2
        Sigma = (phi_f * phi_f).mean(dim=0)                 # (B,)
        D = (Acoef * (kappa ** 2) + Sigma).clamp_min(1e-9)  # (B,)

        # log posterior
        prior_term = -0.5 * (w*w).sum(dim=1) / gw2                         # (B,)
        data_quad  = -0.5 * (JY - Jm).pow(2) / ((kappa**2) * D)            # CORRECT

        log_det    = -0.5 * float(P) * torch.log(D)                        # (B,)
        logp = prior_term + data_quad + log_det

        grad = torch.autograd.grad(logp.sum(), w, retain_graph=False, create_graph=False)[0]
        grad = torch.where(torch.isfinite(grad), grad, torch.zeros_like(grad))
        if mcmc.grad_clip and mcmc.grad_clip>0:
            gn = grad.norm(dim=1, keepdim=True).clamp_min(1e-12)
            grad = grad * (mcmc.grad_clip/gn).clamp(max=1.0)

        noise = torch.randn_like(w)
        if mcmc.langevin_sqrt2:
            w = w + step * grad + noise * math.sqrt(2.0*step)
        else:
            w = w + 0.5*step*grad + noise * math.sqrt(step)
        if mcmc.clamp_w: w = torch.clamp(w, -mcmc.clamp_w, mcmc.clamp_w)
        step *= mcmc.step_decay

    # final phi, D reused to build alpha later
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=w.is_cuda):
        z = torch.matmul(X, w.t().contiguous())             # (P,B)
        phi = activation(z, mdl.act)
    phi_f = phi.float()
    Sigma = (phi_f*phi_f).mean(dim=0)
    D = (Acoef * (kappa**2) + Sigma).clamp_min(1e-9)
    return w.detach(), phi.detach(), D.detach()

# ----------------------------- B/b oracles and matvecs -----------------------------
class BbOracle:
    """
    Implements y = B v and b, using:
      B = (1/P^2) * C^T diag(alpha) C
      b = (1/P^2) * C^T (alpha ⊙ sS)
    where alpha_i = mean_b (u_i^2 / D_b).
    """
    def __init__(self, C_pm: torch.Tensor, sS: torch.Tensor, device: torch.device):
        # C_pm: (P,M) int8 ±1
        self.device = device
        self.C = C_pm.to(device)
        self.P, self.M = self.C.shape
        # use float32 copies for fast GEMM
        self.Cf = self.C.float()
        self.Ct = self.Cf.t().contiguous()   # (M,P)
        self.sS = sS.to(device).float()
        self.alpha = torch.ones(self.P, device=device)
        self.diag_B_scalar = float(self.alpha.sum().item()) / (self.P**2)

    def update_alpha(self, phi: torch.Tensor, D: torch.Tensor):
        """
        phi: (P,B) float/bf16
        D:   (B,)  float
        alpha_i = mean_b ( u_i^2 / D_b )
        """
        phi_f = phi.float()
        B = phi_f.shape[1]
        invD = (1.0 / D.float()).view(1, B)          # (1,B)
        a = (phi_f*phi_f) * invD                     # (P,B)
        self.alpha = a.mean(dim=1).contiguous()      # (P,)
        self.diag_B_scalar = float(self.alpha.sum().item()) / (self.P**2)

    def matvec_B(self, v_full: torch.Tensor) -> torch.Tensor:
        """
        v_full: (M,)
        returns y = B v = (1/P^2) C^T ( alpha ⊙ (C v) )
        """
        s = torch.matmul(self.Cf, v_full)               # (P,)
        y = torch.matmul(self.Ct, self.alpha * s) / float(self.P**2)  # (M,)
        return y

    def vec_b(self) -> torch.Tensor:
        """
        b = (1/P^2) * C^T (alpha ⊙ sS)
        """
        return torch.matmul(self.Ct, self.alpha * self.sS) / float(self.P**2)

# ----------------------------- PCG utilities -----------------------------
def pcg(apply_A, rhs, diag_A, tol=1e-5, maxit=400):
    """
    Jacobi-PCG on GPU. All tensors on same device.
    """
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
        self.T_idx = T_idx  # (M_T,)
        self.H_idx = H_idx  # (M_H,)
        self.N = float(N)
        self.lam = float(ridge_lambda)
        self.tol_outer = tol_outer
        self.max_outer = max_outer
        self.tol_inner = tol_inner
        self.max_inner = max_inner

        self.M_T = T_idx.numel()
        self.M_H = H_idx.numel()
        self.device = oracle.C.device

        # diag(B) is constant across coords: alpha.sum()/P^2
        self.diagB_scalar = oracle.diag_B_scalar
        self.diag_AH = (1.0 + self.lam) + self.N * self.diagB_scalar
        self.diag_AT = (1.0 + self.lam) + self.N * self.diagB_scalar

        self.M_all = oracle.M
        self.mask_T = torch.zeros(self.M_all, device=self.device, dtype=torch.bool)
        self.mask_H = torch.zeros(self.M_all, device=self.device, dtype=torch.bool)
        self.mask_T[T_idx] = True
        self.mask_H[H_idx] = True

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
        """
        rhs_T = N b_T - N^2 B_TH AH^{-1} b_H
        """
        b_T = b_full[self.T_idx]
        b_H = b_full[self.H_idx]
        z_H, relres_in, it_in = self.solve_AH(b_H)
        corr_T = self.apply_B_TH(z_H)
        rhs = self.N * b_T - (self.N**2) * corr_T
        stats = {"inner_rhs_relres": relres_in, "inner_rhs_iters": it_in}
        return rhs, stats

    def apply_Schur(self, vT: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        S(vT) = (1+λ)vT + N B_TT vT - N^2 B_TH AH^{-1} B_HT vT
        """
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
        """
        m_H = AH^{-1} [ N (b_H - B_HT m_T) ]
        """
        b_H = b_full[self.H_idx]
        r_H = b_H - self.apply_B_HT(mT)
        rhs_H = self.N * r_H
        mH, relres, it = self.solve_AH(rhs_H)
        return mH, {"rec_relres": relres, "rec_iters": it}

# ----------------------------- Main experiment (single run) -----------------------------
def run_schur_quenched(
    d: int = 25, P: int = 2000, k: int = 4, kappa: float = 1e-3,
    H_card_small: int = 2, H_per_card_high: int = 50, T_extra: List[Tuple[int]] = None,
    mdl: ModelParams = None, mcmc: MCMCParams = None, sol: SolveParams = None,
    steps: int = 300, save_dir: str = "./results_schur_quenched", run_tag: str = ""
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mdl = mdl or ModelParams(d=d, N=1024, k=k, sigma_a=1.0, sigma_w=1.0, gamma=1.0, act="relu")
    mcmc = mcmc or MCMCParams(B=32768, steps=400, step_size=2e-3, step_decay=0.999, grad_clip=1e4, clamp_w=10.0)
    sol = sol or SolveParams(ridge_lambda=1e-3, pcg_tol_outer=1e-5, pcg_max_outer=400,
                             pcg_tol_inner=5e-4, pcg_max_inner=200, print_every=10,
                             saem_a0=1.0, saem_t0=150.0, saem_damping=3.0)

    os.makedirs(save_dir, exist_ok=True)

    # Dataset (quenched)
    g = torch.Generator(device=device).manual_seed(0)
    X = (torch.randint(0,2,(P,d), generator=g, device=device, dtype=torch.int8).float()*2.0 - 1.0)
    teacher_S = make_parity_indices(d, k, seed=0)

    # Build T/H
    T_list, H_list = build_modes_T_H(d, teacher_S, H_card_small, H_per_card_high, seed=1, T_extra=T_extra)
    assert tuple(teacher_S.tolist()) == T_list[0]  # teacher first

    # Convert lists to padded and chi
    T_idx_pad, T_mask, _ = modes_to_padded_idx(T_list)
    H_idx_pad, H_mask, _ = modes_to_padded_idx(H_list)
    all_modes = T_list + H_list
    all_idx_pad, all_mask, _ = modes_to_padded_idx(all_modes)

    # χ for all modes
    C_all = chi_for_modes(X, all_idx_pad, all_mask)  # (P, M_all) int8
    M_all = C_all.shape[1]
    # Indices for T and H inside the concatenated list
    T_idx = torch.arange(0, len(T_list), device=device, dtype=torch.long)
    H_idx = torch.arange(len(T_list), len(all_modes), device=device, dtype=torch.long)

    # Teacher per-sample sign
    C_T = chi_for_modes(X, T_idx_pad, T_mask).float() # (P, M_T)
    sS = C_T[:,0].contiguous().float()

    # Oracles
    oracle = BbOracle(C_all, sS, device=device)

    # Initialize w and m_T
    w = torch.randn(mcmc.B, d, device=device) * math.sqrt(mdl.sigma_w/mdl.d)
    m_T = torch.zeros(len(T_list), device=device, dtype=torch.float32)
    m_T[0] = 0.5  # teacher mean init

    # Logs
    print({"device": str(device), "P": P, "B": mcmc.B, "M_T": len(T_list), "M_H": len(H_list)})
    traj = {"mS": [], "m_noise_norm2": [], "m_noise_rms": [], "err01_clt": [], "mse": [], "time_s": []}
    t0 = time.time()

    for it in range(1, steps+1):
        # --------- E-step: SGLD over w on the same fixed dataset ----------
        w, phi, D = sgld_sample_w(w, X, C_T, sS, m_T, kappa, mdl, mcmc)

        # --------- Update alpha, build B/b oracles ----------
        oracle.update_alpha(phi, D)
        b_full = oracle.vec_b()

        # --------- Schur solve on T ----------
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
        # projection on teacher in [0,1]
        m_T[0] = m_T[0].clamp(0.0, 1.0)

        # Reconstruct m_H for diagnostics
        m_H, rec_stats = schur.reconstruct_mH(m_T, b_full)
        noise_norm2 = float((m_H*m_H).sum().item())
        noise_rms = float((m_H*m_H).mean().sqrt().item())
        mS = float(m_T[0].item())
        # Theory-consistent MSE and CLT 0-1 error
        mse = 0.5*(1.0 - mS)**2 + noise_norm2
        denom = max(noise_norm2, 1e-30)
        clt = float(torch.distributions.Normal(0,1).cdf(torch.tensor(-mS/math.sqrt(denom))).item())

        traj["mS"].append(mS); traj["m_noise_norm2"].append(noise_norm2); traj["m_noise_rms"].append(noise_rms)
        traj["mse"].append(mse); traj["err01_clt"].append(clt); traj["time_s"].append(time.time()-t0)

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
                "diagB_scalar": oracle.diag_B_scalar,
                "time_s": round(traj["time_s"][-1], 2)
            }
            print(json.dumps(msg))

    # Save snapshot
    out = {
        "summary": {
            "P": P, "kappa": kappa, "M_T": len(T_list), "M_H": len(H_list),
            "mS_last": traj["mS"][-1], "mse_last": traj["mse"][-1],
            "err01_clt_last": traj["err01_clt"][-1]
        },
        "traj": traj,
        "config": {
            "d": d, "k": k, "H_card_small": H_card_small, "H_per_card_high": H_per_card_high,
            "T_extra": T_extra, "mdl": vars(mdl), "mcmc": vars(mcmc), "sol": vars(sol)
        }
    }
    tag = run_tag or time.strftime("%Y%m%d_%H%M%S")
    fname = f"schur_quenched_{tag}_kap{float(kappa):.3e}_P{P}_M{M_all}.json"
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
    run_tag: str = ""
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
                save_dir=save_dir, run_tag=index["run_tag"]
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
    P_list     = [100, 500,1000, 2000]
    kappa_list = [1e-3]              # e.g. [1e-3, 1e-2]
    steps = 800
    save_dir = "/home/goring/mean_field_langevin/MCMC_finiteP/results4/c2_h200"
    run_tag  = ""  # optional string; if empty, timestamp is used
    # H split: include all modes up to this cardinality, then sample this many per higher card
    H_card_small     = 2
    H_per_card_high  = 200
    # Model/MCMC/Solver knobs
    d = 25
    k = 4
    mdl  = ModelParams(d=d, N=1024, k=k, sigma_a=1.0, sigma_w=1.0, gamma=1.0, act="relu")
    mcmc = MCMCParams(B=32768, steps=steps, step_size=2e-3, step_decay=0.999,
                      grad_clip=1e4, clamp_w=10.0)
    sol  = SolveParams(ridge_lambda=1e-3, pcg_tol_outer=1e-5, pcg_max_outer=400,
                       pcg_tol_inner=5e-4, pcg_max_inner=200, print_every=10,
                       saem_a0=1.0, saem_t0=150.0, saem_damping=3.0)
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
        run_tag=run_tag
    )
