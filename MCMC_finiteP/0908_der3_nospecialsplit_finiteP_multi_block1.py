import os, json, time, math, random
from typing import Dict, List, Tuple
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch
import torch._dynamo as dynamo

# --------------------------------------------------------------------------------------
# Global perf knobs
# --------------------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# --------------------------------------------------------------------------------------
# Utils
# --------------------------------------------------------------------------------------

def check_gpu():
    if not torch.cuda.is_available():
        print("CUDA not available. Running on CPU.", flush=True)
        return []
    n = torch.cuda.device_count()
    info = []
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)
        total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        info.append({"index": i, "name": name, "capability": cap, "mem_GB": round(total_mem, 2)})
    print("GPUs:", info, flush=True)
    return list(range(n))

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

_get = lambda BASE, k, default=None: BASE.get(k, default)

# --------------------------------------------------------------------------------------
# Core math helpers
# --------------------------------------------------------------------------------------

def make_parity_indices(d: int, k: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    S = torch.randperm(d, generator=g)[:k]
    return S.sort().values


def enumerate_combos(d: int, c: int) -> torch.Tensor:
    rows = list(combinations(range(d), c))
    if len(rows) == 0:
        return torch.empty((0, c), dtype=torch.long)
    return torch.tensor(rows, dtype=torch.long)


def build_modes_index_varcard(
    d: int,
    teacher_S: torch.Tensor,
    card_track_max: int,
    sample_high_per_card: int,
    seed: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a set of modes including the teacher + all modes up to `card_track_max`
    and a random sample (per cardinality) above. Teacher is row 0.

    Returns:
      idx_padded: (M, k_max) long — index tuples padded with zeros
      mask:       (M, k_max) bool — true where index is valid
    """
    rng = np.random.default_rng(seed)
    mode_lists: List[List[int]] = []

    # teacher first
    mode_lists.append(teacher_S.tolist())

    for c in range(1, d + 1):
        all_c = enumerate_combos(d, c)
        if c == teacher_S.numel() and all_c.numel() > 0:
            teacher_tuple = tuple(teacher_S.tolist())
            keep = [row for row in all_c.tolist() if tuple(row) != teacher_tuple]
            all_c = torch.tensor(keep, dtype=torch.long) if len(keep) > 0 else torch.empty((0, c), dtype=torch.long)

        Nc = all_c.size(0)
        if Nc == 0:
            continue
        if c <= card_track_max:
            mode_lists.extend([row for row in all_c.tolist()])
        else:
            if sample_high_per_card > 0:
                k_take = min(sample_high_per_card, Nc)
                idx = rng.choice(Nc, size=k_take, replace=False)
                mode_lists.extend([all_c[i].tolist() for i in idx])

    if len(mode_lists) == 0:
        mode_lists = [teacher_S.tolist()]

    k_max = max(len(x) for x in mode_lists)
    M = len(mode_lists)
    idx_padded = torch.zeros((M, k_max), dtype=torch.long)
    mask = torch.zeros((M, k_max), dtype=torch.bool)
    for i, row in enumerate(mode_lists):
        L = len(row)
        if L > 0:
            idx_padded[i, :L] = torch.tensor(row, dtype=torch.long)
            mask[i, :L] = True
    return idx_padded, mask


def parity_blocks_from_indices(X_blocks: torch.Tensor, modes_idx_pad: torch.Tensor, modes_mask: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """Compute parity features chi for all blocks and modes.

    Args:
      X_blocks:       (K, P, d) float32 in {+1,-1}
      modes_idx_pad:  (M, k_max) long
      modes_mask:     (M, k_max) bool

    Returns:
      chi: (K, P, M) int8 in {+1,-1}
    """
    K, P, d = X_blocks.shape
    M, k_max = modes_idx_pad.shape
    device = X_blocks.device
    chi = torch.empty((K, P, M), device=device, dtype=torch.int8)

    if chunk <= 0:
        chunk = M
    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        idx = modes_idx_pad[s:e].to(device)           # (mchunk,k_max)
        msk = modes_mask[s:e].to(device)              # (mchunk,k_max)

        idx_exp = idx.view(1, 1, e - s, k_max).expand(K, P, e - s, k_max)
        gathered = torch.gather(
            X_blocks.unsqueeze(2).expand(K, P, e - s, d),  # float
            3,
            idx_exp
        )  # (K,P,mchunk,k_max) float

        msk_exp = msk.view(1, 1, e - s, k_max).expand(K, P, e - s, k_max)
        ones = torch.ones_like(gathered)
        gathered = torch.where(msk_exp, gathered, ones)  # float

        out = gathered.prod(dim=3)  # float +/-1
        chi[:, :, s:e] = out.to(torch.int8)
    return chi


def make_boolean_blocks(K: int, P: int, d: int, device: torch.device, seed: int) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.randint(0, 2, (K, P, d), generator=g, device=device, dtype=torch.int8)
    return X.to(torch.float32) * 2.0 - 1.0


def activation(z: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "relu":
        return torch.relu(z)
    if kind == "tanh":
        return torch.tanh(z)
    raise ValueError(f"Unknown activation: {kind}")

# --------------------------------------------------------------------------------------
# Params
# --------------------------------------------------------------------------------------

@dataclass
class ModelParams:
    d: int = 30
    N: int = 1024
    k: int = 4
    sigma_a: float = 1.0
    sigma_w: float = 1.0
    gamma: float = 1.0
    act: str = "relu"

@dataclass
class MCMCParams:
    n_chains_per_device: int = 1024   # B
    n_steps: int = 300
    step_size: float = 5e-3
    step_decay: float = 0.999
    langevin_sqrt2: bool = True
    grad_clip: float = 1e5
    clamp_w: float = 10.0

@dataclass
class SAEMParams:
    max_iters: int = 800
    a0: float = 1.0
    t0: float = 150.0
    damping_m: float = 3.0
    eps_D: float = 1e-9
    print_every: int = 20
    project_mS: bool = False
    m_noise_clip: float = 0.0
    ridge_lambda: float = 1e-4
    blocks: int = 4
    resample_blocks_every: int = 0
    crossfit_shuffle_each_iter: bool = False

# --------------------------------------------------------------------------------------
# E-step: log pi(w|m, D) and grad (exact finite-P, single quenched dataset)
# --------------------------------------------------------------------------------------

@dynamo.disable
def compute_logp_and_grad_w_blocks(
    w_blocks: torch.Tensor,
    X_blocks: torch.Tensor,
    C_all: torch.Tensor,             # (K,P,M_all) — all tracked modes included here
    m_all: torch.Tensor,             # (K,M_all)
    C_teacher_col: torch.Tensor,     # (K,P) float32
    kappa: float,
    mdl: ModelParams,
    saem: SAEMParams,
    backreact_lambda: float = 1.0,
):
    """Exact finite-P posterior for w after integrating out 'a' on the SAME dataset."""
    # Prior variance per weight component
    gw2 = (mdl.sigma_w ** 2) / mdl.d
    # Precision on a: A = N^gamma / sigma_a^2
    A = (mdl.N ** mdl.gamma) / (mdl.sigma_a ** 2)

    w_blocks = w_blocks.detach().requires_grad_(True)

    # Compute phi in fp32 for stability
    z = torch.matmul(X_blocks, w_blocks.transpose(2, 1).contiguous())              # (K,P,B)
    phi = activation(z, mdl.act).float()

    Ptot = X_blocks.shape[1]

    # J_Y = (1/P) Phi^T C_S (teacher column)
    JY = torch.matmul(phi.transpose(1, 2).contiguous(), C_teacher_col.unsqueeze(2)).squeeze(2) / float(Ptot)  # (K,B)

    # J_m with all tracked modes
    Cm = torch.matmul(C_all.float(), m_all.float().unsqueeze(2)).squeeze(2)        # (K,P)
    Jm = torch.matmul(phi.transpose(1, 2).contiguous(), Cm.unsqueeze(2)).squeeze(2) / float(Ptot)             # (K,B)

    # Sigma and D (fp32)
    Sigma = (phi * phi).mean(dim=1)  # (K,B)
    D = torch.clamp(A * (kappa ** 2) + Sigma, min=saem.eps_D)

    # log posterior (up to const): prior + (1/(2 kappa^2)) (JY - lambda*Jm)^2 / D - 0.5 log D
    Jdiff = (JY - backreact_lambda * Jm)
    prior_term = -0.5 * (w_blocks * w_blocks).sum(dim=2) / gw2
    data_quad = 0.5 * (Jdiff * Jdiff) / ((kappa ** 2) * D)
    logp = prior_term + data_quad - 0.5 * torch.log(D)

    grad = torch.autograd.grad(logp.sum(), w_blocks, create_graph=False, retain_graph=False)[0]
    grad = torch.where(torch.isfinite(grad), grad, torch.zeros_like(grad))
    return logp.detach(), grad.detach()


def mcmc_sgld_w_blocks(
    w_blocks: torch.Tensor,
    X_blocks: torch.Tensor,
    C_all: torch.Tensor,
    m_all: torch.Tensor,
    C_teacher_col: torch.Tensor,
    kappa: float,
    mdl: ModelParams,
    saem: SAEMParams,
    mcmc: MCMCParams,
    backreact_lambda: float = 1.0,
):
    step = mcmc.step_size
    for _ in range(mcmc.n_steps):
        _, grad = compute_logp_and_grad_w_blocks(
            w_blocks, X_blocks, C_all, m_all, C_teacher_col, kappa, mdl, saem, backreact_lambda
        )
        if mcmc.grad_clip and mcmc.grad_clip > 0:
            gn = grad.norm(dim=2, keepdim=True).clamp_min(1e-12)
            grad = grad * (mcmc.grad_clip / gn).clamp(max=1.0)

        noise = torch.randn_like(w_blocks)
        if mcmc.langevin_sqrt2:
            w_blocks = w_blocks + step * grad + noise * math.sqrt(2.0 * step)
        else:
            w_blocks = w_blocks + 0.5 * step * grad + noise * math.sqrt(step)

        if mcmc.clamp_w:
            w_blocks = torch.clamp(w_blocks, -mcmc.clamp_w, mcmc.clamp_w)
        step *= mcmc.step_decay
    return w_blocks.detach()

# --------------------------------------------------------------------------------------
# PCG (Jacobi) for batched linear systems A x = rhs (robust)
# --------------------------------------------------------------------------------------

def pcg_solve_operator(apply_A, rhs, diag_A, tol=1e-6, max_iter=200):
    """
    PCG with Jacobi preconditioner. Batched over K rows.
    Strong NaN/Inf guards. Returns (x, relres_mean, iters).
    """
    x = torch.zeros_like(rhs)
    r = rhs - apply_A(x)
    r = torch.nan_to_num(r)
    Minv = 1.0 / diag_A.clamp_min(1e-30)
    z = Minv * r
    p = z.clone()

    rz = (r * z).sum(dim=1, keepdim=True)
    rhs_norm = rhs.norm(dim=1, keepdim=True).clamp_min(1e-30)

    it = 0
    while it < max_iter:
        Ap = apply_A(p)
        Ap = torch.nan_to_num(Ap)
        denom = (p * Ap).sum(dim=1, keepdim=True).clamp_min(1e-30)
        alpha = rz / denom
        x = x + alpha * p
        r = r - alpha * Ap
        r = torch.nan_to_num(r)
        relres = (r.norm(dim=1, keepdim=True) / rhs_norm)
        if (relres <= tol).all():
            break
        z_new = Minv * r
        rz_new = (r * z_new).sum(dim=1, keepdim=True)
        beta = rz_new / rz.clamp_min(1e-30)
        p = z_new + beta * p
        rz = rz_new
        it += 1

    return x, float(relres.mean().item()), it + 1

# --------------------------------------------------------------------------------------
# SAEM with FULL-SYSTEM solve (no Schur), using cached T = Phi^T C / P
# --------------------------------------------------------------------------------------

@dataclass
class Sanity:
    pass


def saem_quenched_full(
    kappa: float, P: int, devices: List[int], BASE: Dict, SAVE_DIR: str, run_tag: str = ""
):
    # ----------------- params -----------------
    mdl = ModelParams(
        d=_get(BASE, "d", 25), N=_get(BASE, "N", 1024), k=_get(BASE, "k", 4),
        sigma_a=_get(BASE, "σa", 1.0), sigma_w=_get(BASE, "σw", 1.0),
        gamma=_get(BASE, "γ", 1.0), act=_get(BASE, "act", "relu")
    )
    mcmc = MCMCParams(
        n_chains_per_device=_get(BASE, "chains_per_device", 2048),
        n_steps=_get(BASE, "mcmc_steps", 200),
        step_size=_get(BASE, "mcmc_step_size", 1e-3),
        step_decay=_get(BASE, "mcmc_step_decay", 0.999),
        langevin_sqrt2=_get(BASE, "langevin_sqrt2", True),
        grad_clip=_get(BASE, "grad_clip", 1e5),
        clamp_w=_get(BASE, "clamp_w", 10.0),
    )
    saem = SAEMParams(
        max_iters=_get(BASE, "opt_steps", 400), a0=_get(BASE, "a0", 1.0), t0=_get(BASE, "t0", 150.0),
        damping_m=_get(BASE, "damping_m", 3.0), eps_D=_get(BASE, "eps_D", 1e-9),
        print_every=_get(BASE, "print_every", 10), project_mS=_get(BASE, "project_mS", False),
        m_noise_clip=_get(BASE, "m_noise_clip", 0.0), ridge_lambda=_get(BASE, "ridge_lambda", 1e-3),
        blocks=_get(BASE, "blocks", 1), resample_blocks_every=_get(BASE, "resample_blocks_every", 0),
        crossfit_shuffle_each_iter=_get(BASE, "crossfit_shuffle_each_iter", False),
    )

    # solver knobs
    PCG_MAX_ITERS = int(_get(BASE, "pcg_max_iters", 2000))
    PCG_TOL = float(_get(BASE, "pcg_tol", 1e-5))
    FALLBACK_TO_DIAG = bool(_get(BASE, "fallback_to_diag_if_bad", True))
    FALLBACK_RELRES_MAX = float(_get(BASE, "fallback_relres_max", 1e-3))

    # modes selection
    teacher_S = make_parity_indices(mdl.d, mdl.k, seed=_get(BASE, "teacher_seed", 0))
    card_track_max = int(_get(BASE, "card_track_max", 2))
    sample_high_per_card = int(_get(BASE, "sample_high_per_card", 2000))

    modes_idx_pad, modes_mask = build_modes_index_varcard(
        d=mdl.d, teacher_S=teacher_S, card_track_max=card_track_max, sample_high_per_card=sample_high_per_card,
        seed=_get(BASE, "noise_seed", 1)
    )

    M_all = modes_idx_pad.shape[0]

    per_dev = []
    for di in devices if len(devices) > 0 else [-1]:
        device = torch.device(f"cuda:{di}") if di >= 0 and torch.cuda.is_available() else torch.device("cpu")
        K = saem.blocks
        B = mcmc.n_chains_per_device

        # Proper weight init: std = sigma_w / sqrt(d)
        w_blocks = torch.randn(K, B, mdl.d, device=device) * (mdl.sigma_w / math.sqrt(mdl.d))

        data_seed = _get(BASE, "data_seed", 0) + 10000 * (di + 1)
        X_blocks = make_boolean_blocks(K, P, mdl.d, device, seed=data_seed)

        # chi for selected modes (teacher is index 0)
        chi_all = parity_blocks_from_indices(X_blocks, modes_idx_pad.to(device), modes_mask.to(device))
        C_all = chi_all  # (K,P,M_all)
        C_teacher = chi_all[:, :, 0].float()

        # coefficients
        m_all = torch.zeros(K, M_all, device=device, dtype=torch.float32)
        m_all[:, 0] = _get(BASE, "m_init", 0.5)

        print({"device": str(device), "K": K, "B": B, "P": P, "M_all": M_all}, flush=True)
        per_dev.append(dict(device=device, X_blocks=X_blocks, w_blocks=w_blocks, C_all=C_all, C_teacher=C_teacher, m_all=m_all))

    # constant A = N^gamma / sigma_a^2
    Acoef = (mdl.N ** mdl.gamma) / (mdl.sigma_a ** 2)

    t_start = time.time()
    traj_mS, traj_mnoise_rms, traj_time_s = [], [], []

    for it in range(1, saem.max_iters + 1):
        # back-reaction ramp for E-step (optional)
        br_ramp_iters = int(_get(BASE, "backreact_ramp_iters", 0))
        br_ramp_pow = float(_get(BASE, "backreact_ramp_pow", 1.0))
        lam_br = min(1.0, (it / float(br_ramp_iters)) ** br_ramp_pow) if br_ramp_iters else 1.0

        # ------------------ E-step: SGLD over w ------------------
        for slot in per_dev:
            slot["w_blocks"] = mcmc_sgld_w_blocks(
                slot["w_blocks"], slot["X_blocks"], slot["C_all"], slot["m_all"], slot["C_teacher"],
                kappa, mdl, saem, mcmc, backreact_lambda=lam_br
            )

        # ------------------ M-step: FULL system solve ------------------
        for slot in per_dev:
            device = slot["device"]
            Xb, C_all = slot["X_blocks"], slot["C_all"].float()
            w_blocks, m_all = slot["w_blocks"], slot["m_all"]
            Kb, Ptot, _ = Xb.shape
            Bsize = w_blocks.shape[1]

            # Phi, S, D (fp32) on same dataset
            phi = activation(torch.matmul(Xb, w_blocks.transpose(2, 1).contiguous()), mdl.act).float()  # (K,P,B)
            S = (phi * phi).mean(dim=1)                                                                # (K,B)
            D = torch.clamp(Acoef * (kappa ** 2) + S, min=saem.eps_D)
            invD = 1.0 / D

            # Teacher empirical J_S
            J_S = torch.matmul(phi.transpose(1, 2).contiguous(), slot["C_teacher"].unsqueeze(2)).squeeze(2) / float(Ptot)  # (K,B)

            # --- Cache T = (Phi^T C_all)/P and diagonal of B ---
            def compute_T_and_diag(C_slice: torch.Tensor, chunk: int = 4096):
                Mloc = C_slice.shape[2]
                T = torch.empty((Kb, Bsize, Mloc), device=device, dtype=torch.float32)
                diag_B = torch.empty((Kb, Mloc), device=device, dtype=torch.float32)
                for m0 in range(0, Mloc, chunk):
                    m1 = min(m0 + chunk, Mloc)
                    Cc = C_slice[:, :, m0:m1]
                    T_block = torch.matmul(phi.transpose(1, 2).contiguous(), Cc) / float(Ptot)  # (K,B,mc)
                    T[:, :, m0:m1] = T_block
                    d_block = torch.einsum('kbm,kbm,kb->km', T_block, T_block, invD) / float(Bsize)
                    diag_B[:, m0:m1] = d_block
                return torch.nan_to_num(T), torch.nan_to_num(diag_B)

            T_all, diag_B_all = compute_T_and_diag(C_all)

            # --- Matvec for B_full using cached T ---
            def B_full(v_all):  # v_all: (K, M_all)
                t = torch.bmm(T_all, v_all.unsqueeze(2)).squeeze(2)           # (K,B)
                t = t * invD                                                  # (K,B)
                u = torch.bmm(T_all.transpose(1, 2).contiguous(), t.unsqueeze(2)).squeeze(2)  # (K,M_all)
                return u / float(Bsize)

            lam = saem.ridge_lambda
            Nf = float(mdl.N)

            def apply_A_full(v_all):
                return (1.0 + lam) * v_all + Nf * B_full(v_all)

            diag_A_full = (1.0 + lam) + Nf * diag_B_all

            # b_all = (1/B) * T^T * (J_S * invD)
            t_S = J_S * invD
            b_all = torch.bmm(T_all.transpose(1, 2).contiguous(), t_S.unsqueeze(2)).squeeze(2) / float(Bsize)  # (K,M_all)

            # Solve A_full m_all = N b_all
            rhs = Nf * b_all
            m_all_star, relres, iters = pcg_solve_operator(apply_A_full, rhs, diag_A_full, tol=PCG_TOL, max_iter=PCG_MAX_ITERS)
            if (not math.isfinite(relres)) or relres > FALLBACK_RELRES_MAX:
                print(f"[pcg full fallback] relres={relres:.2e} -> diag solution", flush=True)
                m_all_star = torch.nan_to_num(rhs / diag_A_full)

            # SAEM averaging
            a_t = saem.a0 / (it + saem.t0)
            m_new = (1 - saem.damping_m * a_t) * m_all + saem.damping_m * a_t * m_all_star
            if saem.project_mS:
                m_new[:, 0] = m_new[:, 0].clamp(0.0, 1.0)
            if saem.m_noise_clip and saem.m_noise_clip > 0 and m_new.shape[1] > 1:
                m_new[:, 1:] = m_new[:, 1:].clamp(-saem.m_noise_clip, saem.m_noise_clip)
            slot["m_all"] = torch.nan_to_num(m_new)

        # ------------------ Trajectories & logging ------------------
        with torch.no_grad():
            mS_all = torch.cat([slot["m_all"][:, 0].detach().float().cpu() for slot in per_dev], dim=0)
            noise_sq_all = torch.cat([
                (slot["m_all"][:, 1:] * slot["m_all"][:, 1:]).sum(dim=1).detach().float().cpu() if slot["m_all"].shape[1] > 1 else torch.zeros(slot["m_all"].shape[0])
                for slot in per_dev
            ], dim=0)
            ge_all = 0.5 * (1.0 - mS_all.numpy()) ** 2 + noise_sq_all.numpy()

            traj_mS.append(mS_all.tolist())
            traj_mnoise_rms.append(noise_sq_all.sqrt().tolist())
            traj_time_s.append(time.time() - t_start)

        if it % saem.print_every == 0 or it == 1:
            msg = {
                "iter": it,
                "kappa": kappa,
                "P": P,
                "blocks": sum(_get(BASE, "blocks", 1) for _ in per_dev),
                "m_S_mean": float(mS_all.mean().item()),
                "m_S_std": float(mS_all.std(unbiased=False).item()),
                "gen_err_mean": float(np.mean(ge_all)),
                "m_noise_rms_mean": float(noise_sq_all.sqrt().mean().item()),
                "pcg_relres": relres,
                "pcg_iters": iters,
                "time_s": round(traj_time_s[-1], 2),
            }
            print(json.dumps(msg), flush=True)

    # Snapshot & save
    with torch.no_grad():
        mS_all = torch.cat([slot["m_all"][:, 0].detach().float().cpu() for slot in per_dev], dim=0)
        noise_sq_all = torch.cat([
            (slot["m_all"][:, 1:] * slot["m_all"][:, 1:]).sum(dim=1).detach().float().cpu() if slot["m_all"].shape[1] > 1 else torch.zeros(slot["m_all"].shape[0])
            for slot in per_dev
        ], dim=0)
        ge_all = 0.5 * (1.0 - mS_all.numpy()) ** 2 + noise_sq_all.numpy()

    summary = {
        "kappa": kappa,
        "P": P,
        "mode": "quenched-finiteP-fullsystem",
        "blocks_total": int(mS_all.numel()),
        "mS_mean": float(mS_all.mean().item()),
        "mS_std": float(mS_all.std(unbiased=False).item()),
        "gen_err_mean": float(np.mean(ge_all)),
        "m_noise_rms_mean": float(noise_sq_all.sqrt().mean().item()),
        "BASE": BASE,
        "M_all": int(M_all),
    }

    trajectories = {
        "iters": int(saem.max_iters),
        "time_s": traj_time_s,
        "m_S_per_iter": traj_mS,
        "m_noise_rms_per_iter": traj_mnoise_rms,
    }

    result = {"summary": summary, "trajectories": trajectories}

    os.makedirs(SAVE_DIR, exist_ok=True)
    tag = run_tag if run_tag else f"kap{kappa:.3e}_P{P}_K{_get(BASE,'blocks',1)}_it{_get(BASE,'opt_steps',400)}_B{_get(BASE,'chains_per_device',2048)}"
    out_path = os.path.join(SAVE_DIR, f"quenched_full_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[saved] {out_path}", flush=True)
    return result

# --------------------------------------------------------------------------------------
# Grid runner
# --------------------------------------------------------------------------------------

def run_grid(kappa_list: List[float], P_list: List[int], devices: List[int], BASE: Dict, SAVE_DIR: str, run_tag_prefix: str = ""):
    for kappa in kappa_list:
        for P in P_list:
            tag = f"{run_tag_prefix}_kap{kappa:.3e}_P{P}_K{_get(BASE,'blocks',1)}"
            _ = saem_quenched_full(kappa=kappa, P=P, devices=devices, BASE=BASE, SAVE_DIR=SAVE_DIR, run_tag=tag)

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    set_seed(42)
    devices = check_gpu()

    BASE = dict(
        d=25, N=1024, k=4,
        σa=1.0, σw=1.0, γ=1.0,
        act="relu",
        opt_steps=3000,
        # SGLD
        chains_per_device=65536,
        mcmc_steps=400,
        mcmc_step_size=5e-3,   # conservative while verifying
        mcmc_step_decay=0.999,
        langevin_sqrt2=True,
        grad_clip=1e5,
        clamp_w=100.0,
        # SAEM / quenched runs
        a0=1.0, t0=150.0, damping_m=3.0,
        eps_D=1e-9, print_every=10,
        ridge_lambda=1e-3,
        blocks=1,
        resample_blocks_every=0,
        crossfit_shuffle_each_iter=False,
        # seeds
        teacher_seed=0, noise_seed=1, data_seed=0,
        m_init=0.5,
        project_mS=False,
        m_noise_clip=0.0,
        # PCG controls
        pcg_max_iters=1000,
        pcg_tol=1e-5,
        fallback_to_diag_if_bad=True,
        fallback_relres_max=1e-3,
        # modes
        card_track_max=2,
        sample_high_per_card=50,
        # E-step backreaction ramp
        backreact_ramp_iters=50,
        backreact_ramp_pow=1.0,
    )

    save_dir = "/home/goring/mean_field_langevin/MCMC_finiteP/results3/c2_h50_64k_2000_2e-3_2"
    os.makedirs(save_dir, exist_ok=True)
    run_tag_prefix = time.strftime("%Y%m%d_%H%M%S")

    kappa_list = [1e-3]
    P_list     = [10, 100, 250, 500, 700, 900, 1000, 2000,5000]

    run_grid(kappa_list=kappa_list, P_list=P_list, devices=devices, BASE=BASE, SAVE_DIR=save_dir, run_tag_prefix=run_tag_prefix)
