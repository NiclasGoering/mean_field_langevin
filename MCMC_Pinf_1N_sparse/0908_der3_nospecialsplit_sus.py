import os, json, time, math, random
from typing import Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
import torch

# ================================================================
# Utils
# ================================================================

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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ================================================================
# Core math (data + activations)
# ================================================================

def make_parity_indices(d: int, k: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    S = torch.randperm(d, generator=g)[:k]
    return S.sort().values

def make_boolean_batch(R: int, d: int, device: torch.device, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.randint(low=0, high=2, size=(R, d), generator=g, device=device, dtype=torch.int8)
    X = X.to(torch.float32) * 2.0 - 1.0
    return X

def chi_set_of_X(X: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    # idx: (..., k) int64; returns (R, ...) parity per A
    return X[:, idx.long()].prod(dim=-1)

def chi_S_of_X(X: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    return X[:, S.long()].prod(dim=1)

def activation(z: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "relu":
        return torch.relu(z)
    elif kind == "tanh":
        return torch.tanh(z)
    else:
        raise ValueError(f"Unknown activation: {kind}")

# ================================================================
# Model / algo params
# ================================================================

@dataclass
class ModelParams:
    d: int = 30
    N: int = 1024
    k: int = 4
    sigma_a: float = 1.0     # σ_a
    sigma_w: float = 1.0     # σ_w
    gamma: float = 1.0
    act: str = "relu"

@dataclass
class MCMCParams:
    R_inputs: int = 8192
    n_chains_per_device: int = 8192
    n_steps: int = 30
    step_size: float = 5e-3
    step_decay: float = 0.999
    langevin_sqrt2: bool = True
    grad_clip: float = 10.0
    clamp_w: float = 10.0

@dataclass
class SAEMParams:
    max_iters: int = 4000
    a0: float = 0.5
    t0: float = 100.0
    damping: float = 1.0
    eps_D: float = 1e-6
    print_every: int = 50

# ================================================================
# J_S, Σ via φ; block-J via product approx with calibration
# ================================================================

def compute_J_Sigma_exact(
    w: torch.Tensor, X: torch.Tensor, chi_S_vec: torch.Tensor, act: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact (Monte-Carlo over X) J_S and Sigma for each chain.
    w: (B,d) ; X: (R,d); chi_S_vec: (R,)
    returns:
      J_S: (B,), Sigma: (B,)
    """
    z = X @ w.T                # (R,B)
    phi = activation(z, act)   # (R,B)
    J_S = (phi * chi_S_vec[:, None]).mean(dim=0)     # (B,)
    Sigma = (phi * phi).mean(dim=0)                  # (B,)
    return J_S, Sigma

def sample_k_subsets(d: int, k: int, L: int, device, seed: int) -> torch.Tensor:
    """
    Return L subsets (each length-k sorted) as int64 tensor (L,k).
    Sampling with replacement, uniform over all C(d,k) subsets.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    # sample by shuffling [0..d-1] per subset and taking first k
    # (cheap, close to uniform for d=40,k=4)
    perm = torch.stack([torch.randperm(d, generator=g, device=device) for _ in range(L)], dim=0)
    idx = perm[:, :k]
    idx, _ = torch.sort(idx, dim=1)
    return idx.long()

def overlap_matrix_from_masks(A_mask: torch.Tensor) -> torch.Tensor:
    """
    A_mask: (L,d) {0,1}
    returns T: (L,L) with overlaps |A_i ∩ A_j|.
    Use float matmul on CUDA; integer matmul is not implemented.
    """
    Af = A_mask.to(torch.float32)
    T = Af @ Af.T                      # (L,L) float32, exact small integers
    return T.round().to(torch.int16)   # keep overlaps as ints if you like




def build_overlap_masks(T: torch.Tensor, k: int) -> List[torch.Tensor]:
    """
    T: (L,L) overlaps
    returns list of boolean masks M_t for t=0..k ; includes diagonal (t=k)
    """
    Ms = []
    for t in range(k+1):
        Ms.append((T == t))
    return Ms


def prod_over_indices(w: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    w: (B, d), idx: (L, k) on the SAME device as w
    returns P: (B, L) with ∏_{i in A_l} w[:, i]
    """
    idx = idx.to(w.device, non_blocking=True).long()  # <-- ensure same device & dtype
    B, d = w.shape
    L, k = idx.shape

    # gather -> (B, L, k)
    idx_exp = idx.view(1, L, k).expand(B, L, k)
    w_exp = w.unsqueeze(1).expand(B, L, d)
    gathered = torch.gather(w_exp, 2, idx_exp)  # (B, L, k)  (safe: all CUDA)
    P = gathered.prod(dim=2)                    # (B, L)
    return P


def calibrate_beta(J_S_exact: torch.Tensor, w: torch.Tensor, S: torch.Tensor) -> float:
    """
    Fit scalar beta to minimize ||beta * prod(w_S) - J_S_exact||^2 across chains.
    """
    S_dev = S.to(w.device, non_blocking=True).long()
    prod_S = prod_over_indices(w, S_dev.view(1, -1)).squeeze(1)  # (B,)

    num = (J_S_exact * prod_S).sum().item()
    den = (prod_S * prod_S).sum().item() + 1e-12
    beta = num / den
    return float(beta)

# ================================================================
# logπ and grad (no TAP in gradient; TAP only in stats via D_block)
# ================================================================

def compute_logp_and_grad_noreact(
    w: torch.Tensor, X: torch.Tensor, chi_S_vec: torch.Tensor,
    m: float, kappa: float,
    mdl: ModelParams, saem: SAEMParams, mcmc: MCMCParams
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Gibbs density for w after integrating out a, but ignoring reaction in the gradient:
      g_w^2 = sigma_w / d
      g_a^2 = sigma_a / N^gamma
      A     = 1 / g_a^2  = N^gamma / sigma_a
      D0    = A*kappa^2 + Sigma
      logp  = -||w||^2/(2 g_w^2) - 0.5*log D0 + ((1-m)^2/(2 kappa^2)) * (J_S^2 / D0)
    """
    gw2 = mdl.sigma_w / mdl.d
    ga2 = mdl.sigma_a / (mdl.N ** mdl.gamma)
    A   = 1.0 / ga2

    w = w.detach().requires_grad_(True)

    # exact J_S and Sigma for gradient
    z = X @ w.T
    phi = activation(z, mdl.act)
    J_S = (phi * chi_S_vec[:, None]).mean(dim=0)     # (B,)
    Sigma = (phi * phi).mean(dim=0)                  # (B,)

    D0 = A * (kappa**2) + Sigma
    D0_safe = torch.clamp(D0, min=saem.eps_D)

    # log prior: - ||w||^2 / (2 g_w^2)
    prior_term = -0.5 * (w * w).sum(dim=1) / gw2

    # terms from integrating out a (no reaction inside gradient)
    log_det_term = -0.5 * torch.log(D0_safe)
    data_term    = ((1.0 - m)**2) / (2.0 * (kappa**2)) * (J_S * J_S) / D0_safe

    logp = prior_term + log_det_term + data_term
    grad = torch.autograd.grad(logp.sum(), w, create_graph=False, retain_graph=False)[0]

    if mcmc.grad_clip and mcmc.grad_clip > 0:
        gn = grad.norm(dim=1, keepdim=True).clamp_min(1e-12)
        grad = grad * (mcmc.grad_clip / gn).clamp(max=1.0)
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    return logp.detach(), grad.detach(), J_S.detach(), Sigma.detach()

# ================================================================
# SGLD sampler (no reaction in gradient)
# ================================================================

def mcmc_sgld(
    w: torch.Tensor, X: torch.Tensor, chi_S_vec: torch.Tensor,
    m: float, kappa: float,
    mdl: ModelParams, saem: SAEMParams, mcmc: MCMCParams
) -> torch.Tensor:
    step = mcmc.step_size
    for _ in range(mcmc.n_steps):
        _, grad, _, _ = compute_logp_and_grad_noreact(
            w, X, chi_S_vec, m, kappa, mdl, saem, mcmc
        )
        if mcmc.langevin_sqrt2:
            noise = torch.randn_like(w) * math.sqrt(2.0 * step)
            w = w + step * grad + noise
        else:
            noise = torch.randn_like(w) * math.sqrt(step)
            w = w + 0.5 * step * grad + noise

        if mcmc.clamp_w:
            w = torch.clamp(w, min=-mcmc.clamp_w, max=mcmc.clamp_w)
        step *= mcmc.step_decay
    return w.detach()

# ================================================================
# k-block stats: R_k(w), g1,g2 with block denominator, and f_k(t) update
# ================================================================

def kblock_stats_and_update(
    w: torch.Tensor, X: torch.Tensor, S: torch.Tensor, act: str,
    f_k: torch.Tensor,  # shape (k+1,)
    L: int,
    mdl: ModelParams, saem: SAEMParams,
    rng_seed_pairs: int
):
    """
    Compute:
      - Exact J_S, Sigma
      - Calibrate beta so product-approx matches J_S
      - Sample L k-subsets, build overlap masks
      - Approximate J_k via beta * prod(w_A)
      - R_k(w) for each chain using f_k
      - D_block = A kappa^2 + Sigma - R_k(w)
      - g1, g2 (using block D) and a2 for χ^(k) update
      - Update f_k by Robbins–Monro
    Returns tuple with per-chain tensors and per-iteration moments:
      J_S (B,), Sigma (B,), D_block (B,), a2 (B,),
      f_k_new (k+1,), overlap_pair_counts (k+1,), beta (float),
      helper dict with sample info (for debugging/logging)
    """
    device = w.device
    S = S.to(w.device, non_blocking=True).long()

    B, d = w.shape
    k = S.numel()
    # exact J_S, Sigma
    chi_S_vec = chi_S_of_X(X, S)  # (R,)
    J_S, Sigma = compute_J_Sigma_exact(w, X, chi_S_vec, act)

    # combinatorics
    M = math.comb(d, k)

    # sample A's and overlaps
    A_idx = sample_k_subsets(d=d, k=k, L=L, device=device, seed=rng_seed_pairs)
    A_mask = torch.zeros((L, d), dtype=torch.int8, device=device)
    A_mask.scatter_(1, A_idx, 1)
    T = overlap_matrix_from_masks(A_mask)                   # (L,L), entries in {0..k}
    Ms = build_overlap_masks(T, k)                          # list of (L,L) boolean
    pair_counts = torch.tensor([m.sum().item() for m in Ms], device=device, dtype=torch.float32)  # ordered pairs incl. diag

    # calibrate beta using the teacher subset S
    beta = calibrate_beta(J_S_exact=J_S, w=w, S=S)

    # approximate J_k(w) for sampled subsets via product*beta
    J_prod = prod_over_indices(w, A_idx)                    # (B,L)
    Jk = beta * J_prod                                      # (B,L)

    # weighted overlap matrix for current f_k: M_w = sum_t f_k[t] * M_t
    # we will use ordered pairs, and later scale by (M/L)^2 to approximate the full sum
    M_w = torch.zeros_like(T, dtype=torch.float32)
    for t in range(k+1):
        if f_k[t].item() != 0.0:
            M_w = M_w + f_k[t].float() * Ms[t].float()

    # compute s_vec = J^T M_w J for each chain: s = diag(Jk @ M_w @ Jk^T)
    # do it as V = Jk @ M_w ; s = (V * Jk).sum(1)
    V = Jk @ M_w.to(Jk.dtype)                   # (B,L)
    s_vec = (V * Jk).sum(dim=1)                 # (B,)

    # scale from sample (L^2 pairs) to full block (M^2 ordered pairs)
    scale_pairs = (M / L)**2
    Rk_vec = (scale_pairs / mdl.N) * s_vec      # (B,)

    # block denominator
    Acoef = (mdl.N ** mdl.gamma) / mdl.sigma_a  # A = N^gamma / sigma_a
    D_block = Acoef * (saem_kappa**2) + Sigma - Rk_vec
    D_block = torch.clamp(D_block, min=saem.eps_D)

    # gains for m update (same formulas, but with D_block)
    g1_chain = (J_S * J_S) / D_block
    g2_chain = (J_S**4) / ((saem_kappa**2) * (D_block * D_block))

    # a^2 conditional expectation used for χ^(k) update
    # Var(a|w) = κ^2 / D_block ; Mean(a|w) = (1-m) J_S / D_block
    a2_chain = (saem_kappa**2) / D_block + ((1.0 - saem_m)**2) * (J_S * J_S) / (D_block * D_block)

    # per-overlap average of J_A J_B on the L-sample, for each chain
    # avg_t = (J^T M_t J) / (#pairs_t_sample)
    f_k_new = torch.zeros_like(f_k, dtype=torch.float32)
    for t in range(k+1):
        if pair_counts[t] < 1:
            continue
        Mt = Ms[t].float()
        Vt = Jk @ Mt            # (B,L)
        s_t = (Vt * Jk).sum(dim=1)  # (B,)
        avg_t = s_t / pair_counts[t]
        # χ^(k) entry for overlap t: f_k(t) = (N/κ^2) ⟨ a^2 * E_{pairs:t}[J_A J_B] ⟩
        f_k_new[t] = (mdl.N / (saem_kappa**2)) * (a2_chain * avg_t).mean()

    # Clamp tiny negatives (numerical noise) and avoid explosion
    f_k_new = torch.nan_to_num(f_k_new, nan=0.0, posinf=0.0, neginf=0.0)
    f_k_new = torch.clamp(f_k_new, min=0.0)

    helpers = dict(
        L=L, M=M, beta=beta, pair_counts=[float(pc.item()) for pc in pair_counts],
        f_k_current=[float(x.item()) for x in f_k], f_k_new=[float(x.item()) for x in f_k_new]
    )

    return J_S, Sigma, D_block, a2_chain, g1_chain, g2_chain, f_k_new, helpers

# ================================================================
# SAEM loop with k-block susceptibility
# ================================================================

def saem_solve_for_kappa_kblock(
    kappa: float, devices: List[int],
    BASE: Dict, SAVE_DIR: str, run_tag: str = ""
):
    global saem_kappa, saem_m  # used in kblock_stats_and_update
    saem_kappa = float(kappa)

    # Unpack params
    mdl = ModelParams(
        d=BASE["d"], N=BASE["N"], k=BASE["k"],
        sigma_a=BASE["σa"], sigma_w=BASE["σw"],
        gamma=BASE["γ"], act=BASE.get("act", "relu")
    )
    mcmc = MCMCParams(
        R_inputs=BASE.get("R_inputs", 8192),
        n_chains_per_device=BASE.get("chains_per_device", 8192),
        n_steps=BASE.get("mcmc_steps", 30),
        step_size=BASE.get("mcmc_step_size", 5e-3),
        step_decay=BASE.get("mcmc_step_decay", 0.999),
        langevin_sqrt2=BASE.get("langevin_sqrt2", True),
        grad_clip=BASE.get("grad_clip", 10.0),
        clamp_w=BASE.get("clamp_w", 10.0),
    )
    saem = SAEMParams(
        max_iters=BASE.get("opt_steps", 4000),
        a0=BASE.get("a0", 0.5),
        t0=BASE.get("t0", 100.0),
        damping=BASE.get("damping", 1.0),
        eps_D=BASE.get("eps_D", 1e-6),
        print_every=BASE.get("print_every", 50),
    )
    L = BASE.get("kblock_L", 256)         # number of sampled k-subsets per iter for k-block
    seed_A = BASE.get("kblock_seed", 123)

    # Teacher indices
    S = make_parity_indices(mdl.d, mdl.k, seed=BASE.get("teacher_seed", 0))

    # Per-device state
    per_dev = []
    total_chains = 0
    for di in devices if len(devices) > 0 else [-1]:
        device = torch.device(f"cuda:{di}") if di >= 0 and torch.cuda.is_available() else torch.device("cpu")
        X = make_boolean_batch(mcmc.R_inputs, mdl.d, device, seed=BASE.get("data_seed", 0))
        S_dev = S.to(device, non_blocking=True).long()                 # <-- move S to device
        chi_vec_S = chi_S_of_X(X, S_dev)   
        chains = mcmc.n_chains_per_device

        # init w with variance g_w^2 = σ_w/d
        gw2 = BASE["σw"] / BASE["d"]
        w = torch.randn(chains, mdl.d, device=device) * math.sqrt(gw2)
        per_dev.append({"device": device, "X": X, "chi_vec_S": chi_vec_S, "w": w, "S": S_dev})


        total_chains += chains

    # Init order parameters
    m = BASE.get("m_init", 0.0)
    saem_m = float(m)

    # k-block compressed susceptibility values f_k(t), t=0..k (initialize to zeros)
    f_k = torch.zeros(mdl.k + 1, dtype=torch.float32)

    # PR averages (optional)
    m_bar = m
    history = []
    t_start = time.time()

    for it in range(1, saem.max_iters + 1):
        # Robbins–Monro step size
        a_t = saem.a0 / (it + saem.t0)

        # accumulators (pooled over all chains / devices)
        g1_sum = 0.0
        g2_sum = 0.0
        n_sum  = 0

        # f_k_new accumulator (simple average across devices)
        f_k_new_accum = torch.zeros_like(f_k)
        helper_any = None

        # E-step: refresh chains (SGLD with no reaction in gradient)
        for slot_id, slot in enumerate(per_dev):
            device = slot["device"]
            X, chi_vec_S, w = slot["X"], slot["chi_vec_S"], slot["w"]

            # SGLD (no reaction in gradient)
            w = mcmc_sgld(w, X, chi_vec_S, m, kappa, mdl, saem, mcmc)
            slot["w"] = w

            # k-block stats + update proposal for f_k
            J_S, Sigma, D_block, a2_chain, g1_chain, g2_chain, f_k_new_dev, helpers = \
    kblock_stats_and_update(
        w=w, X=X, S=slot["S"], act=mdl.act, f_k=f_k.clone().to(w.device),
        L=L, mdl=mdl, saem=saem, rng_seed_pairs=seed_A + 7919 * it + 104729 * slot_id
    )

            g1_sum += g1_chain.sum().item()
            g2_sum += g2_chain.sum().item()
            n_sum  += g1_chain.numel()

            f_k_new_accum += f_k_new_dev.cpu()

            if helper_any is None:
                helper_any = helpers

        # finalize pooled averages
        g1_mean = g1_sum / max(1, n_sum)
        g2_mean = g2_sum / max(1, n_sum)

        # M-step: m update via fixed-point residual (with block-corrected D)
        F1 = m - mdl.N * (1.0 - m) * g1_mean
        m_new = m - saem.damping * a_t * F1
        m = float(min(max(m_new, 0.0), 1.0))  # clamp to [0,1]
        saem_m = m  # update global for next iter

        # M-step: f_k update (Robbins–Monro smoothing)
        f_k_est = f_k_new_accum / max(1, len(per_dev))
        f_k = (1.0 - a_t) * f_k + a_t * f_k_est
        f_k = torch.nan_to_num(f_k, nan=0.0, posinf=0.0, neginf=0.0)
        f_k = torch.clamp(f_k, min=0.0)  # keep PSD-ish

        # Polyak–Ruppert average for m (optional)
        m_bar = ((it - 1) * m_bar + m) / it

        if it % saem.print_every == 0 or it == 1:
            dt = time.time() - t_start
            msg = {
                "iter": it, "kappa": kappa,
                "m": m, "m_bar": m_bar,
                "g1_mean": g1_mean, "g2_mean": g2_mean,
                "f_k": [float(x.item()) for x in f_k],
                "pair_counts": helper_any["pair_counts"] if helper_any else None,
                "beta": helper_any["beta"] if helper_any else None,
                "time_s": round(dt, 2)
            }
            print(json.dumps(msg))
            history.append(msg)

    # Final snapshot
    result = {
        "kappa": kappa,
        "m_final": m,
        "m_bar": m_bar,
        "f_k": [float(x.item()) for x in f_k],
        "history": history[-10:],  # last few for quick glance
        "BASE": BASE
    }

    os.makedirs(SAVE_DIR, exist_ok=True)
    tag = run_tag if run_tag else f"kappa_{kappa:.6f}"
    out_path = os.path.join(SAVE_DIR, f"saem_kblock_result_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[saved] {out_path}")
    return result

# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    set_seed(42)
    devices = check_gpu()

    BASE = dict(
        d=40, N=1024, k=4,
        σa=1.0, σw=1.0, γ=1.0,
        act="relu",
        opt_steps=4000,                # you can raise if needed
        # MCMC controls
        R_inputs=131072,
        chains_per_device=8192,
        mcmc_steps=120,
        mcmc_step_size=1e-3,
        mcmc_step_decay=0.999,
        langevin_sqrt2=True,
        grad_clip=1e5,
        clamp_w=10.0,
        # SAEM controls
        a0=0.5, t0=100.0, damping=1.0,
        eps_D=1e-6, print_every=50,
        # Seeds
        teacher_seed=0, data_seed=0,
        kblock_seed=123,
        # k-block MC sample size
        kblock_L=2048,
        # init
        m_init=0.59,
    )

    # Warm start from large -> small kappa
    kappa_list = np.logspace(np.log10(1e-4), np.log10(5e1), 14)

    # Choose a save directory
    save_dir = "/home/goring/mean_field_langevin/MCMC_Pinf_1N_sparse/results/1908_d40k4_kblock"
    os.makedirs(save_dir, exist_ok=True)

    run_tag_prefix = time.strftime("%Y%m%d_%H%M%S")

    # Warm start across kappas
    m_ws  = BASE.get("m_init", 0.0)

    for idx, kappa in enumerate(kappa_list):
        print(f"\n=== SAEM (k-block) for kappa={kappa} ===")
        BASE["m_init"] = m_ws
        tag = f"{run_tag_prefix}_k{idx}_{kappa:.6f}"
        result = saem_solve_for_kappa_kblock(
            kappa=kappa,
            devices=devices,
            BASE=BASE,
            SAVE_DIR=save_dir,
            run_tag=tag
        )
        m_ws = result["m_final"]
