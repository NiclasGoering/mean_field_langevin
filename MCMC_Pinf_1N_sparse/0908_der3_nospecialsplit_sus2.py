import os, json, time, math, random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

# ----------------------------- Utils -----------------------------

def check_gpu():
    if not torch.cuda.is_available():
        print("CUDA not available. Running on CPU.")
        return []
    n = torch.cuda.device_count()
    info = []
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        cap  = torch.cuda.get_device_capability(i)
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

# ----------------------------- Core math -----------------------------

def make_parity_indices(d: int, k: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    S = torch.randperm(d, generator=g)[:k]
    return S.sort().values

def make_boolean_batch(R: int, d: int, device: torch.device, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.randint(low=0, high=2, size=(R, d), generator=g, device=device, dtype=torch.int8)
    X = X.to(torch.float32) * 2.0 - 1.0
    return X

def chi_S_of_X(X: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    return X[:, S.long()].prod(dim=1)

def activation(z: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "relu":
        return torch.relu(z)
    elif kind == "tanh":
        return torch.tanh(z)
    else:
        raise ValueError(f"Unknown activation: {kind}")

# ----------------------------- Models/Params -----------------------------

@dataclass
class ModelParams:
    d: int = 40
    N: int = 1024
    k: int = 4
    sigma_a: float = 1.0
    sigma_w: float = 1.0
    gamma: float = 1.0
    act: str = "relu"

@dataclass
class MCMCParams:
    R_inputs: int = 131072
    n_chains_per_device: int = 1024
    n_steps: int = 200
    step_size: float = 1e-3
    step_decay: float = 0.999
    langevin_sqrt2: bool = True
    grad_clip: float = 1e5
    clamp_w: float = 10.0

@dataclass
class SAEMParams:
    max_iters: int = 2000
    a0: float = 0.5
    t0: float = 100.0
    damping: float = 0.5
    eps_D: float = 1e-6
    eps_proj: float = 1e-3
    print_every: int = 50

# ----------------------------- k-block helpers -----------------------------

def comb(n, r):
    if r < 0 or r > n: return 0
    return math.comb(n, r)

def kblock_pair_multiplicity(d: int, k: int) -> torch.Tensor:
    """
    Ordered pair multiplicities per overlap t:
    M_t = C(d,k) * C(k,t) * C(d-k, k-t),  t=0..k.
    """
    M = comb(d, k)
    vec = [M * comb(k, t) * comb(d - k, k - t) for t in range(k + 1)]
    return torch.tensor(vec, dtype=torch.float32)

def make_random_k_masks(d: int, k: int, L: int, device: torch.device, seed: int):
    """
    Returns:
      idx_A: (L,k) long indices,
      A_mask: (L,d) 0/1 float mask on device
    """
    g = torch.Generator(device=device).manual_seed(seed)
    idx_rows = []
    for _ in range(L):
        idx = torch.randperm(d, generator=g, device=device)[:k]
        idx_rows.append(idx.sort().values)
    idx_A = torch.stack(idx_rows, dim=0).long()              # (L,k)
    A_mask = torch.zeros(L, d, device=device, dtype=torch.float32)
    A_mask.scatter_(1, idx_A, 1.0)
    return idx_A, A_mask

def overlap_matrix_from_masks(A_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute T = |A_i ∩ A_j|. Use float matmul (int matmul unsupported on CUDA).
    Returns int16 on the same device.
    """
    Af = A_mask.to(torch.float32)
    T  = Af @ Af.T
    return T.round().to(torch.int16)

def build_masks_from_T(T: torch.Tensor, k: int) -> List[torch.Tensor]:
    """Build float masks M_t = 1_{|A∩B| = t} as (L,L) float matrices."""
    masks = []
    for t in range(k + 1):
        masks.append((T == t).to(torch.float32))
    return masks

def precompute_chi_A_X(X: torch.Tensor, idx_A: torch.Tensor) -> torch.Tensor:
    """
    chi_A_X: (R, L) whose column l is chi_{A_l}(X) = prod_{i in A_l} X[:,i]
    """
    R, d = X.shape
    L, k = idx_A.shape
    flat = idx_A.reshape(-1)                      # (L*k,)
    X_sel = X[:, flat]                            # (R, L*k)
    chi_A_X = X_sel.view(R, L, k).prod(dim=2)     # (R,L)
    return chi_A_X

def make_Jk_eval(chi_A_X: torch.Tensor):
    """
    Returns a callable eval_Jvec(phi) -> (B,L) computing
    J_A(w) = E_x [ phi(w·x) * chi_A(x) ] for all A in the sampled block.
    """
    R, L = chi_A_X.shape
    def eval_Jvec(phi: torch.Tensor) -> torch.Tensor:
        # phi: (R,B)
        return (phi.transpose(0,1) @ chi_A_X) / float(R)  # (B,L)
    return eval_Jvec

def sums_by_overlap_fast(J: torch.Tensor, masks: List[torch.Tensor]) -> torch.Tensor:
    """
    J: (B,L), masks[t]: (L,L) float
    Returns S_t: (B, len(masks)) with S_t[:,t] = sum_{i,j:overlap=t} J_i J_j for each chain.
    """
    out = []
    for Mt in masks:
        U  = J @ Mt                    # (B,L)
        out.append((J * U).sum(dim=1)) # (B,)
    return torch.stack(out, dim=1)     # (B,T)

def sums_by_overlap_total_weighted(J: torch.Tensor, a2: torch.Tensor, masks: List[torch.Tensor]) -> torch.Tensor:
    """
    Weighted total sum across chains:
      S_num[t] = sum_b a2_b * sum_{i,j:overlap=t} J_bi J_bj
               = 1^T ( (J^T @ diag(a2) @ J) ⊙ M_t ) 1
               = sum of elementwise product with mask M_t.
    Compute via Cw = J^T @ (J * a2[:,None]).
    Returns S_num: (T,) on J.device
    """
    B, L = J.shape
    Cw = J.transpose(0,1) @ (J * a2.view(B,1))   # (L,L)
    out = []
    for Mt in masks:
        out.append((Cw * Mt).sum())
    return torch.stack(out)

# ----------------------------- Logp/Grad (k-block with SS spike) -----------------------------

def unbiased_Rk_offdiag_per_chain(
    J: torch.Tensor,                 # (B,L)
    masks_off: List[torch.Tensor],   # t=0..k-1
    f_k_off: torch.Tensor,           # (k,)  bins for t=0..k-1
    pair_counts_off: torch.Tensor,   # (k,)  sample counts per t
    pair_mult_full_off: torch.Tensor,# (k,)  full multiplicities per t
    N: int,
) -> torch.Tensor:
    """
    Off-diagonal reaction (overlaps t=0..k-1):
      R_off(b) = (1/N) * sum_t f_k_off[t] * ( M_full[t] / n_sample[t] ) * sum_{i,j in sample, overlap=t} J_bi J_bj
    """
    S_t = sums_by_overlap_fast(J, masks_off)                               # (B,k)
    scale = (pair_mult_full_off / pair_counts_off.clamp(min=1)).view(1,-1) # (1,k)
    contrib = S_t * scale                                                  # (B,k)
    R_off = (contrib * f_k_off.view(1,-1)).sum(dim=1) / float(N)          # (B,)
    return R_off

def compute_logp_and_grad_kblock(
    w: torch.Tensor, X: torch.Tensor, chi_S_vec: torch.Tensor,
    eval_Jvec,
    masks_off: List[torch.Tensor], pair_counts_off: torch.Tensor,
    f_k_off: torch.Tensor, pair_mult_full_off: torch.Tensor,
    chi_SS_scalar: float,                     # scalar spike
    m: float, kappa: float,
    mdl: ModelParams, saem: SAEMParams
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute logp, grad, J_S, Sigma, Rk using:
      Rk = R_offdiag + (chi_SS_scalar / N) * J_S^2
    """
    gw2 = mdl.sigma_w / mdl.d
    ga2 = mdl.sigma_a / (mdl.N ** mdl.gamma)
    A   = 1.0 / ga2

    w = w.detach().requires_grad_(True)

    # Core features
    z = X @ w.T                          # (R,B)
    phi = activation(z, mdl.act)         # (R,B)
    J_S = (phi * chi_S_vec[:, None]).mean(dim=0)  # (B,)
    Sigma = (phi * phi).mean(dim=0)               # (B,)

    # k-block projections
    J_vec = eval_Jvec(phi)                       # (B,L)

    # Reaction = off-diagonal (overlaps t<k) + SS spike
    R_off = unbiased_Rk_offdiag_per_chain(
        J=J_vec, masks_off=masks_off, f_k_off=f_k_off,
        pair_counts_off=pair_counts_off,
        pair_mult_full_off=pair_mult_full_off, N=mdl.N
    )                                             # (B,)
    R_spk = (chi_SS_scalar / float(mdl.N)) * (J_S * J_S)   # (B,)
    Rk = R_off + R_spk

    D = A*(kappa**2) + Sigma - Rk
    D_safe = torch.clamp(D, min=saem.eps_D)

    prior_term = -0.5 * (w * w).sum(dim=1) / gw2
    log_det_term = -0.5 * torch.log(D_safe)
    data_term = ((1.0 - m)**2) / (2.0 * (kappa**2)) * (J_S * J_S) / D_safe

    logp = prior_term + log_det_term + data_term

    grad = torch.autograd.grad(logp.sum(), w, create_graph=False, retain_graph=False)[0]
    if grad is None:
        grad = torch.zeros_like(w)
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    return logp.detach(), grad.detach(), J_S.detach(), Sigma.detach(), Rk.detach()

def mcmc_sgld_kblock(
    w: torch.Tensor, X: torch.Tensor, chi_S_vec: torch.Tensor,
    eval_Jvec,
    masks_off: List[torch.Tensor], pair_counts_off: torch.Tensor,
    f_k_off: torch.Tensor, pair_mult_full_off: torch.Tensor,
    chi_SS_scalar: float,
    m: float, kappa: float,
    mdl: ModelParams, saem: SAEMParams, mcmc: MCMCParams
) -> torch.Tensor:
    step = mcmc.step_size
    for _ in range(mcmc.n_steps):
        _, grad, _, _, _ = compute_logp_and_grad_kblock(
            w, X, chi_S_vec, eval_Jvec,
            masks_off, pair_counts_off, f_k_off, pair_mult_full_off,
            chi_SS_scalar, m, kappa, mdl, saem
        )
        if mcmc.grad_clip and mcmc.grad_clip > 0:
            gn = grad.norm(dim=1, keepdim=True).clamp_min(1e-12)
            grad = grad * (mcmc.grad_clip / gn).clamp(max=1.0)

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

# ----------------------------- SAEM (k-block with SS spike) -----------------------------

def clamp_m(m: float) -> float:
    if m < 0.0: return 0.0
    if m > 1.0: return 1.0
    return m

def saem_solve_for_kappa_kblock(
    kappa: float, devices: List[int],
    BASE: Dict, SAVE_DIR: str, run_tag: str = ""
):
    # Unpack params
    mdl = ModelParams(
        d=BASE["d"], N=BASE["N"], k=BASE["k"],
        sigma_a=BASE["σa"], sigma_w=BASE["σw"],
        gamma=BASE["γ"], act=BASE.get("act", "relu")
    )
    mcmc = MCMCParams(
        R_inputs=BASE.get("R_inputs", 131072),
        n_chains_per_device=BASE.get("chains_per_device", 1024),
        n_steps=BASE.get("mcmc_steps", 200),
        step_size=BASE.get("mcmc_step_size", 1e-3),
        step_decay=BASE.get("mcmc_step_decay", 0.999),
        langevin_sqrt2=BASE.get("langevin_sqrt2", True),
        grad_clip=BASE.get("grad_clip", 1e5),
        clamp_w=BASE.get("clamp_w", 10.0),
    )
    saem = SAEMParams(
        max_iters=BASE.get("opt_steps", 2000),
        a0=BASE.get("a0", 0.5),
        t0=BASE.get("t0", 100.0),
        damping=BASE.get("damping", 0.5),
        eps_D=BASE.get("eps_D", 1e-6),
        eps_proj=BASE.get("eps_proj", 1e-3),
        print_every=BASE.get("print_every", 50),
    )

    # Teacher subset S (CPU)
    S = make_parity_indices(mdl.d, mdl.k, seed=BASE.get("teacher_seed", 0))

    # Per-device state
    per_dev = []
    total_chains = 0
    L = BASE.get("kblock_L", 512)
    kblock_seed = BASE.get("kblock_seed", 123)

    for dev_id in devices if len(devices) > 0 else [-1]:
        device = torch.device(f"cuda:{dev_id}") if dev_id >= 0 and torch.cuda.is_available() else torch.device("cpu")
        X = make_boolean_batch(mcmc.R_inputs, mdl.d, device, seed=BASE.get("data_seed", 0))
        S_dev = S.to(device, non_blocking=True).long()
        chi_vec_S = chi_S_of_X(X, S_dev)

        chains = mcmc.n_chains_per_device
        gw2 = BASE["σw"] / BASE["d"]
        w = torch.randn(chains, mdl.d, device=device) * math.sqrt(gw2)

        # k-block sampling on this device
        idx_A, A_mask = make_random_k_masks(mdl.d, mdl.k, L, device, seed=kblock_seed)
        T = overlap_matrix_from_masks(A_mask)                    # (L,L) int16
        masks_all = build_masks_from_T(T, mdl.k)                 # list of (L,L) float
        # Split masks: off-diagonal bins t=0..k-1, diagonal t=k is excluded here
        masks_off = masks_all[:-1]
        pair_counts_all = torch.stack([(T == t).sum() for t in range(mdl.k + 1)], dim=0).to(torch.float32)
        pair_counts_off = pair_counts_all[:-1]
        chi_A_X = precompute_chi_A_X(X, idx_A)                   # (R,L)
        eval_Jvec = make_Jk_eval(chi_A_X)
        pair_mult_full_all = kblock_pair_multiplicity(mdl.d, mdl.k).to(device)
        pair_mult_full_off = pair_mult_full_all[:-1]

        per_dev.append(dict(
            device=device, X=X, S=S_dev, chi_vec_S=chi_vec_S, w=w,
            idx_A=idx_A, A_mask=A_mask, T=T,
            masks_off=masks_off,
            pair_counts_off=pair_counts_off,
            chi_A_X=chi_A_X, eval_Jvec=eval_Jvec,
            pair_mult_full_off=pair_mult_full_off
        ))
        total_chains += chains

    # SAEM state
    m = BASE.get("m_init", 0.0)
    m_bar = m

    # f_k_off (k entries for t=0..k-1). Init tiny positive.
    f_k_off = torch.full((mdl.k,), 1e-6, dtype=torch.float32)
    # χ_SS spike scalar
    chi_SS_scalar = 0.0

    history = []
    t_start = time.time()

    for it in range(1, saem.max_iters + 1):
        g1_sum = 0.0
        g2_sum = 0.0
        n_sum  = 0

        # accumulators for f_k_off and χ_SS
        f_num_off_global = torch.zeros(mdl.k, dtype=torch.float64)   # t=0..k-1
        f_den_off_global = torch.zeros(mdl.k, dtype=torch.float64)
        ss_num_global    = 0.0
        ss_den_global    = 0

        for slot in per_dev:
            device = slot["device"]
            X = slot["X"]; chi_vec_S = slot["chi_vec_S"]; w = slot["w"]
            masks_off = slot["masks_off"]
            pair_counts_off = slot["pair_counts_off"]
            eval_Jvec = slot["eval_Jvec"]
            pair_mult_full_off = slot["pair_mult_full_off"]

            # move f_k_off to device
            f_k_off_dev = f_k_off.to(device)

            # --- SGLD refresh with current reaction ---
            w = mcmc_sgld_kblock(
                w=w, X=X, chi_S_vec=chi_vec_S,
                eval_Jvec=eval_Jvec,
                masks_off=masks_off, pair_counts_off=pair_counts_off,
                f_k_off=f_k_off_dev, pair_mult_full_off=pair_mult_full_off,
                chi_SS_scalar=chi_SS_scalar,
                m=m, kappa=kappa, mdl=mdl, saem=saem, mcmc=mcmc
            )
            slot["w"] = w

            # --- Stats using same reaction ---
            logp, _, J_S, Sigma, Rk = compute_logp_and_grad_kblock(
                w, X, chi_vec_S, eval_Jvec,
                masks_off, pair_counts_off, f_k_off_dev,
                pair_mult_full_off,
                chi_SS_scalar, m, kappa, mdl, saem
            )

            ga2 = mdl.sigma_a / (mdl.N ** mdl.gamma)
            A = 1.0 / ga2
            D = A*(kappa**2) + Sigma - Rk
            D = torch.clamp(D, min=saem.eps_D)

            g1 = (J_S * J_S) / D
            g2 = ((J_S * J_S) * (J_S * J_S)) / ((kappa**2) * (D * D))

            g1_sum += g1.sum().item()
            g2_sum += g2.sum().item()
            n_sum  += g1.numel()

            # a^2(w) = κ^2/D + ((1-m)^2) * J_S^2 / D^2
            a2 = (kappa**2) / D + ((1.0 - m)**2) * (J_S * J_S) / (D * D)  # (B,)

            # k-block projections for f_k_off and χ_SS
            phi = activation((X @ w.T), mdl.act)
            J_vec = eval_Jvec(phi)  # (B,L)

            # Off-diagonal bins (t=0..k-1)
            S_num_off = sums_by_overlap_total_weighted(J_vec, a2, masks_off).to('cpu')  # (k,)
            denom_off = (J_vec.shape[0]) * slot["pair_counts_off"].to('cpu')            # (k,)

            f_num_off_global += S_num_off.to(torch.float64)
            f_den_off_global += denom_off.to(torch.float64)

            # χ_SS spike
            ss_num_global += float((a2 * (J_S * J_S)).sum().item())
            ss_den_global += int(J_S.shape[0])

        # finalize pooled averages
        g1_mean = g1_sum / max(1, n_sum)
        g2_mean = g2_sum / max(1, n_sum)

        # Robbins–Monro step size
        a_t = saem.a0 / (it + saem.t0)

        # m update:  F1 = m - N(1-m) g1
        F1 = m - mdl.N * (1.0 - m) * g1_mean
        m_new = m - saem.damping * a_t * F1
        m = clamp_m(m_new)

        # f_k_off update
        f_avg_off = torch.zeros_like(f_k_off, dtype=torch.float64)
        nz = f_den_off_global > 0
        f_avg_off[nz] = f_num_off_global[nz] / f_den_off_global[nz]      # ⟨a^2 J_A J_B⟩ per off-diag bin
        f_new_off = (mdl.N / (kappa**2)) * f_avg_off.to(torch.float32)   # χ^{(k)}_off bins
        f_k_off = (1.0 - a_t) * f_k_off + a_t * torch.nan_to_num(f_new_off, nan=0.0, posinf=0.0, neginf=0.0)
        f_k_off = torch.clamp(f_k_off, min=0.0)

        # χ_SS spike update
        if ss_den_global > 0:
            chi_SS_new = (mdl.N / (kappa**2)) * (ss_num_global / ss_den_global)
        else:
            chi_SS_new = chi_SS_scalar
        chi_SS_scalar = (1.0 - a_t) * chi_SS_scalar + a_t * float(chi_SS_new)
        chi_SS_scalar = max(0.0, chi_SS_scalar)

        if it % saem.print_every == 0 or it == 1:
            dt = time.time() - t_start
            msg = {
                "iter": it, "kappa": kappa,
                "m": float(m), "m_bar": float(((it-1)*m_bar + m)/it),
                "g1_mean": g1_mean, "g2_mean": g2_mean,
                "f_k_off": [float(x) for x in f_k_off.to('cpu')],       # bins t=0..k-1
                "chi_SS": chi_SS_scalar,
                "time_s": round(dt, 2)
            }
            print(json.dumps(msg))

    # Final snapshot
    result = {
        "kappa": kappa,
        "m_final": float(m),
        "f_k_off": [float(x) for x in f_k_off.to('cpu')],
        "chi_SS": chi_SS_scalar,
        "BASE": BASE
    }

    os.makedirs(SAVE_DIR, exist_ok=True)
    tag = run_tag if run_tag else f"kappa_{kappa:.6f}"
    out_path = os.path.join(SAVE_DIR, f"kblock_offdiag_plus_SS_result_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[saved] {out_path}")
    return result

# ----------------------------- Main -----------------------------

if __name__ == "__main__":
    set_seed(42)
    devices = check_gpu()

    BASE = dict(
        d=40, N=1024, k=4,
        σa=1.0, σw=1.0, γ=1.0,
        act="relu",
        # SAEM
        opt_steps=2000,
        a0=0.5, t0=100.0, damping=0.5,
        eps_D=1e-6, eps_proj=1e-3, print_every=50,
        # MCMC
        R_inputs=131072,
        chains_per_device=1024,
        mcmc_steps=200,
        mcmc_step_size=1e-3,
        mcmc_step_decay=0.999,
        langevin_sqrt2=True,
        grad_clip=1e5,
        clamp_w=10.0,
        # k-block
        kblock_L=512,
        kblock_seed=123,
        # Seeds
        teacher_seed=0, data_seed=0,
        # init
        m_init=0.6,
    )

    kappa_list = np.logspace(np.log10(1e-4), np.log10(5e1), 14)
    save_dir = "/home/goring/mean_field_langevin/MCMC_Pinf_1N_sparse/results/1908_d40k4_kblock2"
    os.makedirs(save_dir, exist_ok=True)

    run_tag_prefix = time.strftime("%Y%m%d_%H%M%S")
    m_ws = BASE.get("m_init", 0.0)

    for idx, kappa in enumerate(kappa_list):
        print(f"\n=== SAEM (k-block offdiag + SS spike) for kappa={kappa} ===")
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
