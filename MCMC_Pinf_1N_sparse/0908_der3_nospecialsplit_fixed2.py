import os, json, time, math, random
from typing import Dict, List, Tuple
from dataclasses import dataclass

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

# ----------------------------- Core math -----------------------------

def make_parity_indices(d: int, k: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    S = torch.randperm(d, generator=g)[:k]
    return S.sort().values

def activation(z: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "relu":
        return torch.relu(z)
    elif kind == "tanh":
        return torch.tanh(z)
    else:
        raise ValueError(f"Unknown activation: {kind}")

def all_parity_patterns(k: int, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    if k <= 0:
        return torch.ones(1, 0, device=device, dtype=dtype)
    base = torch.tensor([-1.0, 1.0], device=device, dtype=dtype)
    grids = torch.meshgrid(*([base] * k), indexing="ij")
    pat = torch.stack(grids, dim=-1).reshape(-1, k)
    return pat  # (2^k, k)

# ----------------------------- Params -----------------------------

@dataclass
class ModelParams:
    d: int = 30
    N: int = 1024
    k: int = 4
    sigma_a: float = 1.0     # σ_a
    sigma_w: float = 1.0     # σ_w
    gamma: float = 1.0
    act: str = "relu"
    estimate_sigma: bool = False  # if True, estimate Σ even for ReLU

@dataclass
class MCMCParams:
    n_chains_per_device: int = 8192
    n_steps: int = 120
    step_size: float = 1e-3
    step_decay: float = 0.999
    langevin_sqrt2: bool = True
    grad_clip: float = 10.0
    clamp_w: float = 10.0

@dataclass
class SAEMParams:
    max_iters: int = 1000
    a0: float = 0.5
    t0: float = 100.0
    damping: float = 1.0
    eps_D: float = 1e-6
    eps_proj: float = 1e-3
    print_every: int = 5

@dataclass
class EstimatorChunking:
    ctx_chunk: int = 65536        # contexts generated per "pass"
    ctx_passes: int = 16          # passes per E-step
    ctx_inner: int = 4096         # micro-batch over contexts (controls peak memory)
    pat_chunk: int = 16           # parity patterns per micro-batch (<= 2^k)
    batch_chunk: int = 4096       # chains per micro-batch

# ----------------------------- k-exact J, Σ and gradients (streaming, no autograd graph) -----------------------------

@torch.no_grad()
@torch.no_grad()
def compute_J_Sigma_grad_kexact(
    w: torch.Tensor,                  # (B, d)
    S: torch.Tensor,                  # (k,)
    device: torch.device,
    mdl: ModelParams,
    chunks: EstimatorChunking,
    rng: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized, streaming, BF16 autocast:
      - No autograd graph is built.
      - Processes 'pat_chunk' parity patterns at once (no Python loop over patterns).
      - Uses BF16 for the large (P × Ci × B) tensors; accumulators in FP32.
    Returns:
      J:      (B,)
      Sigma:  (B,)
      dJ:     (B,d)
      dSigma: (B,d)  (for ReLU with estimate_sigma=False, uses exact Σ=||w||^2/2 ⇒ dΣ=w)
    """
    B, d = w.shape
    k = S.numel()
    S = S.long().to(device)

    # Complement indices
    all_idx = torch.arange(d, device=device, dtype=torch.long)
    mask = torch.ones(d, device=device, dtype=torch.bool); mask[S] = False
    Sc = all_idx[mask]

    # Precompute all parity patterns (2^k × k) and chi over them
    patterns_full = all_parity_patterns(k, device=device, dtype=torch.bfloat16)  # (P_all,k) BF16
    chi_full = patterns_full.prod(dim=1).to(torch.bfloat16)                      # (P_all,)

    # Accumulators (FP32 for speed/accuracy balance)
    J_sum     = torch.zeros(B, device=device, dtype=torch.float32)
    dJ_sum    = torch.zeros(B, d, device=device, dtype=torch.float32)

    need_empirical_sigma = (mdl.act == "tanh") or (mdl.act == "relu" and mdl.estimate_sigma)
    if need_empirical_sigma:
        Sigma_sum  = torch.zeros(B, device=device, dtype=torch.float32)
        dSigma_sum = torch.zeros(B, d, device=device, dtype=torch.float32)

    # Total (contexts × patterns) count accumulated
    ctx_pat_count = 0

    # === Chain batches ===
    for b0 in range(0, B, chunks.batch_chunk):
        b1 = min(B, b0 + chunks.batch_chunk)
        wB   = w[b0:b1]                                  # (Bb,d)
        Bb   = wB.shape[0]
        wB_S  = wB[:, S].to(torch.bfloat16)              # (Bb,k)
        wB_Sc = wB[:, Sc].to(torch.bfloat16)             # (Bb,d-k)

        # === Passes ===
        for _ in range(chunks.ctx_passes):
            remaining = chunks.ctx_chunk
            while remaining > 0:
                Ci = min(chunks.ctx_inner, remaining)
                remaining -= Ci

                # contexts Xc: (Ci, d-k) in {-1, +1}
                Xc = torch.randint(
                    0, 2, (Ci, Sc.numel()),
                    device=device, dtype=torch.int8, generator=rng
                ).to(torch.bfloat16)
                Xc = Xc * 2 - 1                          # {-1,+1} in BF16

                # b_block = Xc @ w_Sc^T  => (Ci, Bb) BF16
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    b_block = Xc @ wB_Sc.T               # (Ci,Bb)

                # === Parity pattern chunks (vectorized over patterns) ===
                for p0 in range(0, patterns_full.shape[0], chunks.pat_chunk):
                    p1 = min(patterns_full.shape[0], p0 + chunks.pat_chunk)
                    pat   = patterns_full[p0:p1, :]      # (P,k) BF16
                    chi_p = chi_full[p0:p1]              # (P,) BF16  (±1)

                    # sdotw = pat @ w_S^T => (P,Bb)  (BF16)
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        sdotw = pat @ wB_S.T            # (P,Bb)
                        # z = b_block[None,:,:] + sdotw[:,None,:] => (P,Ci,Bb)
                        z = b_block.unsqueeze(0) + sdotw.unsqueeze(1)

                        if mdl.act == "relu":
                            phi   = torch.nn.functional.relu(z)           # (P,Ci,Bb)
                            phi_p = (z > 0).to(torch.bfloat16)            # (P,Ci,Bb)
                        elif mdl.act == "tanh":
                            phi   = torch.tanh(z)                          # (P,Ci,Bb)
                            phi_p = 1.0 - phi * phi                        # (P,Ci,Bb)
                        else:
                            raise ValueError("activation")

                        # Sums over contexts
                        sumC_phi   = phi.sum(dim=1)                       # (P,Bb) BF16
                        sumC_phip  = phi_p.sum(dim=1)                     # (P,Bb) BF16

                        # ---------- J accumulation ----------
                        # J += sum_{patterns} chi * sum_C phi
                        J_sum[b0:b1] += (chi_p[:, None] * sumC_phi).sum(dim=0).to(torch.float32)

                        # ---------- dJ for Sc ----------
                        # sum over patterns of chi * phi_p  -> (Ci,Bb)
                        comb_C_B = (chi_p[:, None, None] * phi_p).sum(dim=0)  # (Ci,Bb)
                        # Xc.T @ comb_C_B -> (d-k,Bb)
                        dJ_Sc = (Xc.T @ comb_C_B).to(torch.float32)           # (d-k,Bb)
                        dJ_sum[b0:b1, Sc] += dJ_Sc.T                           # (Bb,d-k) -> assign

                        # ---------- dJ for S ----------
                        # (Bb,k) = einsum_p [ (sumC_phip[p,b]) * (pat[p,k]) * chi[p] ]
                        dJ_S = torch.einsum('pb,pk,p->bk',
                                            sumC_phip.to(torch.float32),
                                            pat.to(torch.float32),
                                            chi_p.to(torch.float32))
                        dJ_sum[b0:b1, S] += dJ_S

                        # ---------- Σ & dΣ (if empirical) ----------
                        if need_empirical_sigma:
                            phi2 = phi * phi                                 # (P,Ci,Bb)
                            sumC_phi2  = phi2.sum(dim=1)                     # (P,Bb)
                            Sigma_sum[b0:b1] += sumC_phi2.sum(dim=0).to(torch.float32)

                            two_phi_phi_p = 2.0 * phi * phi_p                # (P,Ci,Bb)
                            # Sc part: Xc.T @ sum_patterns two_phi_phi_p
                            comb2_C_B = two_phi_phi_p.sum(dim=0)             # (Ci,Bb)
                            dSigma_Sc = (Xc.T @ comb2_C_B).to(torch.float32) # (d-k,Bb)
                            dSigma_sum[b0:b1, Sc] += dSigma_Sc.T

                            # S part: einsum over patterns
                            sumC_two = two_phi_phi_p.sum(dim=1).to(torch.float32)  # (P,Bb)
                            dSigma_S = torch.einsum('pb,pk->bk', sumC_two, pat.to(torch.float32))
                            dSigma_sum[b0:b1, S] += dSigma_S

                    ctx_pat_count += Ci * (p1 - p0)

    # Finalize means (divide by number of (contexts × patterns))
    denom = max(1, ctx_pat_count)
    J  = (J_sum / denom).to(torch.float32)
    dJ = (dJ_sum / denom).to(torch.float32)

    if need_empirical_sigma:
        Sigma  = (Sigma_sum / denom).to(torch.float32)
        dSigma = (dSigma_sum / denom).to(torch.float32)
    else:
        # ReLU exact Σ = 0.5 * ||w||^2, dΣ = w
        Sigma  = 0.5 * (w * w).sum(dim=1)
        dSigma = w.clone()

    return J, Sigma, dJ, dSigma



# ----------------------------- logπ and grad (closed-form using J,Σ and their grads) -----------------------------

def compute_logp_and_grad(
    w: torch.Tensor,
    S: torch.Tensor,
    device: torch.device,
    m: float, chi: float, kappa: float,
    mdl: ModelParams, saem: SAEMParams,
    chunks: EstimatorChunking,
    rng: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    logp(w) = -||w||^2/(2 gw^2) - 0.5 log D + ((1-m)^2/(2 kappa^2)) * (J^2 / D)
    D = A kappa^2 + Sigma - (chi/N) J^2
    """
    gw2 = mdl.sigma_w / mdl.d
    ga2 = mdl.sigma_a / (mdl.N ** mdl.gamma)
    A   = 1.0 / ga2

    # Estimate J, Sigma and their gradients w.r.t w (no autograd graph)
    J, Sigma, dJ, dSigma = compute_J_Sigma_grad_kexact(
        w=w, S=S, device=device, mdl=mdl, chunks=chunks, rng=rng
    )

    D = A * (kappa**2) + Sigma - (chi / mdl.N) * (J * J)
    D_safe = torch.clamp(D, min=saem.eps_D)

    # scalars per-chain
    c = ((1.0 - m)**2) / (2.0 * (kappa**2))

    # ∇ prior
    grad = -w / gw2

    # dD = dSigma - (chi/N) * 2 J dJ
    dD = dSigma - (2.0 * chi / mdl.N) * (J[:, None] * dJ)

    # Add gradient of -0.5 * log D
    grad += (-0.5) * (dD / D_safe[:, None])

    # Add gradient of c * (J^2 / D)
    # ∇[(J^2)/D] = (2 J dJ)/D - (J^2 / D^2) dD
    term1 = (2.0 * J)[:, None] * dJ / D_safe[:, None]
    term2 = ((J * J) / (D_safe * D_safe))[:, None] * dD
    grad += c * (term1 - term2)

    # logp (diagnostics only)
    prior_term = -0.5 * (w * w).sum(dim=1) / gw2
    log_det_term = -0.5 * torch.log(D_safe)
    data_term = c * (J * J) / D_safe
    logp = prior_term + log_det_term + data_term

    return logp.detach(), grad, J.detach(), Sigma.detach()

# ----------------------------- Sampler -----------------------------

def mcmc_sgld(
    w: torch.Tensor,
    S: torch.Tensor,
    device: torch.device,
    m: float, chi: float, kappa: float,
    mdl: ModelParams, saem: SAEMParams,
    mcmc: MCMCParams,
    chunks: EstimatorChunking,
    rng: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    step = mcmc.step_size
    for _ in range(mcmc.n_steps):
        _, grad, J, Sigma = compute_logp_and_grad(
            w, S, device, m, chi, kappa, mdl, saem, chunks, rng
        )
        # clip grad
        if mcmc.grad_clip and mcmc.grad_clip > 0:
            gn = grad.norm(dim=1, keepdim=True).clamp_min(1e-12)
            grad = grad * (mcmc.grad_clip / gn).clamp(max=1.0)

        if mcmc.langevin_sqrt2:
            noise = torch.randn(w.shape, device=device, dtype=w.dtype, generator=rng) * math.sqrt(2.0 * step)
            w = w + step * grad + noise
        else:
            noise = torch.randn(w.shape, device=device, dtype=w.dtype, generator=rng) * math.sqrt(step)
            w = w + 0.5 * step * grad + noise

        if mcmc.clamp_w:
            w = torch.clamp(w, min=-mcmc.clamp_w, max=mcmc.clamp_w)
        step *= mcmc.step_decay
    return w.detach(), J.detach(), Sigma.detach()

# ----------------------------- Feasibility (D>0) -----------------------------

def feasible_project(m: float, chi: float, mdl: ModelParams, saem: SAEMParams):
    """
    Hard global cap from Cauchy–Schwarz: sup_w J^2/Sigma ≤ 1  ⇒  χ ≤ (1-ε) N.
    Also clamp 0 ≤ m ≤ 1.
    """
    hard_cap = (1.0 - saem.eps_proj) * mdl.N
    chi_proj = min(max(chi, 0.0), hard_cap)
    m_proj   = min(max(m, 0.0), 1.0)
    return m_proj, chi_proj, hard_cap

# ----------------------------- SAEM Loop (raw mean pooling) -----------------------------

def saem_solve_for_kappa(
    kappa: float, devices: List[int],
    BASE: Dict, SAVE_DIR: str, run_tag: str = ""
):
    mdl = ModelParams(
        d=BASE["d"], N=BASE["N"], k=BASE["k"],
        sigma_a=BASE["σa"], sigma_w=BASE["σw"],
        gamma=BASE["γ"], act=BASE.get("act", "relu"),
        estimate_sigma=BASE.get("estimate_sigma", False),
    )
    mcmc = MCMCParams(
        n_chains_per_device=BASE.get("chains_per_device", 8192),
        n_steps=BASE.get("mcmc_steps", 120),
        step_size=BASE.get("mcmc_step_size", 1e-3),
        step_decay=BASE.get("mcmc_step_decay", 0.999),
        langevin_sqrt2=BASE.get("langevin_sqrt2", True),
        grad_clip=BASE.get("grad_clip", 10.0),
        clamp_w=BASE.get("clamp_w", 10.0),
    )
    saem = SAEMParams(
        max_iters=BASE.get("opt_steps", 1000),
        a0=BASE.get("a0", 0.5),
        t0=BASE.get("t0", 100.0),
        damping=BASE.get("damping", 1.0),
        eps_D=BASE.get("eps_D", 1e-6),
        eps_proj=BASE.get("eps_proj", 1e-3),
        print_every=BASE.get("print_every", 5),
    )
    chunks = EstimatorChunking(
        ctx_chunk=BASE.get("ctx_chunk", 65536),
        ctx_passes=BASE.get("ctx_passes", 16),
        ctx_inner=BASE.get("ctx_inner", 4096),
        pat_chunk=BASE.get("pat_chunk", 16),
        batch_chunk=BASE.get("batch_chunk", 4096),
    )

    # Teacher indices
    S = make_parity_indices(mdl.d, mdl.k, seed=BASE.get("teacher_seed", 0))

    # Per-device state
    per_dev = []
    total_chains = 0
    for di in devices if len(devices) > 0 else [-1]:
        device = torch.device(f"cuda:{di}") if di >= 0 and torch.cuda.is_available() else torch.device("cpu")
        chains = mcmc.n_chains_per_device
        gw2 = BASE["σw"] / BASE["d"]
        w = torch.randn(chains, mdl.d, device=device) * math.sqrt(gw2)
        # One RNG per device
        rng = torch.Generator(device=device).manual_seed(BASE.get("data_seed", 0) + (di+1)*12345)
        per_dev.append({"device": device, "w": w, "rng": rng})
        total_chains += chains

    # Init order parameters
    m = BASE.get("m_init", 0.2)
    chi = BASE.get("chi_init", 1e-6)
    m_bar, chi_bar = m, chi

    history = []
    t_start = time.time()

    for it in range(1, saem.max_iters + 1):
        g1_sum = 0.0
        g2_sum = 0.0
        n_sum  = 0
        rho_sum = 0.0
        rho_max = 0.0
        D_min_global = float("inf")
        small_D_count = 0
        total_B = 0

        for slot in per_dev:
            device = slot["device"]
            w = slot["w"]
            rng = slot["rng"]

            w, J, Sigma = mcmc_sgld(w, S, device, m, chi, kappa, mdl, saem, mcmc, chunks, rng)
            slot["w"] = w

            ga2 = mdl.sigma_a / (mdl.N ** mdl.gamma)
            A = 1.0 / ga2
            D = A * (kappa**2) + Sigma - (chi / mdl.N) * (J * J)
            D = torch.clamp(D, min=saem.eps_D)

            g1 = (J * J) / D
            g2 = ((J * J) * (J * J)) / ((kappa**2) * (D * D))

            g1_sum += g1.sum().item()
            g2_sum += g2.sum().item()
            n_sum  += g1.numel()

            rho = (J * J / torch.clamp(Sigma, min=1e-12))
            rho_sum += rho.mean().item()
            rho_max = max(rho_max, rho.max().item())

            D_min_global = min(D_min_global, D.min().item())
            small_D_count += (D < 1e-3).sum().item()
            total_B += D.numel()

        g1_mean = g1_sum / max(1, n_sum)
        g2_mean = g2_sum / max(1, n_sum)
        rho_mean = rho_sum / max(1, len(per_dev))

        a_t = saem.a0 / (it + saem.t0)

        F1 = m   - mdl.N * (1.0 - m) * g1_mean
        F2 = chi - mdl.N * (1.0 - m) * (1.0 - m) * g2_mean - mdl.N * g1_mean

        m_new   = m   - saem.damping * a_t * F1
        chi_new = chi - saem.damping * a_t * F2

        m, chi, chi_bound = feasible_project(m_new, chi_new, mdl, saem)

        m_bar   = ((it-1)*m_bar   + m  ) / it
        chi_bar = ((it-1)*chi_bar + chi) / it

        if it % saem.print_every == 0 or it == 1:
            dt = time.time() - t_start
            msg = {
                "iter": it, "kappa": kappa,
                "m": m, "chi": chi,
                "m_bar": m_bar, "chi_bar": chi_bar,
                "g1_mean": g1_mean, "g2_mean": g2_mean,
                "rho_mean": rho_mean, "rho_max": rho_max,
                "D_min": D_min_global,
                "frac_small_D": float(small_D_count) / max(1, total_B),
                "chi_bound": chi_bound,
                "time_s": round(dt, 2)
            }
            print(json.dumps(msg))
            history.append(msg)

    result = {
        "kappa": kappa,
        "m_final": m, "chi_final": chi,
        "m_bar": m_bar, "chi_bar": chi_bar,
        "history": history[-10:],
        "BASE": BASE
    }

    os.makedirs(SAVE_DIR, exist_ok=True)
    tag = run_tag if run_tag else f"kappa_{kappa:.6f}x"
    out_path = os.path.join(SAVE_DIR, f"saem_kexact_stream_result_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[saved] {out_path}")
    return result

# ----------------------------- Main -----------------------------

if __name__ == "__main__":
    # Optional: slightly faster matmuls on Hopper
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    set_seed(42)
    devices = check_gpu()

    # === H100-friendly defaults (OOM-safe) ===
    BASE = dict(
        d=40, N=1024, k=4,
        σa=1.0, σw=1.0, γ=1.0,
        act="relu",
        estimate_sigma=False,  # set True to MC-estimate Σ even for ReLU
        opt_steps=1000,
        # SGLD
        chains_per_device=8192,
        mcmc_steps=120,
        mcmc_step_size=1e-3,
        mcmc_step_decay=0.999,
        langevin_sqrt2=True,
        grad_clip=10.0,
        clamp_w=10.0,
        # k-exact estimator chunking
        ctx_chunk=262144  ,      # contexts per pass
        ctx_passes=8,        # passes per iteration
        ctx_inner=65536 ,       # micro-batch over contexts
        pat_chunk=16,         # ≤ 2^k
        batch_chunk=16384   ,     # chains per micro-batch
        # SAEM controls
        a0=0.5, t0=100.0, damping=1.0,
        eps_D=1e-6, eps_proj=1e-3, print_every=5,
        # Seeds
        teacher_seed=0, data_seed=0,
        # inits
        m_init=0.2, chi_init=1e-6,
    )

    # Warm start from small -> larger kappa
    kappa_list = np.logspace(np.log10(1e-3), np.log10(1e-1), 12)

    save_dir = "/home/goring/mean_field_langevin/MCMC_Pinf_1N_sparse/results/1808_d40k4_nospecialst_Rnew"
    os.makedirs(save_dir, exist_ok=True)

    run_tag_prefix = time.strftime("%Y%m%d_%H%M%S")

    m_ws  = BASE.get("m_init", 0.2)
    chi_ws= BASE.get("chi_init", 1e-6)

    for idx, kappa in enumerate(kappa_list):
        print(f"\n=== SAEM (pooled, k-exact, streaming) for kappa={kappa} ===")
        BASE["m_init"] = m_ws
        BASE["chi_init"] = chi_ws
        tag = f"{run_tag_prefix}_k{idx}_{kappa:.6f}"
        result = saem_solve_for_kappa(
            kappa=kappa,
            devices=devices,
            BASE=BASE,
            SAVE_DIR=save_dir,
            run_tag=tag
        )
        m_ws, chi_ws = result["m_final"], result["chi_final"]
