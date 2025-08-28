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

def make_Sc_from_S(d: int, S: torch.Tensor, device: torch.device) -> torch.Tensor:
    Sset = set(S.tolist())
    Sc = torch.tensor([i for i in range(d) if i not in Sset], device=device, dtype=torch.long)
    return Sc

def build_patterns_and_chi(k: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if k == 0:
        patterns = torch.zeros(1, 0, device=device)
        chi_s = torch.ones(1, device=device)
        return patterns, chi_s
    base = torch.tensor([-1.0, 1.0], device=device)
    grids = torch.meshgrid(*([base] * k), indexing="ij")
    patterns = torch.stack(grids, dim=-1).reshape(-1, k)  # (2^k, k)
    chi_s = patterns.prod(dim=1)  # (2^k,)
    return patterns, chi_s

# ----------------------------- Dataclasses -----------------------------

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
    # SGLD
    n_chains_per_device: int = 8192
    n_steps: int = 30
    step_size: float = 5e-3
    step_decay: float = 0.999
    langevin_sqrt2: bool = True
    grad_clip: float = 10.0
    clamp_w: float = 10.0
    # Estimator (k-exact)
    R_ctx: int = 32768
    ctx_chunk: int = 65536
    pat_chunk: int = 16
    batch_chunk: int = 8192

@dataclass
class SAEMParams:
    max_iters: int = 4000
    a0: float = 0.5
    t0: float = 100.0
    damping: float = 1.0
    eps_D: float = 1e-6
    eps_proj: float = 1e-3
    print_every: int = 50
    trim_frac: float = 0.10
    d_guard_threshold: float = 1e-3
    d_guard_backoff: float = 0.9

# ----------------------------- k-exact J, Σ (chunked) -----------------------------

def compute_J_Sigma_kexact_chunked(
    w: torch.Tensor,                      # (B, d)
    S: torch.Tensor,                      # (k,)
    Sc: torch.Tensor,                     # (d-k,)
    patterns: torch.Tensor,               # (P, k) in {-1, +1}
    chi_s: torch.Tensor,                  # (P,)
    R_ctx: int,
    device: torch.device,
    act: str,
    ctx_chunk: int,
    pat_chunk: int,
    batch_chunk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact over parity bits (2^k patterns), Monte Carlo over context x_{-S}.
    Always estimates Sigma by averaging phi^2.
    Returns:
        J:     (B,)
        Sigma: (B,)
    """
    B, d = w.shape
    P = patterns.size(0)

    J_out = torch.empty(B, device=device)
    Sig_out = torch.empty(B, device=device)

    for b0 in range(0, B, batch_chunk):
        b1 = min(B, b0 + batch_chunk)
        wb = w[b0:b1]                          # (Bb, d)
        Bb = wb.size(0)

        wb_S  = wb[:, S]                       # (Bb, k)
        wb_Sc = wb[:, Sc]                      # (Bb, d-k)

        J_sum   = torch.zeros(Bb, device=device)
        Sig_sum = torch.zeros(Bb, device=device)
        tot_cnt = 0

        for p0 in range(0, P, pat_chunk):
            p1 = min(P, p0 + pat_chunk)
            patt = patterns[p0:p1, :]          # (Pp, k)
            chi_p = chi_s[p0:p1]               # (Pp,)

            sdotw = patt @ wb_S.T              # (Pp, Bb)

            remain = R_ctx
            while remain > 0:
                C = min(ctx_chunk, remain)
                remain -= C
                Xc = torch.randint(0, 2, (C, wb_Sc.size(1)), device=device, dtype=torch.int8).float() * 2 - 1  # (C, d-k)
                b_ctx = Xc @ wb_Sc.T            # (C, Bb)

                z = b_ctx.unsqueeze(0) + sdotw[:, None, :]   # (Pp, C, Bb)
                phi = activation(z, act)

                Sig_sum += (phi * phi).sum(dim=(0, 1))
                J_sum   += (phi * chi_p.view(-1, 1, 1)).sum(dim=(0, 1))
                tot_cnt += (p1 - p0) * C

                del Xc, b_ctx, z, phi

            del patt, chi_p, sdotw

        J_out[b0:b1]   = J_sum / max(1, tot_cnt)
        Sig_out[b0:b1] = Sig_sum / max(1, tot_cnt)

    return J_out, Sig_out

# ----------------------------- logπ and grad -----------------------------

def compute_logp_and_grad(
    w: torch.Tensor,
    m: float, chi: float, kappa: float,
    mdl: ModelParams, saem: SAEMParams, mcmc: MCMCParams,
    S: torch.Tensor, Sc: torch.Tensor, patterns: torch.Tensor, chi_s: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    log p(w) after integrating out a:
      g_w^2 = sigma_w / d
      g_a^2 = sigma_a / N^gamma
      A     = 1 / g_a^2  = N^gamma / sigma_a
      D     = A*kappa^2 + Sigma - (chi/N) J^2
      logp  = -||w||^2/(2 g_w^2) - 0.5*log D + ((1-m)^2/(2 kappa^2)) * (J^2 / D)
    """
    gw2 = mdl.sigma_w / mdl.d
    ga2 = mdl.sigma_a / (mdl.N ** mdl.gamma)
    A   = 1.0 / ga2

    w = w.detach().requires_grad_(True)

    J, Sigma = compute_J_Sigma_kexact_chunked(
        w, S, Sc, patterns, chi_s,
        R_ctx=mcmc.R_ctx, device=w.device, act=mdl.act,
        ctx_chunk=mcmc.ctx_chunk, pat_chunk=mcmc.pat_chunk, batch_chunk=mcmc.batch_chunk
    )

    D = A * (kappa**2) + Sigma - (chi / mdl.N) * (J * J)
    D_safe = torch.clamp(D, min=saem.eps_D)

    prior_term = -0.5 * (w * w).sum(dim=1) / gw2
    log_det_term = -0.5 * torch.log(D_safe)
    data_term    = ((1.0 - m)**2) / (2.0 * (kappa**2)) * (J * J) / D_safe

    logp = prior_term + log_det_term + data_term

    grad = torch.autograd.grad(logp.sum(), w, create_graph=False, retain_graph=False)[0]
    if mcmc.grad_clip and mcmc.grad_clip > 0:
        gn = grad.norm(dim=1, keepdim=True).clamp_min(1e-12)
        grad = grad * (mcmc.grad_clip / gn).clamp(max=1.0)
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    return logp.detach(), grad.detach(), J.detach(), Sigma.detach(), D_safe.detach()

# ----------------------------- Sampler -----------------------------

def mcmc_sgld(
    w: torch.Tensor,
    m: float, chi: float, kappa: float,
    mdl: ModelParams, saem: SAEMParams, mcmc: MCMCParams,
    S: torch.Tensor, Sc: torch.Tensor, patterns: torch.Tensor, chi_s: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    step = mcmc.step_size
    J_last = None
    Sigma_last = None
    D_last = None
    for _ in range(mcmc.n_steps):
        _, grad, J, Sigma, D = compute_logp_and_grad(
            w, m, chi, kappa, mdl, saem, mcmc, S, Sc, patterns, chi_s
        )
        J_last, Sigma_last, D_last = J, Sigma, D

        if mcmc.langevin_sqrt2:
            noise = torch.randn_like(w) * math.sqrt(2.0 * step)
            w = w + step * grad + noise
        else:
            noise = torch.randn_like(w) * math.sqrt(step)
            w = w + 0.5 * step * grad + noise

        if mcmc.clamp_w:
            w = torch.clamp(w, min=-mcmc.clamp_w, max=mcmc.clamp_w)
        step *= mcmc.step_decay
    return w.detach(), J_last.detach(), Sigma_last.detach(), D_last.detach()

# ----------------------------- Robust pooling & feasibility -----------------------------

def trimmed_mean(v: torch.Tensor, trim: float = 0.1) -> torch.Tensor:
    if v.numel() == 0:
        return torch.tensor(0.0, device=v.device)
    t = max(0.0, min(0.49, float(trim)))
    q_lo = v.quantile(torch.tensor(t, device=v.device))
    q_hi = v.quantile(torch.tensor(1.0 - t, device=v.device))
    mask = (v >= q_lo) & (v <= q_hi)
    if mask.sum() == 0:
        return v.mean()
    return v[mask].mean()

def feasible_project(m: float, chi: float, mdl: ModelParams, saem: SAEMParams):
    # Hard global cap from Cauchy–Schwarz: sup_w J^2/Σ ≤ 1  ⇒  χ ≤ (1 - eps) * N
    hard_cap = (1.0 - saem.eps_proj) * mdl.N
    chi_proj = min(max(chi, 0.0), hard_cap)
    m_proj   = min(max(m, 0.0), 1.0)
    return m_proj, chi_proj, hard_cap

# ----------------------------- SAEM Loop (pooled estimator) -----------------------------

def saem_solve_for_kappa(
    kappa: float, devices: List[int],
    BASE: Dict, SAVE_DIR: str, run_tag: str = ""
):
    mdl = ModelParams(
        d=BASE["d"], N=BASE["N"], k=BASE["k"],
        sigma_a=BASE["σa"], sigma_w=BASE["σw"],
        gamma=BASE["γ"], act=BASE.get("act", "relu")
    )
    mcmc = MCMCParams(
        n_chains_per_device=BASE.get("chains_per_device", 8192),
        n_steps=BASE.get("mcmc_steps", 30),
        step_size=BASE.get("mcmc_step_size", 5e-3),
        step_decay=BASE.get("mcmc_step_decay", 0.999),
        langevin_sqrt2=BASE.get("langevin_sqrt2", True),
        grad_clip=BASE.get("grad_clip", 10.0),
        clamp_w=BASE.get("clamp_w", 10.0),
        R_ctx=BASE.get("R_inputs", 32768),
        ctx_chunk=BASE.get("ctx_chunk", 65536),
        pat_chunk=BASE.get("pat_chunk", 16),
        batch_chunk=BASE.get("batch_chunk", 8192),
    )
    saem = SAEMParams(
        max_iters=BASE.get("opt_steps", 4000),
        a0=BASE.get("a0", 0.5),
        t0=BASE.get("t0", 100.0),
        damping=BASE.get("damping", 1.0),
        eps_D=BASE.get("eps_D", 1e-6),
        eps_proj=BASE.get("eps_proj", 1e-3),
        print_every=BASE.get("print_every", 50),
        trim_frac=BASE.get("trim_frac", 0.10),
        d_guard_threshold=BASE.get("d_guard_threshold", 1e-3),
        d_guard_backoff=BASE.get("d_guard_backoff", 0.9),
    )

    # Teacher indices
    S_cpu = make_parity_indices(mdl.d, mdl.k, seed=BASE.get("teacher_seed", 0))

    # Per-device state
    per_dev = []
    for di in devices if len(devices) > 0 else [-1]:
        device = torch.device(f"cuda:{di}") if di >= 0 and torch.cuda.is_available() else torch.device("cpu")
        S_dev  = S_cpu.to(device)
        Sc_dev = make_Sc_from_S(mdl.d, S_dev, device)
        patterns, chi_s = build_patterns_and_chi(mdl.k, device)

        gw2 = BASE["σw"] / BASE["d"]
        chains = mcmc.n_chains_per_device
        w = torch.randn(chains, mdl.d, device=device) * math.sqrt(gw2)

        per_dev.append({
            "device": device, "w": w,
            "S": S_dev, "Sc": Sc_dev,
            "patterns": patterns, "chi_s": chi_s
        })

    # Init order parameters
    m = BASE.get("m_init", 0.0)
    chi = BASE.get("chi_init", 1e-6)
    m_bar, chi_bar = m, chi

    t_start = time.time()
    print(f"=== SAEM (pooled, k-exact) for kappa={kappa} ===")

    for it in range(1, saem.max_iters + 1):
        g1_all = []
        g2_all = []
        rho_all = []
        D_min_global = float("inf")
        D_small_frac_acc = 0.0
        n_total = 0

        for slot in per_dev:
            w = slot["w"]
            S = slot["S"]; Sc = slot["Sc"]
            patterns = slot["patterns"]; chi_s = slot["chi_s"]

            w, J, Sigma, D = mcmc_sgld(
                w, m, chi, kappa, mdl, saem, mcmc, S, Sc, patterns, chi_s
            )
            slot["w"] = w
            D_safe = torch.clamp(D, min=saem.eps_D)

            g1_vec = (J * J) / D_safe
            g2_vec = ((J * J) * (J * J)) / ((kappa**2) * (D_safe * D_safe))
            rho_vec = (J * J) / torch.clamp(Sigma, min=1e-12)

            g1_all.append(g1_vec)
            g2_all.append(g2_vec)
            rho_all.append(rho_vec)

            D_min_global = min(D_min_global, float(D_safe.min().item()))
            D_small_frac_acc += (D_safe < saem.d_guard_threshold).float().sum().item()
            n_total += D_safe.numel()

        g1_all = torch.cat(g1_all, dim=0)
        g2_all = torch.cat(g2_all, dim=0)
        rho_all = torch.cat(rho_all, dim=0)

        g1_mean_raw = g1_all.mean().item()
        g2_mean_raw = g2_all.mean().item()
        g1_mean = float(trimmed_mean(g1_all, saem.trim_frac).item())
        g2_mean = float(trimmed_mean(g2_all, saem.trim_frac).item())
        rho_mean = float(trimmed_mean(rho_all, saem.trim_frac).item())
        rho_max  = float(rho_all.max().item())

        frac_small = D_small_frac_acc / max(1, n_total)
        if frac_small > 0.05 or D_min_global < saem.d_guard_threshold:
            chi *= saem.d_guard_backoff

        a_t = saem.a0 / (it + saem.t0)

        # >>> Correct F1 (no extra 1/kappa^2) <<<
        F1 = m   - mdl.N * (1.0 - m) * g1_mean
        F2 = chi - mdl.N * (1.0 - m) * (1.0 - m) * g2_mean - mdl.N * g1_mean

        m_new   = m   - saem.damping * a_t * F1
        chi_new = chi - saem.damping * a_t * F2

        m, chi, chi_bound = feasible_project(m_new, chi_new, mdl, saem)

        m_bar   = ((it - 1) * m_bar   + m  ) / it
        chi_bar = ((it - 1) * chi_bar + chi) / it

        if it % saem.print_every == 0 or it == 1:
            dt = time.time() - t_start
            msg = {
                "iter": it, "kappa": kappa,
                "m": m, "chi": chi,
                "m_bar": m_bar, "chi_bar": chi_bar,
                "g1_mean": g1_mean, "g2_mean": g2_mean,
                "g1_mean_raw": g1_mean_raw, "g2_mean_raw": g2_mean_raw,
                "rho_mean": rho_mean, "rho_max": rho_max,
                "D_min": D_min_global, "frac_small_D": frac_small,
                "chi_bound": chi_bound,
                "time_s": round(dt, 2),
            }
            print(json.dumps(msg))

    result = {
        "kappa": kappa,
        "m_final": m, "chi_final": chi,
        "m_bar": m_bar, "chi_bar": chi_bar,
        "BASE": BASE
    }

    os.makedirs(SAVE_DIR, exist_ok=True)
    tag = run_tag if run_tag else f"kappa_{kappa:.6f}"
    out_path = os.path.join(SAVE_DIR, f"saem_nomix_result_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[saved] {out_path}")
    return result

# ----------------------------- Main -----------------------------

if __name__ == "__main__":
    set_seed(42)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

    devices = check_gpu()

    BASE = dict(
        d=40, N=1024, k=4,
        σa=1.0, σw=1.0, γ=1.0,
        act="relu",
        opt_steps=4000,
        # Estimator controls (k-exact)
        R_inputs=32768,          # used as R_ctx
        # SGLD / chains
        chains_per_device=8192,
        mcmc_steps=150,
        mcmc_step_size=1e-3,
        mcmc_step_decay=0.999,
        langevin_sqrt2=True,
        grad_clip=10.0,
        clamp_w=10.0,
        # Chunking (H100)
        ctx_chunk=65536,
        pat_chunk=32,            # >= 2^k (we slice internally)
        batch_chunk=8192,
        # SAEM controls
        a0=0.5, t0=100.0, damping=0.5,
        eps_D=1e-6, eps_proj=1e-3, print_every=5,
        trim_frac=0.10,
        d_guard_threshold=1e-3,
        d_guard_backoff=0.9,
        # Seeds
        teacher_seed=0,
        # inits
        m_init=0.2, chi_init=1e-6,
    )

    kappa_list = np.logspace(np.log10(1e-3), np.log10(1e-1), 12)

    save_dir = "/home/goring/mean_field_langevin/MCMC_Pinf_1N_sparse/results/1808_d40k4_nospecialst_Rnew"
    os.makedirs(save_dir, exist_ok=True)
    run_tag_prefix = time.strftime("%Y%m%d_%H%M%S")

    m_ws  = BASE.get("m_init", 0.0)
    chi_ws= BASE.get("chi_init", 1e-6)

    for idx, kappa in enumerate(kappa_list):
        print(f"\n=== SAEM (pooled, k-exact) for kappa={kappa} ===")
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
