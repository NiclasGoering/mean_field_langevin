# nngp_relu_gp_composite.py
import os, json, time, math, random, re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from torch import Tensor


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


# ----------------------------- Parity / Composite labels -----------------------------
def make_parity_indices(d: int, k: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    S = torch.randperm(d, generator=g)[:k]
    return S.sort().values  # (k,)


def parity_character(X_pm1: Tensor, S: Tensor) -> Tensor:
    """
    χ_S(x) = ∏_{i in S} x_i for x_i ∈ {±1}.
    X_pm1: (P, d) with entries ±1
    S:     (k,)
    Returns: (P,) with entries ±1
    """
    if S.numel() == 0:
        return torch.ones(X_pm1.shape[0], device=X_pm1.device, dtype=torch.float32)
    feats = X_pm1[:, S]             # (P, k)
    y = feats.prod(dim=1)           # (P,)
    return y.to(torch.float32)


def parse_composite_spec(spec: str) -> List[List[int]]:
    """
    Parse strings like:
      "{0,1,2,3}+{0,1,2,3,4,5,6,7}"
      " { 1 , 3 } + { 2 , 4 ,5 } "
    into [[0,1,2,3], [0,1,2,3,4,5,6,7]]
    """
    sets = re.findall(r"\{([^}]*)\}", spec)
    out: List[List[int]] = []
    for s in sets:
        if s.strip() == "":
            out.append([])
            continue
        elems = [int(tok.strip()) for tok in s.split(",") if tok.strip() != ""]
        out.append(sorted(elems))
    if len(out) == 0:
        raise ValueError(f"Failed to parse composite spec: {spec!r}")
    return out


def compute_characters_matrix(X_pm1: Tensor, sets: List[Tensor]) -> Tensor:
    """
    Return C of shape (P, M) where C[:, j] = χ_{S_j}(x)
    """
    P = X_pm1.shape[0]
    M = len(sets)
    if M == 0:
        return torch.zeros(P, 0, device=X_pm1.device, dtype=torch.float32)
    C = torch.empty(P, M, device=X_pm1.device, dtype=torch.float32)
    for j, Sj in enumerate(sets):
        C[:, j] = parity_character(X_pm1, Sj)
    return C


def generate_composite_data(P: int, d: int, sets: List[Tensor], device) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Composite teacher: y(x) = sum_j χ_{S_j}(x)
    Returns:
      X: (P, d) in {±1}
      y: (P,) real-valued
      C: (P, M) characters matrix with C[:, j] = χ_{S_j}(x)
    """
    g = torch.Generator(device=device).manual_seed(0)
    X = (torch.randint(0, 2, (P, d), generator=g, device=device, dtype=torch.int8).float() * 2.0 - 1.0)
    C = compute_characters_matrix(X, sets)  # (P, M)
    y = C.sum(dim=1) if C.numel() > 0 else torch.zeros(P, device=device, dtype=torch.float32)
    return X, y, C


# ----------------------------- ReLU arc-cosine (NNGP) kernel -----------------------------
@torch.no_grad()
def relu_arccos_kernel_block(
    X_rows: Tensor, X_cols: Tensor,
    sigma_w: float = 1.0, sigma_a: float = 1.0,
    autocast_bf16: bool = True
) -> Tensor:
    """
    Compute K_block = σ_a^2 * κ_ReLU(X_rows, X_cols)
    with w ~ N(0, σ_w^2 I / d), bias=0.
    κ_ReLU(x,x') = (σ_w^2 / (2π)) * ||x|| ||x'|| * (sinθ + (π-θ) cosθ),
    where cosθ = (x·x') / (||x|| ||x'||).

    Shapes:
      X_rows: (R, d)
      X_cols: (C, d)
    Returns:
      (R, C) kernel block on the same device.
    """
    device = X_rows.device
    dtype_out = torch.float32

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(autocast_bf16 and X_rows.is_cuda)):
        dot = X_rows @ X_cols.t()                                   # (R, C)
        nrms_r = X_rows.norm(dim=1).clamp_min(1e-12).unsqueeze(1)   # (R, 1)
        nrms_c = X_cols.norm(dim=1).clamp_min(1e-12).unsqueeze(0)   # (1, C)
        cos_theta = (dot / (nrms_r * nrms_c)).clamp(-1.0, 1.0)
        theta = torch.acos(cos_theta)

        pref = (sigma_w ** 2) / (2.0 * math.pi) * (nrms_r * nrms_c)  # (R,C) broadcast
        k = pref * (torch.sin(theta) + (math.pi - theta) * torch.cos(theta))  # (R,C)

    return (sigma_a ** 2) * k.to(dtype_out)


class KernelMatvec:
    """
    Matrix-free multiplication y = (K + λ I) v for the ReLU NNGP kernel,
    with on-the-fly block kernel construction to avoid materializing K.

    Args:
      X_train: (P,d) training inputs
      sigma_w, sigma_a: kernel hyperparameters (match priors)
      lam: ridge parameter (λ = κ^2)
      row_block: number of rows per block
      col_block: number of columns per block (for cache locality)
    """
    def __init__(
        self,
        X_train: Tensor,
        sigma_w: float,
        sigma_a: float,
        lam: float,
        row_block: int = 4096,
        col_block: int = 4096,
        autocast_bf16: bool = True
    ):
        self.X = X_train
        self.sigma_w = sigma_w
        self.sigma_a = sigma_a
        self.lam = lam
        self.row_block = row_block
        self.col_block = col_block
        self.autocast_bf16 = autocast_bf16

    @torch.no_grad()
    def matvec(self, v: Tensor) -> Tensor:
        """
        Compute y = K v + λ v, without forming K.
        v: (P,)
        returns: (P,)
        """
        X = self.X
        P = X.shape[0]
        device = X.device
        v = v.view(-1)
        y = torch.zeros(P, device=device, dtype=torch.float32)

        # Process in row blocks; within each row block, multiply by v via col blocks
        for r0 in range(0, P, self.row_block):
            r1 = min(r0 + self.row_block, P)
            Xr = X[r0:r1]  # (R,d)

            acc = torch.zeros(r1 - r0, device=device, dtype=torch.float32)

            # col blocks for cache locality
            for c0 in range(0, P, self.col_block):
                c1 = min(c0 + self.col_block, P)
                Xc = X[c0:c1]                      # (C,d)
                K_block = relu_arccos_kernel_block(
                    Xr, Xc,
                    sigma_w=self.sigma_w, sigma_a=self.sigma_a,
                    autocast_bf16=self.autocast_bf16
                )                                   # (R,C)
                acc += K_block @ v[c0:c1]          # (R,)

            y[r0:r1] = acc

        # Add ridge
        return y + self.lam * v


@torch.no_grad()
def kernel_times_vector_eval(
    X_eval: Tensor, X_train: Tensor, alpha: Tensor,
    sigma_w: float, sigma_a: float,
    row_block: int = 4096, col_block: int = 4096,
    autocast_bf16: bool = True
) -> Tensor:
    """
    Compute f_eval = K(X_eval, X_train) @ alpha (matrix-free, batched).
    """
    Ne = X_eval.shape[0]
    device = X_eval.device
    f = torch.zeros(Ne, device=device, dtype=torch.float32)
    for r0 in range(0, Ne, row_block):
        r1 = min(r0 + row_block, Ne)
        Xr = X_eval[r0:r1]
        acc = torch.zeros(r1 - r0, device=device, dtype=torch.float32)
        # loop over train columns
        P = X_train.shape[0]
        for c0 in range(0, P, col_block):
            c1 = min(c0 + col_block, P)
            Xc = X_train[c0:c1]
            K_block = relu_arccos_kernel_block(
                Xr, Xc, sigma_w=sigma_w, sigma_a=sigma_a, autocast_bf16=autocast_bf16
            )
            acc += K_block @ alpha[c0:c1]
        f[r0:r1] = acc
    return f


# ----------------------------- Conjugate Gradients -----------------------------
@torch.no_grad()
def conjugate_gradients(
    A_mv, b: Tensor,
    max_iters: int = 300,
    tol: float = 1e-6,
    callback_every: int = 10,
    callback=None
) -> Tuple[Tensor, Dict]:
    """
    Solve A x = b with matrix-free CG, where A_mv is a callable: v -> A v.
    b: (P,)
    Returns:
      x, logs
    """
    device = b.device
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs_old = torch.dot(r, r)

    logs = {"iter": [], "resid_norm": []}

    for it in range(1, max_iters + 1):
        Ap = A_mv(p)
        denom = torch.dot(p, Ap).clamp_min(1e-30)
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)
        resid = torch.sqrt(rs_new).item()

        logs["iter"].append(it)
        logs["resid_norm"].append(resid)

        if callback and (it % max(1, callback_every) == 0 or it == 1 or it == max_iters):
            callback(it, x, resid)

        if resid < tol:
            break

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x, logs


# ----------------------------- Configs -----------------------------
@dataclass
class GPParams:
    sigma_w: float = 1.0     # matches prior w ~ N(0, sigma_w^2 I / d)
    sigma_a: float = 1.0     # output weight prior variance; scales kernel
    # CG / batching knobs
    cg_max_iters: int = 300
    cg_tol: float = 1e-6
    cg_log_every: int = 10
    row_block: int = 4096
    col_block: int = 4096
    eval_every: int = 50      # how often to compute eval metrics during CG (costly)
    autocast_bf16: bool = True


# ----------------------------- Metrics helpers -----------------------------
@torch.no_grad()
def compute_train_metrics(y_tr: Tensor, f_tr: Tensor) -> Dict[str, float]:
    mse = float(((y_tr - f_tr) ** 2).mean().item())
    corr = float((y_tr * f_tr).mean().item())
    y2 = float((y_tr * y_tr).mean().item())
    f2 = float((f_tr * f_tr).mean().item())
    return {"train_mse": mse, "train_corr": corr, "train_y2": y2, "train_f2": f2}


@torch.no_grad()
def compute_eval_metrics(
    X_eval: Tensor, y_eval: Tensor, f_eval: Tensor,
    teacher_sets: List[Tensor]
) -> Dict:
    device = y_eval.device
    P_eval = X_eval.shape[0]

    if len(teacher_sets) > 0:
        C_eval = compute_characters_matrix(X_eval, teacher_sets)   # (P, M)
        m_vec = (C_eval * f_eval.view(-1, 1)).mean(dim=0)          # (M,)
        f_signal = (C_eval * m_vec.view(1, -1)).sum(dim=1)         # (P,)
        r = f_eval - f_signal
        mS_vec = m_vec.detach().cpu().tolist()
        mS_norm2 = float((m_vec * m_vec).sum().item())
        true_coeff = torch.ones(len(teacher_sets), device=device, dtype=torch.float32)
        coeff_mse = float(((m_vec - true_coeff) ** 2).mean().item())
        coeff_mae = float((m_vec - true_coeff).abs().mean().item())
        coeff_sign_acc = float(((m_vec > 0).float().mean()).item())
        noise_norm2 = float((r * r).mean().item())
    else:
        mS_vec, mS_norm2 = [], 0.0
        coeff_mse = coeff_mae = coeff_sign_acc = float('nan')
        noise_norm2 = float('nan')

    mse_total = float(((y_eval - f_eval) ** 2).mean().item())
    var_y = float(((y_eval - y_eval.mean()) ** 2).mean().item())
    yf = float((y_eval * f_eval).mean().item())
    corr_yf = yf / max(var_y, 1e-12) if var_y > 0 else float('nan')

    # χ_AA is not defined for pure GP baseline (no neurons); report zeros to keep schema
    chi_AA_vec = [0.0] * len(teacher_sets)
    chi_AA_norm1 = float(sum(abs(x) for x in chi_AA_vec))
    chi_AA_norm2 = 0.0

    return {
        "eval_mS_vec": mS_vec,
        "eval_mS_norm2": mS_norm2,
        "eval_coeff_mse": coeff_mse,
        "eval_coeff_mae": coeff_mae,
        "eval_coeff_sign_acc": coeff_sign_acc,
        "eval_noise_norm2": noise_norm2,
        "eval_mse_total": mse_total,
        "eval_corr_yf": corr_yf,
        "chi_AA_vec": chi_AA_vec,
        "chi_AA_norm1": chi_AA_norm1,
        "chi_AA_norm2": chi_AA_norm2,
    }


# ----------------------------- One run (single P, kappa) -----------------------------
@torch.no_grad()
def run_gp_relu_composite(
    X_train: Tensor, y_train: Tensor,
    X_eval: Tensor, y_eval: Tensor,
    kappa: float,
    gp: GPParams,
    teacher_sets: List[Tensor],
    log_dir: str,
    run_tag: str
) -> Dict:
    os.makedirs(log_dir, exist_ok=True)

    device = X_train.device
    P = X_train.shape[0]
    lam = float(kappa ** 2)  # λ = κ^2
    # Build matrix-free (K + λ I) matvec
    Aop = KernelMatvec(
        X_train, sigma_w=gp.sigma_w, sigma_a=gp.sigma_a, lam=lam,
        row_block=gp.row_block, col_block=gp.col_block, autocast_bf16=gp.autocast_bf16
    )

    traj = {
        "iter": [], "time_s": [],
        "train_mse": [], "train_corr": [], "train_y2": [], "train_f2": [],
        "eval_mS_vec": [], "eval_mS_norm2": [],
        "eval_coeff_mse": [], "eval_coeff_mae": [], "eval_coeff_sign_acc": [],
        "eval_noise_norm2": [], "eval_mse_total": [], "eval_corr_yf": [],
        "chi_AA_vec": [], "chi_AA_norm1": [], "chi_AA_norm2": []
    }
    t0 = time.time()

    # callback to log metrics during CG
    def cb(it: int, alpha: Tensor, resid: float):
        # train predictions via one matvec: f_tr = K alpha = (A - λI) alpha
        f_tr = Aop.matvec(alpha) - lam * alpha
        tm = compute_train_metrics(y_train, f_tr)
        # (Optional) eval every gp.eval_every its (costly)
        if (it % max(1, gp.eval_every) == 0) or it == 1:
            f_ev = kernel_times_vector_eval(
                X_eval, X_train, alpha,
                sigma_w=gp.sigma_w, sigma_a=gp.sigma_a,
                row_block=gp.row_block, col_block=gp.col_block, autocast_bf16=gp.autocast_bf16
            )
            em = compute_eval_metrics(X_eval, y_eval, f_ev, teacher_sets)
        else:
            # keep last eval metrics or fill NaN on first call
            if len(traj["eval_mse_total"]) > 0:
                em = {k: traj[k][-1] for k in [
                    "eval_mS_vec", "eval_mS_norm2",
                    "eval_coeff_mse", "eval_coeff_mae", "eval_coeff_sign_acc",
                    "eval_noise_norm2", "eval_mse_total", "eval_corr_yf",
                    "chi_AA_vec", "chi_AA_norm1", "chi_AA_norm2"
                ]}
            else:
                em = {
                    "eval_mS_vec": [], "eval_mS_norm2": float('nan'),
                    "eval_coeff_mse": float('nan'), "eval_coeff_mae": float('nan'),
                    "eval_coeff_sign_acc": float('nan'),
                    "eval_noise_norm2": float('nan'), "eval_mse_total": float('nan'),
                    "eval_corr_yf": float('nan'),
                    "chi_AA_vec": [], "chi_AA_norm1": 0.0, "chi_AA_norm2": 0.0
                }

        # append
        traj["iter"].append(int(it))
        traj["time_s"].append(float(time.time() - t0))
        traj["train_mse"].append(tm["train_mse"])
        traj["train_corr"].append(tm["train_corr"])
        traj["train_y2"].append(tm["train_y2"])
        traj["train_f2"].append(tm["train_f2"])
        traj["eval_mS_vec"].append(em["eval_mS_vec"])
        traj["eval_mS_norm2"].append(em["eval_mS_norm2"])
        traj["eval_coeff_mse"].append(em["eval_coeff_mse"])
        traj["eval_coeff_mae"].append(em["eval_coeff_mae"])
        traj["eval_coeff_sign_acc"].append(em["eval_coeff_sign_acc"])
        traj["eval_noise_norm2"].append(em["eval_noise_norm2"])
        traj["eval_mse_total"].append(em["eval_mse_total"])
        traj["eval_corr_yf"].append(em["eval_corr_yf"])
        traj["chi_AA_vec"].append(em["chi_AA_vec"])
        traj["chi_AA_norm1"].append(em["chi_AA_norm1"])
        traj["chi_AA_norm2"].append(em["chi_AA_norm2"])

        # Print lightweight status
        print(json.dumps({
            "iter": it,
            "resid": resid,
            "train_mse": tm["train_mse"],
            "eval_mse_total": em["eval_mse_total"],
            "eval_mS_norm2": em["eval_mS_norm2"],
            "elapsed_s": round(traj["time_s"][-1], 2),
            "P": int(P),
        }))

    # Right-hand side b = y
    b = y_train.view(-1).to(torch.float32)
    alpha, cg_logs = conjugate_gradients(
        Aop.matvec, b,
        max_iters=gp.cg_max_iters,
        tol=gp.cg_tol,
        callback_every=gp.cg_log_every,
        callback=cb
    )

    # Final predictions
    f_train = Aop.matvec(alpha) - lam * alpha
    f_eval = kernel_times_vector_eval(
        X_eval, X_train, alpha,
        sigma_w=gp.sigma_w, sigma_a=gp.sigma_a,
        row_block=gp.row_block, col_block=gp.col_block, autocast_bf16=gp.autocast_bf16
    )

    # Final metrics
    tm = compute_train_metrics(y_train, f_train)
    em = compute_eval_metrics(X_eval, y_eval, f_eval, teacher_sets)

    summary = {
        "P_train": int(X_train.shape[0]),
        "P_eval": int(X_eval.shape[0]),
        "d": int(X_train.shape[1]),
        "kappa": float(kappa),
        "sigma_w": float(gp.sigma_w),
        "sigma_a": float(gp.sigma_a),
        "cg_max_iters": int(gp.cg_max_iters),
        "cg_tol": float(gp.cg_tol),
        "row_block": int(gp.row_block),
        "col_block": int(gp.col_block),
        "autocast_bf16": bool(gp.autocast_bf16),
        "final_train": tm,
        "final_eval": em,
        "cg_iters_ran": int(len(cg_logs["iter"])),
        "resid_norm_last": float(cg_logs["resid_norm"][-1]) if len(cg_logs["resid_norm"])>0 else None,
        "teacher_sets": [s.cpu().tolist() for s in teacher_sets],
        "note_chi_AA": "Not applicable for GP baseline; reported zeros."
    }

    out = {
        "summary": summary,
        "traj": traj,
        "cg_logs": cg_logs,
        "config": {
            "gp": vars(gp),
            "kappa": float(kappa),
        }
    }

    # Save
    tag = run_tag or time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(
        log_dir,
        f"nngp_relu_composite_{tag}_P{X_train.shape[0]}_Peval{X_eval.shape[0]}_kap{float(kappa):.3e}.json"
    )
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {path}")
    return out


# ----------------------------- Entrypoint / sweep -----------------------------
if __name__ == "__main__":
    set_seed(42)
    _ = check_gpu()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---- Problem setup ----
    d = 35
    # Example composite: two characters, one of size 4 and one of size 8
    teacher_spec = "{0,1,2,3}"
    sets_idx_lists = parse_composite_spec(teacher_spec)
    teacher_sets = [torch.tensor(s, device=device, dtype=torch.long) for s in sets_idx_lists]
    M_components = len(teacher_sets)

    # Eval set (fixed across runs)
    P_eval = 50000
    X_eval, y_eval, _ = generate_composite_data(P_eval, d, teacher_sets, device)

    # ---- GP / CG knobs ----
    gp = GPParams(
        sigma_w=1.0,           # matches w ~ N(0, σ_w^2 I / d)
        sigma_a=1.0,           # kernel scale
        cg_max_iters=300,
        cg_tol=1e-6,
        cg_log_every=10,       # log every 10 CG iterations
        row_block=4096,
        col_block=4096,
        eval_every=50,         # compute eval metrics every 50 CG its
        autocast_bf16=True
    )

    # ---- Sweep lists (edit here) ----
    P_TRAIN_LIST = [10, 100, 500, 750, 1000, 2500, 5000, 7500, 10000, 20000]     # training sizes
    KAPPA_LIST   = [5e-4]             # noise levels

    results_dir = "/home/goring/mean_field_langevin/MCMC_composite/results/2608_NNGP_d35_k4_1"
    os.makedirs(results_dir, exist_ok=True)

    # ---- Run sweep ----
    index: Dict[str, List[Dict]] = {"runs": []}
    for P_train in P_TRAIN_LIST:
        X_train, y_train, _ = generate_composite_data(P_train, d, teacher_sets, device)
        for kappa in KAPPA_LIST:
            print(f"\n=== NNGP ReLU GP: P_train={P_train}, kappa={kappa:.3e} ===")
            run_tag = f"P{P_train}_kap{float(kappa):.3e}_M{M_components}"
            out = run_gp_relu_composite(
                X_train, y_train, X_eval, y_eval,
                kappa=kappa, gp=gp, teacher_sets=teacher_sets,
                log_dir=results_dir, run_tag=run_tag
            )
            index["runs"].append({
                "P_train": int(P_train),
                "P_eval": int(P_eval),
                "kappa": float(kappa),
                "d": d,
                "M_components": M_components,
                "path": os.path.join(
                    results_dir,
                    f"nngp_relu_composite_{run_tag}_P{P_train}_Peval{P_eval}_kap{float(kappa):.3e}.json"
                ),
                "summary": out["summary"]
            })

    # Save sweep index
    index_path = os.path.join(results_dir, "sweep_index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"\n[sweep saved] {index_path}")
