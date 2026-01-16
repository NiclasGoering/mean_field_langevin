# nonlazy_sgld_with_regression.py
# ------------------------------------------------------------
# A minimal, paper-consistent (van Meegen & Sompolinsky, 2025)
# two-layer non-lazy network trained with SGLD. Supports:
#   - Parity classification (multi-output)
#   - Parity regression (scalar: sum of parities)
#   - Parity regression (multi-output)
#   - Toy orthogonal classification (paper-style)
#
# Key ingredients:
#   - Non-lazy scaling: f(x) = (1/N) * sum_i a_i^T phi(w_i · x)
#   - Exact MSE gradients, no diagonal/mean-field shortcuts
#   - Rescaled temperature T (O(1) as width grows) in both drift & noise
#   - Readout prior variance sigma_a^2 ∝ 1/P (L=1)
#
# Toggle task/mode at the bottom in `__main__`.
# ------------------------------------------------------------

import os, math, json, time, random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

# ----------------------------- utils -----------------------------

def set_seed(seed: int = 12345):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def activation(z: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "relu": return F.relu(z)
    if kind == "linear": return z
    if kind == "sigmoid_erf":
        c = math.sqrt(math.pi) / 2.0  # paper’s constant
        return 0.5 * (1.0 + torch.erf(c * z))
    if kind == "tanh": return torch.tanh(z)
    raise ValueError(f"Unknown activation: {kind}")

def act_prime(z: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "relu": return (z > 0).to(z.dtype)
    if kind == "linear": return torch.ones_like(z)
    if kind == "sigmoid_erf":
        c = math.sqrt(math.pi) / 2.0
        return (c / math.sqrt(math.pi)) * torch.exp(- (c * z) ** 2)
    if kind == "tanh": return 1.0 - torch.tanh(z) ** 2
    raise ValueError(f"Unknown activation: {kind}")

def parse_sets(spec: str) -> List[List[int]]:
    import re
    blocks = re.findall(r"\{([^}]*)\}", spec)
    out = []
    for s in blocks:
        toks = [t.strip() for t in s.split(",") if t.strip()!=""]
        out.append(sorted(map(int, toks)))
    if not out: raise ValueError("bad teacher spec (e.g. '{0,1,2,3}{5,6,7,8}')")
    return out

def parity_character(X_pm1: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """
    χ_S(x) = ∏_{j ∈ S} x_j for ±1-coded inputs.
    X_pm1: (P,d) or (E,P,d), dtype float, values in {-1,+1}
    S: LongTensor of indices
    """
    if X_pm1.dim() == 2:
        if S.numel() == 0: return torch.ones(X_pm1.shape[0], device=X_pm1.device, dtype=X_pm1.dtype)
        return X_pm1[:, S].prod(dim=1).to(X_pm1.dtype)
    elif X_pm1.dim() == 3:
        if S.numel() == 0:
            return torch.ones(X_pm1.shape[0], X_pm1.shape[1], device=X_pm1.device, dtype=X_pm1.dtype)
        return X_pm1[:, :, S].prod(dim=2).to(X_pm1.dtype)
    else:
        raise ValueError("X must be (P,d) or (E,P,d)")

# ----------------------------- generators -----------------------------

def generate_parity_classification(
    P: int, d: int, sets: List[torch.Tensor], E: int,
    data_seeds: List[int], device, dtype,
    y_pos: float = 1.0, y_neg: float = -0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Multi-output classification (MSE with ± targets):
      - m = len(sets)
      - Y[e, mu, r] in {y_pos, y_neg} depending on parity of set r
    """
    if len(data_seeds) != E: raise ValueError("len(data_seeds) must equal E")
    m = max(1, len(sets))
    Xs, Ys = [], []
    for e in range(E):
        g = torch.Generator(device=device).manual_seed(int(data_seeds[e]))
        Xe = (torch.randint(0,2,(P,d),generator=g,device=device,dtype=torch.int8).to(dtype) * 2.0 - 1.0)
        cols = [parity_character(Xe, S) for S in sets] if m > 0 else [torch.ones(P, device=device, dtype=dtype)]
        C = torch.stack(cols, dim=1)  # ±1
        Ye = torch.full_like(C, y_neg)
        Ye[C > 0] = y_pos
        Xs.append(Xe); Ys.append(Ye)
    return torch.stack(Xs, dim=0), torch.stack(Ys, dim=0)  # (E,P,d), (E,P,m)

def generate_parity_regression_scalar(
    P: int, d: int, sets: List[torch.Tensor], E: int,
    data_seeds: List[int], device, dtype, normalize: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Regression (scalar): y = sum_r χ_{S_r}(x), shape Y: (E,P,1).
    If normalize=True, divide by sqrt(m) to keep targets O(1).
    """
    if len(data_seeds) != E: raise ValueError("len(data_seeds) must equal E")
    m = max(1, len(sets))
    scale = 1.0 / math.sqrt(m) if (normalize and m > 0) else 1.0
    Xs, Ys = [], []
    for e in range(E):
        g = torch.Generator(device=device).manual_seed(int(data_seeds[e]))
        Xe = (torch.randint(0,2,(P,d),generator=g,device=device,dtype=torch.int8).to(dtype) * 2.0 - 1.0)
        if m == 0:
            Ye = torch.zeros(P, 1, device=device, dtype=dtype)
        else:
            cols = [parity_character(Xe, S) for S in sets]  # ±1
            ysum = torch.stack(cols, dim=1).sum(dim=1, keepdim=True)  # (P,1) in [-m, m]
            Ye = scale * ysum
        Xs.append(Xe); Ys.append(Ye)
    return torch.stack(Xs, dim=0), torch.stack(Ys, dim=0)  # (E,P,d), (E,P,1)

def generate_parity_regression_multi(
    P: int, d: int, sets: List[torch.Tensor], E: int,
    data_seeds: List[int], device, dtype, normalize: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Regression (multi-output): y_r = χ_{S_r}(x), Y: (E,P,m).
    If normalize=True, you can scale each output (usually not needed).
    """
    if len(data_seeds) != E: raise ValueError("len(data_seeds) must equal E")
    m = max(1, len(sets))
    Xs, Ys = [], []
    for e in range(E):
        g = torch.Generator(device=device).manual_seed(int(data_seeds[e]))
        Xe = (torch.randint(0,2,(P,d),generator=g,device=device,dtype=torch.int8).to(dtype) * 2.0 - 1.0)
        if m == 0:
            Ye = torch.zeros(P, 1, device=device, dtype=dtype)
        else:
            cols = [parity_character(Xe, S) for S in sets]  # ±1
            Ye = torch.stack(cols, dim=1)                   # (P,m)
            if normalize:
                Ye = Ye / math.sqrt(m)
        Xs.append(Xe); Ys.append(Ye)
    return torch.stack(Xs, dim=0), torch.stack(Ys, dim=0)

def generate_orthogonal_toy(
    P: int, d: int, m: int, E: int, data_seeds: List[int],
    device, dtype, y_pos: float = 1.0, y_neg: float = -0.5
):
    """
    Paper-style toy classification with ~orthogonal inputs and one-hot style targets.
    Y in {y_pos, y_neg} with a single y_pos per sample across m outputs.
    """
    if len(data_seeds) != E: raise ValueError("len(data_seeds) must equal E")
    Xs, Ys = [], []
    for e in range(E):
        g = torch.Generator(device=device).manual_seed(int(data_seeds[e]))
        M = torch.randn(d, P, generator=g, device=device, dtype=dtype)
        Q, _ = torch.linalg.qr(M, mode="reduced")  # (d,P)
        Xe = (Q.t()).contiguous()                  # (P,d)
        y_idx = torch.randint(0, m, (P,), generator=g, device=device)
        Ye = torch.full((P, m), y_neg, device=device, dtype=dtype)
        Ye[torch.arange(P, device=device), y_idx] = y_pos
        Xs.append(Xe); Ys.append(Ye)
    return torch.stack(Xs, dim=0), torch.stack(Ys, dim=0)

# ----------------------------- config -----------------------------

@dataclass
class Model:
    d: int = 35
    B: int = 512            # width (hidden units)
    N: int = 512            # width used for scaling (should equal B)
    sigma_w: float = 1.0
    act: str = "relu"       # "relu" | "linear" | "sigmoid_erf" | "tanh"

@dataclass
class Algo:
    outer_steps: int = 200_000
    step_size: float = 3e-3
    K: int = 1                       # SGLD inner steps per outer step
    log_every: int = 1_000
    eval_every: int = 5_000
    P_chunk_train: Optional[int] = 262_144
    use_float64: bool = False

    # SGLD / safety
    grad_clip_norm: Optional[float] = None
    kill_nan_particles: bool = True
    max_abs_w: Optional[float] = None
    max_abs_a: Optional[float] = None

    # Early stop on held-out eval
    early_stop_enabled: bool = True
    early_stop_test_mse_threshold: float = 0.01
    test_mse_check_every: int = 5_000

# ----------------------------- core sampler -----------------------------

class RSCavityExplicitMulti:
    """
    Two-layer (single hidden layer) non-lazy network with SGLD:

        f(x) = (1/N) * Φ(x; W) @ a

    where Φ_μi = φ(z_{iμ}), z_{iμ} = w_i · x_μ, and a ∈ ℝ^{B×m}.
    """

    def __init__(self, mdl: Model, algo: Algo, T_rescaled: float,
                 device: torch.device, E: int, seeds_params: List[int]):
        self.mdl, self.algo = mdl, algo
        self.device = device
        self.E = int(E)
        self.seeds_params = [int(s) for s in seeds_params]
        assert len(self.seeds_params) == E
        assert mdl.B == mdl.N, "Set B==N for clean 1/N non-lazy scaling."

        self.dtype = torch.float64 if algo.use_float64 else torch.float32
        self.s = 1.0 / float(mdl.N)        # non-lazy output scale
        self.T = float(T_rescaled)         # rescaled temperature (paper)

        # params will be initialized once P and m are known
        self.W = None   # (E,B,d)
        self.a = None   # (E,B,m)
        self._sigma_a2 = None

        try:
            torch.set_float32_matmul_precision('high')
            torch.backends.cuda.matmul.allow_tf32 = not algo.use_float64
            torch.backends.cudnn.allow_tf32 = not algo.use_float64
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    def _init_params(self, P: int, m: int):
        """Initialize W and a with sigma_a^2 ∝ 1/P (L=1)."""
        g_base = 10_000_003
        W_list, a_list = [], []

        sigma_a2 = 1.0 / float(P)  # works for linear, relu, sigmoid_erf when L=1
        sigma_a = math.sqrt(sigma_a2)

        for e in range(self.E):
            g = torch.Generator(device=self.device).manual_seed(g_base + self.seeds_params[e])
            We = torch.randn(self.mdl.B, self.mdl.d, generator=g, device=self.device, dtype=self.dtype) \
                 * (self.mdl.sigma_w / math.sqrt(self.mdl.d))
            ae = torch.randn(self.mdl.B, m, generator=g, device=self.device, dtype=self.dtype) * sigma_a
            W_list.append(We); a_list.append(ae)
        self.W = torch.stack(W_list, dim=0)  # (E,B,d)
        self.a = torch.stack(a_list, dim=0)  # (E,B,m)
        self._sigma_a2 = sigma_a2

    @torch.no_grad()
    def _sgld_step(self, X: torch.Tensor, Y: torch.Tensor, P_total: int):
        """One SGLD step with exact MSE gradients."""
        Eexp, P, d = X.shape
        _, _, m = Y.shape
        assert Eexp == self.E and d == self.mdl.d

        eta = float(self.algo.step_size)
        invP = 1.0 / float(P_total)
        s = self.s
        sigw2 = self.mdl.sigma_w ** 2
        siga2 = self._sigma_a2

        grad_W = torch.zeros_like(self.W)
        grad_a = torch.zeros_like(self.a)

        step = self.algo.P_chunk_train or P
        for start in range(0, P, step):
            n = min(step, P - start)
            Xc = X[:, start:start+n, :]    # (E,n,d)
            Yc = Y[:, start:start+n, :]    # (E,n,m)

            Z = torch.bmm(Xc, self.W.transpose(1,2))     # (E,n,B)
            Phi = activation(Z, self.mdl.act)            # (E,n,B)
            dPhi = act_prime(Z, self.mdl.act)            # (E,n,B)

            fa = torch.bmm(Phi, self.a)                  # (E,n,m)
            f = s * fa
            R = Yc - f                                    # (E,n,m)

            # grad wrt a:  -(2 s / P) * Phi^T R   -> (E,B,m)
            grad_a += - (2.0 * s * invP) * torch.bmm(Phi.transpose(1,2), R)

            # grad wrt W:  -(2 / P) * sum_{μ,r} R_{μr} * (s a_{ir}) * φ'(z_{iμ}) * x_μ
            a_scaled = s * self.a                        # (E,B,m)
            grad_W += - (2.0 * invP) * torch.einsum('enr,ebr,enb,end->ebd', R, a_scaled, dPhi, Xc)

        # Gaussian priors (rescaled temperature)
        grad_W += (self.T / sigw2) * self.W
        grad_a += (self.T / siga2) * self.a

        if self.algo.grad_clip_norm is not None:
            gW2 = (grad_W * grad_W).sum(dim=(1,2), keepdim=True)
            gA2 = (grad_a * grad_a).sum(dim=(1,2), keepdim=True)
            gN = torch.sqrt(gW2 + gA2) + 1e-12
            scale = torch.clamp(self.algo.grad_clip_norm / gN, max=1.0)
            grad_W *= scale; grad_a *= scale

        # SGLD update
        self.W.add_(grad_W, alpha=-eta)
        self.a.add_(grad_a, alpha=-eta)
        nstd = math.sqrt(2.0 * self.T * eta)
        self.W.add_(torch.randn_like(self.W, dtype=self.dtype), alpha=nstd)
        self.a.add_(torch.randn_like(self.a, dtype=self.dtype), alpha=nstd)

        if self.algo.kill_nan_particles:
            badW = ~torch.isfinite(self.W).all(dim=2)
            bada = ~torch.isfinite(self.a).all(dim=2)
            bad = badW | bada
            if bad.any():
                for e in range(self.E):
                    msk = bad[e]
                    if msk.any():
                        nbad = int(msk.sum().item())
                        g = torch.Generator(device=self.device).manual_seed(7777 + e)
                        self.W[e, msk] = torch.randn(nbad, d, generator=g, device=self.device, dtype=self.dtype) * (self.mdl.sigma_w / math.sqrt(d))
                        self.a[e, msk] = torch.randn(nbad, self.a.shape[2], generator=g, device=self.device, dtype=self.dtype) * math.sqrt(self._sigma_a2)

        if self.algo.max_abs_w is not None: self.W.clamp_(-self.algo.max_abs_w, self.algo.max_abs_w)
        if self.algo.max_abs_a is not None: self.a.clamp_(-self.algo.max_abs_a, self.algo.max_abs_a)

    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        Z = torch.bmm(X, self.W.transpose(1,2))     # (E,P,B)
        Phi = activation(Z, self.mdl.act)           # (E,P,B)
        f = self.s * torch.bmm(Phi, self.a)         # (E,P,m)
        return f

    @torch.no_grad()
    def run(self, X: torch.Tensor, Y: torch.Tensor, out_dir: str, tag: str="",
            X_eval: Optional[torch.Tensor]=None, Y_eval: Optional[torch.Tensor]=None):
        os.makedirs(out_dir, exist_ok=True)
        X = X.to(self.dtype); Y = Y.to(self.dtype)
        Eexp, P, d = X.shape; _, _, m = Y.shape
        assert d == self.mdl.d and Eexp == self.E

        self._init_params(P, m)

        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"nonlazy_sgld_{tag or ts}_P{P}_E{self.E}_N{self.mdl.N}_B{self.mdl.B}_act{self.mdl.act}_T{self.T:.2e}_m{m}_{'f64' if self.algo.use_float64 else 'f32'}.json"
        save_path = os.path.join(out_dir, fname)

        hist: Dict[str, Any] = {"iter": [], "train_mse": [], "test_mse_small": [], "elapsed_s": []}
        t0 = time.time()

        for it in range(1, self.algo.outer_steps+1):
            for _ in range(self.algo.K):
                self._sgld_step(X, Y, P_total=P)

            if (it % self.algo.log_every) == 0:
                f_train = self.predict(X)
                train_mse = float(((Y - f_train)**2).mean().item())
                hist["iter"].append(it)
                hist["train_mse"].append(train_mse)
                hist["elapsed_s"].append(round(time.time()-t0, 2))

            do_eval = (X_eval is not None) and (Y_eval is not None)
            if do_eval and (it % self.algo.eval_every) == 0:
                f_eval = self.predict(X_eval)
                test_mse_small = float(((Y_eval.to(self.dtype) - f_eval)**2).mean().item())
                hist["test_mse_small"].append(test_mse_small)

            if do_eval and self.algo.early_stop_enabled and (it % max(1, self.algo.test_mse_check_every) == 0):
                f_eval = self.predict(X_eval)
                test_mse_small = float(((Y_eval.to(self.dtype) - f_eval)**2).mean().item())
                if test_mse_small < self.algo.early_stop_test_mse_threshold:
                    payload = {
                        "summary": {
                            "iter": it,
                            "train_mse": float(((Y - self.predict(X))**2).mean().item()),
                            "test_mse_small": test_mse_small,
                            "E": self.E, "P": P, "m": m,
                        },
                        "traj": hist,
                        "config": {
                            "model": self.mdl.__dict__, "algo": self.algo.__dict__,
                            "T_rescaled": self.T, "scale": self.s,
                            "sigma_a2": self._sigma_a2,
                        }
                    }
                    with open(save_path, "w") as f:
                        json.dump(payload, f, indent=2)
                    print(f"[EARLY STOP] it={it} test_mse_small={test_mse_small:.6f} < {self.algo.early_stop_test_mse_threshold}")
                    return {"path": save_path, "traj": hist}

        # final save
        payload = {
            "summary": {
                "iter": self.algo.outer_steps,
                "train_mse": float(((Y - self.predict(X))**2).mean().item()),
                "test_mse_small": float(((Y_eval.to(self.dtype) - self.predict(X_eval))**2).mean().item()) if (X_eval is not None and Y_eval is not None) else float('nan'),
                "E": self.E, "P": P, "m": m,
            },
            "traj": hist,
            "config": {
                "model": self.mdl.__dict__, "algo": self.algo.__dict__,
                "T_rescaled": self.T, "scale": self.s,
                "sigma_a2": self._sigma_a2,
            }
        }
        with open(save_path, "w") as f:
            json.dump(payload, f, indent=2)
        return {"path": save_path, "traj": hist}

# ----------------------------- experiment helpers -----------------------------

def build_experiment_grid(P_train_list: List[int], kappa_list: List[float], num_exp: int, base_seed: int):
    exps = []
    idx = 0
    for P in P_train_list:
        for k in kappa_list:
            data_seeds = [base_seed + idx*100000 + e for e in range(num_exp)]
            param_seeds = [base_seed + idx*100000 + 50000 + e for e in range(num_exp)]
            exps.append({
                'P': int(P),
                'kappa': float(k),       # we use T_rescaled = 2 * kappa^2
                'data_seeds': data_seeds,
                'param_seeds': param_seeds,
                'E': num_exp,
            })
            idx += 1
    return exps

def shard_experiments(exps: List[Dict[str, Any]], num_devices: int, strategy: str = "balance_P"):
    if num_devices <= 1:
        return [exps]
    if strategy == "balance_P":
        exps_sorted = sorted(exps, key=lambda e: e['P'], reverse=True)
        loads = [0 for _ in range(num_devices)]
        shards = [[] for _ in range(num_devices)]
        for e in exps_sorted:
            i = int(np.argmin(loads)); shards[i].append(e); loads[i] += e['P']
        return shards
    else:
        shards = [[] for _ in range(num_devices)]
        for i, e in enumerate(exps):
            shards[i % num_devices].append(e)
        return shards

# ----------------------------- worker -----------------------------

def worker_process(dev_id: int, shard: List[Dict[str, Any]],
                   mdl_dict: Dict[str, Any], algo_dict: Dict[str, Any],
                   out_dir: str, teacher_sets_spec: str,
                   use_float64: bool, task_kind: str):
    """
    task_kind ∈ {
      'parity_classification',
      'parity_regression_scalar',
      'parity_regression_multi',
      'toy_orthogonal'
    }
    """
    if torch.cuda.is_available():
        torch.cuda.set_device(dev_id)
        device = torch.device(f"cuda:{dev_id}")
    else:
        device = torch.device("cpu")

    mdl = Model(**mdl_dict)
    algo = Algo(**{**algo_dict, "use_float64": use_float64})
    dtype = torch.float64 if use_float64 else torch.float32

    sets_idx = parse_sets(teacher_sets_spec)
    sets = [torch.tensor(s, device=device, dtype=torch.long) for s in sets_idx]
    m_parity = max(1, len(sets))

    for econf in shard:
        P = econf['P']; kappa = econf['kappa']; E = econf['E']
        data_seeds = econf['data_seeds']; param_seeds = econf['param_seeds']

        if task_kind == "parity_classification":
            X, Y = generate_parity_classification(P, mdl.d, sets, E, data_seeds, device, dtype)
            # small held-out eval of the same type
            P_eval = max(2048, min(32768, P // 4 if P >= 4 else P))
            Xe, Ye = generate_parity_classification(P_eval, mdl.d, sets, E, [s+999 for s in data_seeds], device, dtype)

        elif task_kind == "parity_regression_scalar":
            X, Y = generate_parity_regression_scalar(P, mdl.d, sets, E, data_seeds, device, dtype, normalize=True)
            P_eval = max(2048, min(32768, P // 4 if P >= 4 else P))
            Xe, Ye = generate_parity_regression_scalar(P_eval, mdl.d, sets, E, [s+999 for s in data_seeds], device, dtype, normalize=True)

        elif task_kind == "parity_regression_multi":
            X, Y = generate_parity_regression_multi(P, mdl.d, sets, E, data_seeds, device, dtype, normalize=False)
            P_eval = max(2048, min(32768, P // 4 if P >= 4 else P))
            Xe, Ye = generate_parity_regression_multi(P_eval, mdl.d, sets, E, [s+999 for s in data_seeds], device, dtype, normalize=False)

        elif task_kind == "toy_orthogonal":
            m_toy = 3
            X, Y = generate_orthogonal_toy(P, mdl.d, m_toy, E, data_seeds, device, dtype)
            P_eval = max(1024, min(4096, P // 5 if P >= 5 else P))
            Xe, Ye = generate_orthogonal_toy(P_eval, mdl.d, m_toy, E, [s+999 for s in data_seeds], device, dtype)
        else:
            raise ValueError("task_kind must be one of the documented options.")

        sampler = RSCavityExplicitMulti(
            mdl, algo, T_rescaled=(2.0 * (kappa ** 2)),
            device=device, E=E, seeds_params=param_seeds
        )
        tag = f"{task_kind}_P{P}_kap{kappa:.3e}_act{mdl.act}"
        print(f"\n===== RUN start: task={task_kind}, P={P}, kappa={kappa:.6g}, E={E}, device={device}, dtype={'float64' if algo.use_float64 else 'float32'} =====")
        result = sampler.run(X, Y, out_dir, tag=tag, X_eval=Xe, Y_eval=Ye)
        print(f"===== RUN done: saved -> {result['path']} =====\n")

        del sampler, X, Y, Xe, Ye
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ----------------------------- main -----------------------------

if __name__ == "__main__":
    set_seed(42)

    # ---------- Choose the task ----------
    # Options:
    #   "parity_classification"
    #   "parity_regression_scalar"  (scalar y = sum of parities, normalized)
    #   "parity_regression_multi"   (vector y with one output per set)
    #   "toy_orthogonal"
    task_kind = "parity_regression_scalar"

    # Teacher (choose which parity sets are used)
    teacher_sets_spec = "{0,1,2,3}"  # two 4-bit parities as an example

    d = 35
    use_float64 = False

    # IMPORTANT: consistent with non-lazy scaling -> set B == N
    mdl = Model(d=d, B=512, N=512, sigma_w=1.0, act="relu")  # "linear" or "relu" are good for regression

    algo = Algo(
        outer_steps=200_000,
        step_size=3e-3,
        K=1,
        log_every=1_000,
        eval_every=5_000,
        P_chunk_train=262_144,
        use_float64=use_float64,
        grad_clip_norm=None,
        kill_nan_particles=True,
        early_stop_enabled=True,
        # For regression, you may want to relax this threshold if your targets are not normalized
        early_stop_test_mse_threshold=0.01,
        test_mse_check_every=5_000,
    )

    out_dir = "/home/goring/mean_field_langevin/results_vms/test2"
    os.makedirs(out_dir, exist_ok=True)

    # Grid and temperature (T_rescaled = 2 * kappa^2)
    P_train_list = [500, 1000, 2133, 5000]   # adjust as desired
    kappa_list   = [5e-3]                    # keep small for stable sampling
    num_exp = 3
    base_seed = 123456

    exps = build_experiment_grid(P_train_list, kappa_list, num_exp, base_seed)
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    shards = shard_experiments(exps, num_devices, strategy="round_robin")

    if num_devices > 1:
        ctx = mp.get_context("spawn")
        procs = []
        for dev_id in range(num_devices):
            p = ctx.Process(
                target=worker_process,
                args=(dev_id, shards[dev_id],
                      mdl.__dict__, algo.__dict__,
                      out_dir, teacher_sets_spec, algo.use_float64, task_kind),
            )
            p.start(); procs.append(p)
        for p in procs:
            p.join()
    else:
        worker_process(0, shards[0], mdl.__dict__, algo.__dict__, out_dir,
                       teacher_sets_spec, algo.use_float64, task_kind)
