import json
import time
from pathlib import Path
import re
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from filelock import FileLock
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.nn.functional as F

# Import autocast for mixed precision
from torch.cuda.amp import autocast


# -----------------------------
# Walsh expression utilities
# -----------------------------

def _expand_index_tokens(indices_str):
    """
    Expand tokens like '0,1,2,3' and '0-3' (inclusive) into a sorted list of unique ints.
    Spaces are allowed. An empty string means the empty set {}.
    """
    indices_str = indices_str.strip()
    if indices_str == "" or indices_str == "{}":
        return []  # empty set

    tokens = re.split(r'[,\s]+', indices_str.strip())
    out = []
    for t in tokens:
        if not t:
            continue
        if '-' in t:
            a, b = t.split('-', 1)
            a = int(a.strip())
            b = int(b.strip())
            if a <= b:
                out.extend(range(a, b + 1))
            else:
                out.extend(range(a, b - 1, -1))  # tolerate reversed ranges like "3-0"
        else:
            out.append(int(t))
    # unique + sorted canonical representation
    return sorted(set(out))


def parse_walsh_expression(expr):
    r"""
    Parse expressions like:
      "{0-3} + {0-8}"
      "2*{0,2,5} - 0.5*{1-4} + 0.1"
      "-{1} + { }"   (empty set means constant 1)
    Returns:
      terms: list of (coef: float, indices: List[int])   # indices is the subset S for the parity χ_S(x)=∏_{i∈S} x_i
      const: float
    Notes:
      - Whitespace is ignored.
      - '*' between coefficient and braces is optional.
      - A leading sign is allowed.
      - Standalone constants are allowed (they add to 'const').
    """
    s = expr.strip()
    if not s:
        raise ValueError("Empty target expression.")

    # Regex that matches either a brace-term with optional coefficient and sign, OR a standalone constant.
    # Examples matched:
    #   +{0-3}
    #   - 2.5 * {1,2,4}
    #   3{0,5}
    #   +0.1
    #   - 7.2
    term_re = re.compile(r"""
        (?P<sign>[+\-]?)\s*
        (?:
            (?:(?P<coef>\d+(?:\.\d+)?)(?:\s*\*)?\s*)?
            \{(?P<indices>[^\}]*)\}
          |
            (?P<const>\d+(?:\.\d+)?)
        )
    """, re.VERBOSE)

    terms = []
    const = 0.0
    pos = 0
    for m in term_re.finditer(s):
        if m.start() != pos:
            # Unparsed gap suggests invalid syntax
            gap = s[pos:m.start()].strip()
            if gap:
                raise ValueError(f"Cannot parse target expression near: '{gap}'")
        pos = m.end()

        sign = -1.0 if (m.group("sign") == "-") else 1.0
        if m.group("indices") is not None:
            coef = float(m.group("coef")) if m.group("coef") is not None else 1.0
            indices = _expand_index_tokens(m.group("indices"))
            terms.append((sign * coef, indices))
        else:
            # standalone constant
            c = float(m.group("const"))
            const += sign * c

    if pos != len(s):
        trailing = s[pos:].strip()
        if trailing:
            raise ValueError(f"Cannot parse trailing expression near: '{trailing}'")

    return terms, const


def eval_walsh_terms(X, terms, const=0.0):
    """
    Evaluate y = const + sum_j coef_j * product_{i in indices_j} X[:, i].
    X is (P, d) with entries in {-1, +1}.
    Returns y as shape (P, 1) float32 tensor.
    """
    P, d = X.shape
    y = torch.full((P, 1), float(const), device=X.device, dtype=torch.float32)
    for coef, idx_list in terms:
        if len(idx_list) == 0:
            # empty set χ_∅(x) = 1
            y = y + float(coef)
            continue
        idx_tensor = torch.tensor(idx_list, device=X.device, dtype=torch.long)
        # product over selected coordinates
        prod = torch.prod(X[:, idx_tensor], dim=1, keepdim=True)
        y = y + float(coef) * prod
    return y


def slugify_expr(expr):
    """
    Build a short filesystem-friendly slug for directory/filenames.
    """
    # Replace braces and commas/ranges with readable tokens
    slug = expr
    slug = slug.replace(" ", "")
    slug = slug.replace("{", "S").replace("}", "")
    slug = slug.replace("*", "")
    slug = slug.replace("+", "_PLUS_").replace("-", "_MINUS_")
    slug = slug.replace(",", "-")
    slug = slug.replace(".", "p")
    # Keep it reasonably short
    return slug[:120]


# -----------------------------
# Data generation
# -----------------------------

def generate_walsh_target_data(P, d, expr, target_mode="classification", noise_std=0.0, device='cpu'):
    """
    Generate data X in {-1,+1}^d and targets y based on a Walsh expression.
      - expr: string parsed by parse_walsh_expression (supports ranges, commas, constants, coefficients)
      - target_mode:
          "classification": y = sign(real_sum), mapped to {-1, +1}
          "regression":     y = real_sum  (optionally with Gaussian noise)
      - noise_std: std of Gaussian noise added to the *real-valued* sum before optional sign (0 means none)
    """
    terms, const = parse_walsh_expression(expr)

    # Validate that indices do not exceed d
    max_idx = -1
    for _, idxs in terms:
        if idxs:
            max_idx = max(max_idx, max(idxs))
    if max_idx >= d:
        raise ValueError(
            f"Target expression references index {max_idx}, but d={d}. Increase d or change the expression."
        )

    X = torch.randint(2, size=(P, d), device=device, dtype=torch.float32) * 2 - 1  # {-1, +1}
    y_real = eval_walsh_terms(X, terms, const=const)

    if noise_std > 0.0:
        y_real = y_real + noise_std * torch.randn_like(y_real)

    if target_mode == "classification":
        # sign(0) -> +1 to avoid zeros; add tiny epsilon
        y = torch.sign(y_real + 1e-8)
        # Ensure strictly {-1, +1}
        y[y == 0] = 1.0
    elif target_mode == "regression":
        y = y_real
    else:
        raise ValueError("target_mode must be 'classification' or 'regression'.")

    return X, y


# -----------------------------
# Model
# -----------------------------

class TwoLayerNet(nn.Module):
    """
    A two-layer neural network:
    f(x) = sum_i a_i * phi(w_i^T * x), phi = ReLU
    """
    def __init__(self, d, N, g_w, g_a, gamma_scaling_exponent):
        super().__init__()
        self.d = d
        self.N = N
        sigma_w_sq = g_w / d
        sigma_a_sq = g_a / (N ** gamma_scaling_exponent)
        self.sigma_w = torch.tensor(sigma_w_sq**0.5)
        self.sigma_a = torch.tensor(sigma_a_sq**0.5)
        self.w = nn.Parameter(torch.randn(d, N) * self.sigma_w)
        self.a = nn.Parameter(torch.randn(N, 1) * self.sigma_a)
        self.phi = F.relu  # functional call

    def forward(self, x):
        pre_activation = x @ self.w
        post_activation = self.phi(pre_activation)
        output = post_activation @ self.a
        return output


# -----------------------------
# Plotting
# -----------------------------

def plot_and_save_weights(initial_w, final_w, initial_a, final_a, P, d, expr_slug, exp_id, kappa_0, eta, save_path):
    """
    Creates and saves a 2x2 plot of weight distributions before and after training.
    """
    initial_w, final_w = initial_w.cpu(), final_w.cpu()
    initial_a, final_a = initial_a.cpu(), final_a.cpu()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    sns.set_style("whitegrid")

    sns.histplot(initial_w.flatten(), ax=axes[0, 0], kde=True, stat="density", bins=50)
    axes[0, 0].set_title(f"Initial 'w' (std: {initial_w.std():.4f})")
    axes[0, 0].set_xlabel("Weight Value")

    sns.histplot(final_w.flatten(), ax=axes[0, 1], kde=True, stat="density", bins=50)
    axes[0, 1].set_title(f"Final 'w' (std: {final_w.std():.4f})")
    axes[0, 1].set_xlabel("Weight Value")

    sns.histplot(initial_a.flatten(), ax=axes[1, 0], kde=True, stat="density", bins=50)
    axes[1, 0].set_title(f"Initial 'a' (std: {initial_a.std():.4f})")
    axes[1, 0].set_xlabel("Weight Value")

    sns.histplot(final_a.flatten(), ax=axes[1, 1], kde=True, stat="density", bins=50)
    axes[1, 1].set_title(f"Final 'a' (std: {final_a.std():.4f})")
    axes[1, 1].set_xlabel("Weight Value")

    fig.suptitle(
        f'Weight Distributions | P={P}, d={d}, expr={expr_slug}, exp={exp_id}, κ₀={kappa_0}, η={eta}',
        fontsize=16
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)


# -----------------------------
# Training
# -----------------------------

def train_with_langevin(model, X_train, y_train, X_test, y_test, hyperparams, current_config, rank):
    """
    Full-batch Langevin GD with mini-batch accumulation.
    Optimized for H100 with bfloat16 and reduced host/device sync.
    """
    epochs = hyperparams['epochs']
    log_interval = hyperparams['log_interval']
    batch_size = hyperparams['batch_size']
    early_stop_loss = hyperparams['early_stop_loss']
    early_stop_error = hyperparams['early_stop_error']
    target_mode = hyperparams['target_mode']

    eta = current_config['eta']
    P_train = X_train.shape[0]

    uncompiled_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    N = uncompiled_model.N
    sigma_a = uncompiled_model.sigma_a
    sigma_w = uncompiled_model.sigma_w

    kappa_0 = current_config['kappa_0']
    gamma_scaling_exponent = hyperparams['gamma_scaling_exponent']

    kappa = kappa_0 * (N ** (1 - gamma_scaling_exponent))
    T = 2 * (kappa ** 2)

    loss_fn = nn.MSELoss(reduction='sum')

    d = current_config['d']
    exp_id = current_config['exp_id']
    expr_slug = current_config['expr_slug']

    print(f"GPU {rank} | Start (bf16): P={P_train}, d={d}, expr={expr_slug}, exp={exp_id}, N={N}, k0={kappa_0:.3f}, eta={eta:.1e}, T={T:.4f}")
    start_time = time.time()

    # local accumulators on device
    for epoch in range(epochs + 1):
        model.train()
        model.zero_grad(set_to_none=True)
        train_loss_accum = torch.zeros((), device=X_train.device, dtype=torch.float32)
        train_correct_accum = 0

        for i in range(0, P_train, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            with autocast(dtype=torch.bfloat16):
                y_pred_batch = model(X_batch)
                batch_loss = loss_fn(y_pred_batch, y_batch)

            train_loss_accum += batch_loss.detach()

            # If classification, compute 0-1 inline; else skip (leave 0)
            if target_mode == "classification":
                pred_sign = torch.sign(y_pred_batch.detach())
                pred_sign[pred_sign == 0] = 1.0
                train_correct_accum += (pred_sign == y_batch).sum().item()

            (batch_loss / P_train).backward()

        # Langevin step
        with torch.no_grad():
            grad_a = uncompiled_model.a.grad
            if grad_a is not None:
                noise_a = torch.randn_like(uncompiled_model.a) * (2 * T * eta) ** 0.5
                decay_a = (T / (sigma_a ** 2)) * uncompiled_model.a
                delta_a = -eta * (decay_a + grad_a) + noise_a
                uncompiled_model.a.add_(delta_a)

            grad_w = uncompiled_model.w.grad
            if grad_w is not None:
                noise_w = torch.randn_like(uncompiled_model.w) * (2 * T * eta) ** 0.5
                decay_w = (T / (sigma_w ** 2)) * uncompiled_model.w
                delta_w = -eta * (decay_w + grad_w) + noise_w
                uncompiled_model.w.add_(delta_w)

        if epoch % log_interval == 0:
            model.eval()
            with torch.no_grad():
                train_mse = (train_loss_accum / P_train).item()

                # Test eval
                with autocast(dtype=torch.bfloat16):
                    y_pred_test = model(X_test)
                test_mse = (loss_fn(y_pred_test, y_test) / X_test.shape[0]).item()

                # 0-1 error if classification, else NaN
                if hyperparams['target_mode'] == "classification":
                    train_error_01 = 1.0 - (train_correct_accum / P_train)
                    test_error_01 = (torch.sign(y_pred_test).where(torch.sign(y_pred_test)!=0, torch.tensor(1.0, device=y_pred_test.device)) != y_test).float().mean().item()
                else:
                    train_error_01 = float('nan')
                    test_error_01 = float('nan')

                elapsed_time = time.time() - start_time
                print(
                    f"GPU {rank} | P={P_train}, d={d}, expr={expr_slug}, exp={exp_id}, k0={kappa_0:.3f}, eta={eta:.1e} | "
                    f"Ep {epoch:6d} | Train MSE: {train_mse:.4f} | Test MSE: {test_mse:.4f} | "
                    f"Train Err: {train_error_01:.4f} | Test Err: {test_error_01:.4f} | "
                    f"Time: {elapsed_time:.2f}s"
                )

                if np.isnan(train_mse):
                    print(f"GPU {rank} | P={P_train}, d={d} | Training diverged (NaN). Stopping.")
                    return {
                        "train_mse": float('nan'), "test_mse": float('nan'),
                        "train_error_01": float('nan'), "test_error_01": float('nan'),
                    }

                # Early stopping: always allow on loss; allow on error only for classification
                if train_mse < early_stop_loss or (hyperparams['target_mode'] == "classification" and train_error_01 < early_stop_error):
                    print(f"GPU {rank} | Early stopping at epoch {epoch}.")
                    break

    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_train_mse = train_mse
        final_train_error_01 = train_error_01

        with autocast(dtype=torch.bfloat16):
            y_pred_test = model(X_test)
        final_test_mse = (loss_fn(y_pred_test, y_test) / X_test.shape[0]).item()
        final_test_error_01 = (torch.sign(y_pred_test).where(torch.sign(y_pred_test)!=0, torch.tensor(1.0, device=y_pred_test.device)) != y_test).float().mean().item() if hyperparams['target_mode']=="classification" else float('nan')

    print(f"GPU {rank} | Finished: P={P_train}, d={d}, expr={expr_slug}, exp={exp_id}, κ₀={kappa_0:.3f}, η={eta:.1e}")

    return {
        "train_mse": final_train_mse, "test_mse": final_test_mse,
        "train_error_01": final_train_error_01, "test_error_01": final_test_error_01,
    }


# -----------------------------
# Results I/O
# -----------------------------

def save_result(result, json_path):
    """Safely appends a result to the JSON file using a file lock."""
    lock = FileLock(str(json_path) + ".lock")
    with lock:
        try:
            if json_path.exists() and json_path.stat().st_size > 0:
                with open(json_path, 'r') as f:
                    data = json.load(f)
            else:
                data = []
        except json.JSONDecodeError:
            data = []
        data.append(result)
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)


# -----------------------------
# Worker
# -----------------------------

def worker(rank, world_size, job_queue, hyperparams, base_save_dir):
    """Main worker function for each GPU process."""
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True

    device = f'cuda:{rank}'
    print(f"Worker on GPU {rank} started.")

    target_expr = hyperparams['target_expr']
    target_mode = hyperparams['target_mode']
    noise_std = hyperparams.get('target_noise_std', 0.0)

    expr_slug = slugify_expr(target_expr)

    while not job_queue.empty():
        try:
            current_config = job_queue.get(timeout=1)
        except mp.queues.Empty:
            break

        P_train = current_config['P']
        kappa_0 = current_config['kappa_0']
        d = current_config['d']
        exp_id = current_config['exp_id']
        eta = current_config['eta']
        expr_slug = current_config['expr_slug']

        # Per-(d) test set for this job
        X_test, y_test = generate_walsh_target_data(
            hyperparams['P_test'], d, target_expr, target_mode=target_mode, noise_std=noise_std, device=device
        )

        # Seed per exp for different inits and data draws
        if 'base_seed' in hyperparams and hyperparams['base_seed'] is not None:
            seed = int(hyperparams['base_seed'] + exp_id + 1315423911 * (d + 1))
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed % (2**32 - 1))

        X_train, y_train = generate_walsh_target_data(
            P_train, d, target_expr, target_mode=target_mode, noise_std=noise_std, device=device
        )

        model = TwoLayerNet(
            d=d, N=hyperparams['N'], g_w=hyperparams['g_w'],
            g_a=hyperparams['g_a'], gamma_scaling_exponent=hyperparams['gamma_scaling_exponent']
        ).to(device)

        # Capture initial weights before compiling
        initial_w = model.w.detach().clone()
        initial_a = model.a.detach().clone()

        # Aggressive compile (PyTorch 2+)
        model = torch.compile(model, mode="max-autotune", fullgraph=True)
        uncompiled_model = model._orig_mod if hasattr(model, '_orig_mod') else model

        final_errors = train_with_langevin(
            model, X_train, y_train, X_test, y_test, hyperparams, current_config, rank
        )

        # Capture final weights
        final_w = uncompiled_model.w.detach().clone()
        final_a = uncompiled_model.a.detach().clone()

        # Directory structure includes expr slug (and d)
        save_dir = base_save_dir / f"d{d}_expr_{expr_slug}"
        save_dir.mkdir(exist_ok=True, parents=True)

        # Save weight distribution plot
        plots_dir = save_dir / "weight_distributions"
        plots_dir.mkdir(exist_ok=True)
        plot_filename = f"weights_P_{P_train}_d_{d}_exp_{exp_id}_kappa_{kappa_0:.6f}_eta_{eta:.6e}.png"
        plot_path = plots_dir / plot_filename
        plot_and_save_weights(initial_w, final_w, initial_a, final_a, P_train, d, expr_slug, exp_id, kappa_0, eta, plot_path)

        # Save results
        json_path = save_dir / "training_results.json"
        full_result = {
            **current_config,  # includes P, kappa_0, d, exp_id, eta, expr_slug
            **final_errors,
            "gpu": rank,
            "N": hyperparams['N'],
            "g_w": hyperparams['g_w'],
            "g_a": hyperparams['g_a'],
            "gamma_scaling_exponent": hyperparams['gamma_scaling_exponent'],
            "epochs": hyperparams['epochs'],
            "batch_size": hyperparams['batch_size'],
            "P_test": hyperparams['P_test'],
            "target_expr": target_expr,
            "target_mode": target_mode,
            "target_noise_std": noise_std,
        }
        save_result(full_result, json_path)

    print(f"Worker on GPU {rank} finished.")


# -----------------------------
# Main
# -----------------------------

def main():
    """Set up and run the distributed experiment with arbitrary Walsh targets."""
    # --- Hyperparameters (fixed across jobs) ---
    hyperparams = {
        # Model / data params
        "N": 1024,
        "g_w": 1.0,
        "g_a": 1.0,
        "gamma_scaling_exponent": 1.0,

        # Target function (Walsh expression)
        # EXAMPLES:
        # "{0-3} + {0-8}"
        # "2*{0,2,5} - 0.5*{1-4} + 0.1"
        # "{ }"                       # empty set = constant 1
        "target_expr": "{0-3} + {0-7}",

        # Choose "classification" (sign → ±1) or "regression" (real-valued sum)
        "target_mode": "regression",

        # Optional Gaussian noise added to the real-valued sum BEFORE sign (for classification) or as-is (for regression)
        "target_noise_std": 0.0,

        # Training params (job-specific 'eta' is handled via eta_values below)
        "epochs": 15_000_000,
        "log_interval": 200_000,
        "P_test": 100_000,
        "batch_size": 200_000,
        "early_stop_loss": 0.05,
        "early_stop_error": 0.001,  # only used for classification

        # Number of runs per unique (d, P, kappa_0, eta)
        "num_exp": 1,

        # Optional: set to an int for reproducible families of runs, or None for random
        "base_seed": 12345,
    }

    # --- Save directory ---
    base_save_dir = Path("/home/goring/mean_field_langevin/Langevin_training/results/20_08_target48")
    base_save_dir.mkdir(exist_ok=True, parents=True)

    # --- Experiment Grids ---
    d_values = [30]      # Must be >= max index referenced in target_expr
    P_values = [100000]
    kappa_0_values = np.logspace(np.log10(1e-4), np.log10(5e1), 15)
    eta_values = [2e-4]

    # Persist hyperparameters (incl. the expression text)
    with open(base_save_dir / "hyperparameters.json", 'w') as f:
        json.dump(hyperparams, f, indent=4)

    expr_slug = slugify_expr(hyperparams['target_expr'])

    # Build set of completed jobs (resume safely)
    completed_jobs = set()
    for d in d_values:
        json_path = base_save_dir / f"d{d}_expr_{expr_slug}" / "training_results.json"
        if json_path.exists() and json_path.stat().st_size > 0:
            try:
                with open(json_path, 'r') as f:
                    results = json.load(f)
                for r in results:
                    required = ['P', 'kappa_0', 'd', 'exp_id', 'eta', 'expr_slug']
                    if all(key in r for key in required):
                        completed_jobs.add(
                            (r['P'], f"{r['kappa_0']:.8f}", r['d'], r['exp_id'], f"{r['eta']:.8e}", r['expr_slug'])
                        )
            except json.JSONDecodeError:
                print(f"Warning: Could not decode existing results for d={d}. Continuing.")

    # Create job queue
    job_queue = mp.Queue()
    job_count = 0
    for d in d_values:
        for P in P_values:
            for k0 in kappa_0_values:
                for eta in eta_values:
                    for exp_id in range(hyperparams['num_exp']):
                        key = (P, f"{k0:.8f}", d, exp_id, f"{eta:.8e}", expr_slug)
                        if key not in completed_jobs:
                            job_queue.put({
                                "P": int(P),
                                "kappa_0": float(k0),
                                "d": int(d),
                                "exp_id": int(exp_id),
                                "eta": float(eta),
                                "expr_slug": expr_slug,
                            })
                            job_count += 1

    if job_count == 0:
        print("All experiments already completed. Exiting.")
        return

    print(f"Total jobs to run: {job_count}")

    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("No CUDA devices found. This script requires a GPU.")
        return

    print(f"Found {world_size} GPUs. Starting workers...")

    mp.spawn(worker, args=(world_size, job_queue, hyperparams, base_save_dir), nprocs=world_size, join=True)

    print("\n--- All jobs completed ---")


if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        print("Multiprocessing start method already set.")
        pass
    main()
