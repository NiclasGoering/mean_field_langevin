import os
import json
import time
from pathlib import Path
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from filelock import FileLock
from torch.cuda.amp import autocast
import queue as pyqueue  # for Empty exception in mp.Queue






# -----------------------------
# Data
# -----------------------------
def generate_k_sparse_parity_data(P, d, k, device='cpu'):
    """
    Generates a dataset for the k-sparse parity problem on the specified device.
    X in {-1,+1}^d, label is product of first k features.
    """
    if k > d:
        raise ValueError("k (number of sparse features) cannot be greater than d (dimensionality).")
    X = torch.randint(2, size=(P, d), device=device, dtype=torch.float32) * 2 - 1
    relevant = X[:, :k]
    y = torch.prod(relevant, dim=1, keepdim=True)
    return X, y


# -----------------------------
# Model
# -----------------------------
class TwoLayerNet(nn.Module):
    """
    f(x) = sum_i a_i * ReLU(w_i^T x)
    """
    def __init__(self, d, N, g_w, g_a, gamma_scaling_exponent):
        super().__init__()
        self.d = d
        self.N = N
        sigma_w_sq = float(g_w) / d
        sigma_a_sq = float(g_a) #/ (N ** gamma_scaling_exponent)
        self.sigma_w = math.sqrt(sigma_w_sq)
        self.sigma_a = math.sqrt(sigma_a_sq)
        self.w = nn.Parameter(torch.randn(d, N) * self.sigma_w)
        self.a = nn.Parameter(torch.randn(N, 1) * self.sigma_a)
        self.phi = F.relu

    def forward(self, x):
        return 1/self.N* self.phi(x @ self.w) @ self.a


# -----------------------------
# Custom Optimizer (SGLD/Langevin-GD)
# -----------------------------
class LangevinGD(torch.optim.Optimizer):
    """
    Param groups must include 'sigma_sq' in each group.
    Update: p <- p - lr * ( (T/sigma_sq) * p + grad ) + sqrt(2*T*lr) * N(0,I)
    """
    def __init__(self, params, lr, T):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, T=T)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            T = group['T']
            sigma_sq = group.get('sigma_sq', None)
            if sigma_sq is None:
                raise ValueError("Each param group must have 'sigma_sq' set.")

            for p in group['params']:
                if p.grad is None:
                    continue
                decay = (T / sigma_sq) * p
                noise = torch.randn_like(p) * math.sqrt(2.0 * T * lr)
                p.add_(-lr * (decay + p.grad) + noise)

        return loss


# -----------------------------
# JSON Utils
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
# Pruning success criterion
# -----------------------------
def is_success(train_err_01: float, test_err_01: float) -> bool:
    """
    Success cap:
      - train 0-1 error <= 1e-3
      - test  0-1 error <= 0.25
    """
    return (train_err_01 <= 1e-3) and (test_err_01 <= 0.25)


# -----------------------------
# Training (with conditional full-batch)
# -----------------------------
def train_with_langevin(model, X_train, y_train, X_test, y_test, hyperparams, current_config, device, use_full_batch):
    """
    Trains using Langevin GD.
    - If use_full_batch: single forward/backward per epoch over entire training set.
    - Else: mini-batch SGLD.
    Early stopping: when test 0-1 error < 1e-3, continue for +500k epochs then stop.
    """
    epochs = hyperparams['epochs']
    log_interval = hyperparams['log_interval']
    batch_size = hyperparams['batch_size']

    eta = float(current_config['eta'])
    P_train = int(X_train.shape[0])

    N = model.N
    sigma_a = float(model.sigma_a)
    sigma_w = float(model.sigma_w)

    kappa_0 = float(current_config['kappa_0'])
    gamma_scaling_exponent = hyperparams['gamma_scaling_exponent']
    kappa = kappa_0 * (N ** (1 - gamma_scaling_exponent))
    T =  2.0 * (kappa ** 2) / P_train  #2.0 * (kappa ** 2)

    loss_fn = nn.MSELoss(reduction='mean')  # mean == per-sample MSE

    d = int(current_config['d'])
    k = int(current_config['k'])
    exp_id = int(current_config['exp_id'])

    # Optimizer with two param groups (so each carries its sigma^2)
    optimizer = LangevinGD(
        params=[
            {'params': [model.a], 'sigma_sq': sigma_a ** 2},
            {'params': [model.w], 'sigma_sq': sigma_w ** 2},
        ],
        lr=eta,
        T=T,
    )

    print(f"[GPU {device}] Start (bf16): P={P_train}, d={d}, k={k}, exp={exp_id}, "
          f"N={N}, k0={kappa_0:.3e}, eta={eta:.2e}, T={T:.4e}, full_batch={use_full_batch}")

    start_time = time.time()
    epochs_run = 0
    stop_after_epoch = None
    stopped_early = False

    # training loop
    for epoch in range(epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        if use_full_batch:
            with autocast(dtype=torch.bfloat16):
                y_pred_train = model(X_train)
                train_loss_mean = loss_fn(y_pred_train, y_train)
            train_loss_mean.backward()
            optimizer.step()

            # Metrics on train
            train_mse = train_loss_mean.detach().item()
            train_correct = (torch.sign(y_pred_train.detach()) == y_train).sum().item()
            train_error_01 = 1.0 - (train_correct / P_train)
        else:
            # mini-batch SGLD
            train_mse_sum = 0.0
            train_correct_accum = 0
            for i in range(0, P_train, batch_size):
                xb = X_train[i:i + batch_size]
                yb = y_train[i:i + batch_size]

                optimizer.zero_grad(set_to_none=True)
                with autocast(dtype=torch.bfloat16):
                    yb_pred = model(xb)
                    batch_loss_mean = loss_fn(yb_pred, yb)
                batch_loss_mean.backward()
                optimizer.step()

                train_mse_sum += batch_loss_mean.detach().item() * xb.shape[0]
                train_correct_accum += (torch.sign(yb_pred.detach()) == yb).sum().item()

            train_mse = train_mse_sum / P_train
            train_error_01 = 1.0 - (train_correct_accum / P_train)

        # Logging & early stopping (only at log_interval)
        if epoch % log_interval == 0:
            model.eval()
            with torch.no_grad():
                with autocast(dtype=torch.bfloat16):
                    y_pred_test = model(X_test)
                test_mse = loss_fn(y_pred_test, y_test).item()
                test_error_01 = (torch.sign(y_pred_test) != y_test).float().mean().item()

            elapsed = time.time() - start_time
            print(f"[GPU {device}] P={P_train}, d={d}, k={k}, exp={exp_id}, k0={kappa_0:.3e}, eta={eta:.2e} | "
                  f"Ep {epoch:>7} | Train MSE: {train_mse:.6f} | Test MSE: {test_mse:.6f} | "
                  f"Train Err: {train_error_01:.6f} | Test Err: {test_error_01:.6f} | "
                  f"Time: {elapsed:.1f}s")

            if np.isnan(train_mse) or np.isnan(test_mse):
                print(f"[GPU {device}] NaN detected. Stopping.")
                stopped_early = False
                epochs_run = epoch
                break

            # Early stopping trigger (as before): if test err dips below 1e-3, train +500k epochs then stop
            if stop_after_epoch is None and test_error_01 < 1e-3:
                stop_after_epoch = epoch + 500_000
                print(f"[GPU {device}] Early-stop trigger hit: test err {test_error_01:.3e}. "
                      f"Continuing until epoch {stop_after_epoch}.")

            if stop_after_epoch is not None and epoch >= stop_after_epoch:
                stopped_early = True
                epochs_run = epoch
                print(f"[GPU {device}] Early-stop completed at epoch {epoch}.")
                break

        epochs_run = epoch

    # Final evaluation
    model.eval()
    with torch.no_grad():
        with autocast(dtype=torch.bfloat16):
            y_pred_train_final = model(X_train)
            y_pred_test_final = model(X_test)
        final_train_mse = loss_fn(y_pred_train_final, y_train).item()
        final_test_mse = loss_fn(y_pred_test_final, y_test).item()
        final_train_error_01 = (torch.sign(y_pred_train_final) != y_train).float().mean().item()
        final_test_error_01 = (torch.sign(y_pred_test_final) != y_test).float().mean().item()

    print(f"[GPU {device}] Finished: P={P_train}, d={d}, k={k}, exp={exp_id}, "
          f"k0={kappa_0:.3e}, eta={eta:.2e}, epochs_run={epochs_run}, stopped_early={stopped_early}")

    return {
        "train_mse": final_train_mse,
        "test_mse": final_test_mse,
        "train_error_01": final_train_error_01,
        "test_error_01": final_test_error_01,
        "stopped_early": stopped_early,
        "epochs_run": int(epochs_run),
    }


# -----------------------------
# Worker
# -----------------------------
def worker(global_rank,
           num_gpus,
           per_gpu_workers,
           job_queue,
           hyperparams,
           base_save_dir,
           kappa_values_sorted,   # ascending
           P_values_sorted_desc,  # descending
           shared_status,         # manager dict: status map
           skip_state,            # manager dict: skip thresholds
           shared_lock):          # manager RLock
    """
    Each process pulls jobs from the queue.
    - Uses 3 workers/GPU
    - "Confirm neighbor" rule before pruning: if (k, P) fails, require that (k-1, P_prev_larger) also fails.
    - Saves initial and final model state_dicts (.pt) with descriptive names
    """
    try:
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass

    device_idx = global_rank % num_gpus
    device = f'cuda:{device_idx}'
    if torch.cuda.is_available():
        torch.cuda.set_device(device_idx)
    print(f"Worker rank {global_rank} using device {device} (per-GPU workers: {per_gpu_workers}).")

    # lookup helpers
    def k_idx_of(kappa0: float) -> int:
        return kappa_values_sorted.index(kappa0)

    def p_idx_of(P: int) -> int:
        return P_values_sorted_desc.index(P)

    def status_key(d, k, exp, k_idx, p_idx) -> str:
        return f"{d}:{k}:{exp}:{k_idx}:{p_idx}"

    # should we skip this job based on current skip thresholds?
    def should_skip_job(P, kappa_0):
        with shared_lock:
            if not skip_state['active']:
                return False
            k_idx = k_idx_of(kappa_0)
            return (k_idx >= skip_state['kappa_idx_threshold']) and (P <= skip_state['P_threshold'])

    # After a failure at (k_idx, p_idx), decide whether to expand skip region.
    # Require neighbor (k_idx-1, p_idx-1) to ALSO be failure (if it exists).
    def maybe_expand_skip_region(d, k, exp_id, k_idx, p_idx):
        if k_idx <= 0 or p_idx <= 0:
            return  # no neighbor exists; don't prune yet

        neighbor_key = status_key(d, k, exp_id, k_idx - 1, p_idx - 1)
        with shared_lock:
            neighbor_status = shared_status.get(neighbor_key, None)
            if neighbor_status == "failure":
                cur_P = P_values_sorted_desc[p_idx]
                skip_state['active'] = True
                skip_state['kappa_idx_threshold'] = min(skip_state['kappa_idx_threshold'], k_idx) \
                    if skip_state['kappa_idx_threshold'] != (1 << 30) else k_idx
                skip_state['P_threshold'] = max(skip_state['P_threshold'], cur_P)
                print(f"[SKIP REGION UPDATE] Now skipping all with "
                      f"kappa_idx >= {skip_state['kappa_idx_threshold']} "
                      f"and P <= {skip_state['P_threshold']}.")
            else:
                pass

    while True:
        try:
            current_config = job_queue.get(timeout=1)
        except pyqueue.Empty:
            break

        P_train = int(current_config['P'])
        kappa_0 = float(current_config['kappa_0'])
        d = int(current_config['d'])
        k = int(current_config['k'])
        exp_id = int(current_config['exp_id'])
        eta = float(current_config['eta'])

        if should_skip_job(P_train, kappa_0):
            print(f"[GPU {device}] SKIP job (P={P_train}, k0={kappa_0:.3e}) due to frontier rule. Writing random chance.")
            save_dir = base_save_dir / f"d{d}_k{k}"
            save_dir.mkdir(exist_ok=True, parents=True)
            json_path = save_dir / "training_results.json"
            skipped = {
                **current_config,
                "train_mse": float('nan'),
                "test_mse": float('nan'),
                "train_error_01": 0.5,
                "test_error_01": 0.5,
                "gpu": device_idx,
                "worker_rank": global_rank,
                "N": hyperparams['N'],
                "g_w": hyperparams['g_w'],
                "g_a": hyperparams['g_a'],
                "gamma_scaling_exponent": hyperparams['gamma_scaling_exponent'],
                "epochs": hyperparams['epochs'],
                "batch_size": hyperparams['batch_size'],
                "P_test": hyperparams['P_test'],
                "status": "skipped_random_chance",
            }
            save_result(skipped, json_path)
            continue

        # Build per-(d,k) test set for this job
        X_test, y_test = generate_k_sparse_parity_data(hyperparams['P_test'], d, k, device=device)

        # Seed per exp/d/k so initializations and data are reproducible across jobs
        if 'base_seed' in hyperparams and hyperparams['base_seed'] is not None:
            seed = int(hyperparams['base_seed'] + exp_id + 1315423911 * (d + 1) + 2654435761 * (k + 1))
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed % (2**32 - 1))

        X_train, y_train = generate_k_sparse_parity_data(P_train, d, k, device=device)

        model = TwoLayerNet(
            d=d, N=hyperparams['N'], g_w=hyperparams['g_w'],
            g_a=hyperparams['g_a'], gamma_scaling_exponent=hyperparams['gamma_scaling_exponent']
        ).to(device)

        # Save initial state_dict before training  --->  .pt files
        init_state = model.state_dict()

        # Conditional full-batch (#1) only if dataset fits in one batch
        use_full_batch = P_train <= hyperparams['batch_size']

        final_metrics = train_with_langevin(
            model, X_train, y_train, X_test, y_test, hyperparams, current_config, device, use_full_batch
        )

        # Save final state_dict  --->  .pt files
        final_state = model.state_dict()

        # Directory structure and filenames
        save_dir = base_save_dir / f"d{d}_k{k}"
        save_dir.mkdir(exist_ok=True, parents=True)

        # Save initial and final models (state_dicts) with descriptive names
        def smart_name(prefix):
            return f"{prefix}_P_{P_train}_d_{d}_k_{k}_exp_{exp_id}_kappa_{kappa_0:.6f}_eta_{eta:.6e}.pt"

        init_path = save_dir / smart_name("init_model")
        final_path = save_dir / smart_name("final_model")
        torch.save(init_state, init_path)
        torch.save(final_state, final_path)

        # Save results JSON (unchanged schema + extras)
        json_path = save_dir / "training_results.json"
        full_result = {
            **current_config,
            **final_metrics,
            "gpu": device_idx,
            "worker_rank": global_rank,
            "N": hyperparams['N'],
            "g_w": hyperparams['g_w'],
            "g_a": hyperparams['g_a'],
            "gamma_scaling_exponent": hyperparams['gamma_scaling_exponent'],
            "epochs": hyperparams['epochs'],
            "batch_size": hyperparams['batch_size'],
            "P_test": hyperparams['P_test'],
            "status": "trained",
            "init_model_path": str(init_path),
            "final_model_path": str(final_path),
        }
        save_result(full_result, json_path)

        # ---- Record status & maybe expand skip region (with confirmation) ----
        k_idx = k_idx_of(kappa_0)
        p_idx = p_idx_of(P_train)
        key = status_key(d, k, exp_id, k_idx, p_idx)
        success = is_success(final_metrics["train_error_01"], final_metrics["test_error_01"])
        with shared_lock:
            shared_status[key] = "success" if success else "failure"

        if not success:
            # require neighbor (k_idx-1, p_idx-1) also failure before pruning
            maybe_expand_skip_region(d, k, exp_id, k_idx, p_idx)

    print(f"Worker rank {global_rank} (device {device}) finished.")


# -----------------------------
# Main
# -----------------------------
def main():
    # --- Hyperparameters (fixed across jobs) ---
    hyperparams = {
        # Model / data params
        "N": 1024,
        "g_w": 1.0,
        "g_a": 1.0,
        "gamma_scaling_exponent": 1.0,

        # Training params
        "epochs": 10_000_000,
        "log_interval": 100_000,
        "P_test": 100_000,
        "batch_size": 200_000,   # #1 applies only when P_train <= batch_size
        "early_stop_loss": 1e-20,    # kept for JSON continuity
        "early_stop_error": 1e-20,   # kept for JSON continuity

        # Number of runs per unique (d, k, P, kappa_0, eta)
        "num_exp": 1,

        # Optional seed family
        "base_seed": 12345,
    }

    # --- Save directory ---
    base_save_dir = Path("/home/goring/mean_field_langevin/Langevin_training/results/d25_k4_2508_grid_fixedT2_g1_1N")
    base_save_dir.mkdir(exist_ok=True, parents=True)

    # --- Experiment Grids (ORDER MATTERS) ---
    d_values = [25]
    k_values = [4]

    # P descending (start from largest P)
    P_values = [100,1000,5000,10000,20000] #[10, 100, 250, 500, 600, 700, 800, 900, 1000, 2000]
    P_values = sorted(P_values, reverse=True)  # descending

    # kappa ascending (start from smallest kappa)
    kappa_0_values = [1e-2]
    kappa_0_values = sorted(kappa_0_values)   # ascending

    # LR grid
    eta_values = [2e-4]

    # Save hyperparams snapshot
    with open(base_save_dir / "hyperparameters.json", 'w') as f:
        json.dump(hyperparams, f, indent=4)

    # --- Resume support: collect completed jobs ---
    completed_jobs = set()
    for d in d_values:
        for k in k_values:
            json_path = base_save_dir / f"d{d}_k{k}" / "training_results.json"
            if json_path.exists() and json_path.stat().st_size > 0:
                try:
                    with open(json_path, 'r') as f:
                        results = json.load(f)
                    for r in results:
                        required = ['P', 'kappa_0', 'd', 'k', 'exp_id', 'eta']
                        if all(key in r for key in required):
                            completed_jobs.add(
                                (int(r['P']), f"{float(r['kappa_0']):.8f}", int(r['d']), int(r['k']),
                                 int(r['exp_id']), f"{float(r['eta']):.8e}")
                            )
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode existing results for d={d}, k={k}. Continuing.")

    # --- Build job queue in the requested order (kappa asc, P desc) ---
    job_queue = mp.Queue()
    job_count = 0
    for d in d_values:
        for k in k_values:
            for k0 in kappa_0_values:           # kappa ascending outer loop
                for P in P_values:               # P descending inner loop
                    for eta in eta_values:
                        for exp_id in range(hyperparams['num_exp']):
                            key = (int(P), f"{float(k0):.8f}", int(d), int(k), int(exp_id), f"{float(eta):.8e}")
                            if key not in completed_jobs:
                                job_queue.put({
                                    "P": int(P),
                                    "kappa_0": float(k0),
                                    "d": int(d),
                                    "k": int(k),
                                    "exp_id": int(exp_id),
                                    "eta": float(eta),
                                })
                                job_count += 1

    if job_count == 0:
        print("All experiments already completed. Exiting.")
        return

    print(f"Total jobs to run: {job_count}")

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No CUDA devices found. This script requires a GPU.")
        return
    print(f"Found {num_gpus} GPU(s).")

    # --- 3 workers/GPU (as requested) ---
    per_gpu_workers = 3
    nprocs = num_gpus * per_gpu_workers
    print(f"Launching {nprocs} workers ({per_gpu_workers} per GPU).")

    # Shared state across workers
    manager = mp.Manager()

    # Map of results to coordinate the confirmation rule:
    # key = "d:k:exp:k_idx:p_idx"  -> "success" | "failure"
    shared_status = manager.dict()

    # Skip region thresholds
    skip_state = manager.dict()
    skip_state['active'] = False
    skip_state['kappa_idx_threshold'] = 1 << 30  # large
    skip_state['P_threshold'] = -1               # small

    shared_lock = manager.RLock()

    # Spawn
    mp.spawn(
        worker,
        args=(
            num_gpus,
            per_gpu_workers,
            job_queue,
            hyperparams,
            base_save_dir,
            kappa_0_values,     # ascending
            P_values,           # descending
            shared_status,
            skip_state,
            shared_lock,
        ),
        nprocs=nprocs,
        join=True
    )
    print("\n--- All jobs completed ---")


if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
