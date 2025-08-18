import json
import time
from pathlib import Path
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


def generate_k_sparse_parity_data(P, d, k, device='cpu'):
    """
    Generates a dataset for the k-sparse parity problem on the specified device.
    """
    if k > d:
        raise ValueError("k (number of sparse features) cannot be greater than d (dimensionality).")
    X = torch.randint(2, size=(P, d), device=device, dtype=torch.float32) * 2 - 1
    relevant_features = X[:, :k]
    y = torch.prod(relevant_features, dim=1).unsqueeze(1)
    return X, y


class TwoLayerNet(nn.Module):
    """
    A two-layer neural network as specified by the user.
    f(x) = sum_i a_i * phi(w_i^T * x)
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
        self.phi = F.relu  # functional call (slightly leaner)

    def forward(self, x):
        pre_activation = x @ self.w
        post_activation = self.phi(pre_activation)
        output = post_activation @ self.a
        return output


def plot_and_save_weights(initial_w, final_w, initial_a, final_a, P, d, k, exp_id, kappa_0, save_path):
    """
    Creates and saves a 2x2 plot of weight distributions before and after training.
    """
    # Ensure tensors are on CPU for plotting
    initial_w, final_w = initial_w.cpu(), final_w.cpu()
    initial_a, final_a = initial_a.cpu(), final_a.cpu()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    sns.set_style("whitegrid")

    # Plot initial 'w' distribution
    sns.histplot(initial_w.flatten(), ax=axes[0, 0], kde=True, stat="density", bins=50)
    axes[0, 0].set_title(f"Initial 'w' (std: {initial_w.std():.4f})")
    axes[0, 0].set_xlabel("Weight Value")

    # Plot final 'w' distribution
    sns.histplot(final_w.flatten(), ax=axes[0, 1], kde=True, stat="density", bins=50)
    axes[0, 1].set_title(f"Final 'w' (std: {final_w.std():.4f})")
    axes[0, 1].set_xlabel("Weight Value")

    # Plot initial 'a' distribution
    sns.histplot(initial_a.flatten(), ax=axes[1, 0], kde=True, stat="density", bins=50)
    axes[1, 0].set_title(f"Initial 'a' (std: {initial_a.std():.4f})")
    axes[1, 0].set_xlabel("Weight Value")

    # Plot final 'a' distribution
    sns.histplot(final_a.flatten(), ax=axes[1, 1], kde=True, stat="density", bins=50)
    axes[1, 1].set_title(f"Final 'a' (std: {final_a.std():.4f})")
    axes[1, 1].set_xlabel("Weight Value")

    fig.suptitle(f'Weight Distributions | P={P}, d={d}, k={k}, exp={exp_id}, κ₀={kappa_0}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)  # Close the figure to free up memory


def train_with_langevin(model, X_train, y_train, X_test, y_test, hyperparams, current_config, rank):
    """
    Trains the model using full-batch Langevin GD with mini-batch accumulation.
    >>> Optimized for H100 with bfloat16 and reduced host/device sync. <<<
    """
    # Unpack hyperparameters
    eta = hyperparams['eta']
    epochs = hyperparams['epochs']
    log_interval = hyperparams['log_interval']
    batch_size = hyperparams['batch_size']
    early_stop_loss = hyperparams['early_stop_loss']
    early_stop_error = hyperparams['early_stop_error']

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
    k = current_config['k']
    exp_id = current_config['exp_id']

    print(f"GPU {rank} | Start (bf16): P={P_train}, d={d}, k={k}, exp={exp_id}, N={N}, k0={kappa_0:.3f}, T={T:.4f}")
    start_time = time.time()

    # local tensor accumulators to avoid CPU syncs in the loop
    for epoch in range(epochs + 1):
        model.train()

        # Reset grads & accumulators
        model.zero_grad(set_to_none=True)
        train_loss_accum = torch.zeros((), device=X_train.device, dtype=torch.float32)
        train_correct_accum = 0

        # Mini-batch accumulation
        for i in range(0, P_train, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            # Forward in bf16; grads in fp32
            with autocast(dtype=torch.bfloat16):
                y_pred_batch = model(X_batch)
                batch_loss = loss_fn(y_pred_batch, y_batch)

            # Accumulate loss on GPU (no .item() here)
            train_loss_accum += batch_loss.detach()

            # Accumulate 0-1 training accuracy inline to avoid a second pass
            # (detach to keep it out of autograd)
            pred_sign = torch.sign(y_pred_batch.detach())
            train_correct_accum += (pred_sign == y_batch).sum().item()

            # Backward; normalize by total samples for stable scaling
            (batch_loss / P_train).backward()

        # Manual Langevin update (parameters in fp32)
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

        # Logging & early stopping
        if epoch % log_interval == 0:
            model.eval()
            with torch.no_grad():
                train_mse = (train_loss_accum / P_train).item()
                train_error_01 = 1.0 - (train_correct_accum / P_train)

                # Test evaluation in bfloat16
                with autocast(dtype=torch.bfloat16):
                    y_pred_test = model(X_test)
                test_mse = (loss_fn(y_pred_test, y_test) / X_test.shape[0]).item()
                test_error_01 = (torch.sign(y_pred_test) != y_test).float().mean().item()

                elapsed_time = time.time() - start_time
                print(
                    f"GPU {rank} | P={P_train}, d={d}, k={k}, exp={exp_id}, k0={kappa_0:.3f} | "
                    f"Ep {epoch:6d} | Train MSE: {train_mse:.4f} | Test MSE: {test_mse:.4f} | "
                    f"Train Err: {train_error_01:.4f} | Test Err: {test_error_01:.4f} | "
                    f"Time: {elapsed_time:.2f}s"
                )

                if np.isnan(train_mse):
                    print(f"GPU {rank} | P={P_train}, d={d}, k={k} | Training diverged (NaN). Stopping.")
                    return {
                        "train_mse": float('nan'), "test_mse": float('nan'),
                        "train_error_01": float('nan'), "test_error_01": float('nan'),
                    }

                if train_mse < early_stop_loss or train_error_01 < early_stop_error:
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
        final_test_error_01 = (torch.sign(y_pred_test) != y_test).float().mean().item()

    print(f"GPU {rank} | Finished: P={P_train}, d={d}, k={k}, exp={exp_id}, κ₀={kappa_0:.3f}")

    return {
        "train_mse": final_train_mse, "test_mse": final_test_mse,
        "train_error_01": final_train_error_01, "test_error_01": final_test_error_01,
    }


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


def worker(rank, world_size, job_queue, hyperparams, base_save_dir):
    """The main worker function for each GPU process."""
    # H100-friendly settings
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True

    device = f'cuda:{rank}'
    print(f"Worker on GPU {rank} started.")

    while not job_queue.empty():
        try:
            current_config = job_queue.get(timeout=1)
        except mp.queues.Empty:
            break

        P_train = current_config['P']
        kappa_0 = current_config['kappa_0']
        d = current_config['d']
        k = current_config['k']
        exp_id = current_config['exp_id']

        # Per-(d,k) test set for this job (keeps memory bounded)
        X_test, y_test = generate_k_sparse_parity_data(
            hyperparams['P_test'], d, k, device=device
        )

        # Seed per exp for different inits (and different data draws)
        if 'base_seed' in hyperparams and hyperparams['base_seed'] is not None:
            seed = int(hyperparams['base_seed'] + exp_id + 1315423911 * (d + 1) + 2654435761 * (k + 1))
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed % (2**32 - 1))

        X_train, y_train = generate_k_sparse_parity_data(
            P_train, d, k, device=device
        )

        model = TwoLayerNet(
            d=d, N=hyperparams['N'], g_w=hyperparams['g_w'],
            g_a=hyperparams['g_a'], gamma_scaling_exponent=hyperparams['gamma_scaling_exponent']
        ).to(device)

        # Capture initial weights before compiling
        initial_w = model.w.detach().clone()
        initial_a = model.a.detach().clone()

        # Aggressive compile for H100
        model = torch.compile(model, mode="max-autotune", fullgraph=True)
        uncompiled_model = model._orig_mod if hasattr(model, '_orig_mod') else model

        final_errors = train_with_langevin(
            model, X_train, y_train, X_test, y_test, hyperparams, current_config, rank
        )

        # Capture final weights
        final_w = uncompiled_model.w.detach().clone()
        final_a = uncompiled_model.a.detach().clone()

        # Directory structure now includes d and k
        save_dir = base_save_dir / f"d{d}_k{k}"
        save_dir.mkdir(exist_ok=True, parents=True)

        # Save weight distribution plot
        plots_dir = save_dir / "weight_distributions"
        plots_dir.mkdir(exist_ok=True)
        plot_filename = f"weights_P_{P_train}_d_{d}_k_{k}_exp_{exp_id}_kappa_{kappa_0:.6f}.png"
        plot_path = plots_dir / plot_filename
        plot_and_save_weights(initial_w, final_w, initial_a, final_a, P_train, d, k, exp_id, kappa_0, plot_path)

        # Save results (json per (d,k) directory and also global if you like)
        json_path = save_dir / "training_results.json"
        full_result = {
            **current_config,
            **final_errors,
            "gpu": rank,
            "N": hyperparams['N'],
            "g_w": hyperparams['g_w'],
            "g_a": hyperparams['g_a'],
            "gamma_scaling_exponent": hyperparams['gamma_scaling_exponent'],
            "eta": hyperparams['eta'],
            "epochs": hyperparams['epochs'],
            "batch_size": hyperparams['batch_size'],
            "P_test": hyperparams['P_test'],
        }
        save_result(full_result, json_path)

    print(f"Worker on GPU {rank} finished.")


def main():
    """Main function to set up and run the distributed experiment."""
    # --- Hyperparameters ---
    hyperparams = {
        # Model / data params
        "N": 1024,
        "g_w": 1.0,
        "g_a": 1.0,
        "gamma_scaling_exponent": 1.0,

        # Training params
        "eta": 5e-4,
        "epochs": 4_000_000,
        "log_interval": 10_000,
        "P_test": 100_000,
        "batch_size": 200_000,
        "early_stop_loss": 0.05,
        "early_stop_error": 0.001,

        # Number of runs per unique (d, k, P, kappa_0)
        "num_exp": 3,

        # Optional: set to an int for reproducible families of runs, or None for random
        "base_seed": 12345,
    }

    # --- Save directory (make sure this path exists on your system) ---
    base_save_dir = Path("/home/goring/mean_field_langevin/Langevin_training/results/d40_k4_1708_grid")
    base_save_dir.mkdir(exist_ok=True, parents=True)

    # --- Experiment Grids ---
    d_values = [40]         # Now iterable
    k_values = [4]          # Now iterable
    P_values = [50000, 100000]
    kappa_0_values = np.logspace(np.log10(1e-4), np.log10(5e1), 30)

    # Save the hyperparameters and grids
    with open(base_save_dir / "hyperparameters.json", 'w') as f:
        json.dump(hyperparams, f, indent=4)

    # Build a set of completed jobs (across all d,k) to resume safely
    completed_jobs = set()
    # Search all subdirs d*/k* for their training_results.json
    for d in d_values:
        for k in k_values:
            json_path = base_save_dir / f"d{d}_k{k}" / "training_results.json"
            if json_path.exists() and json_path.stat().st_size > 0:
                try:
                    with open(json_path, 'r') as f:
                        results = json.load(f)
                    for r in results:
                        # Ensure keys exist before adding to set; use tolerant float formatting for kappa_0
                        if all(key in r for key in ['P', 'kappa_0', 'd', 'k', 'exp_id']):
                            completed_jobs.add(
                                (r['P'], f"{r['kappa_0']:.8f}", r['d'], r['k'], r['exp_id'])
                            )
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode existing results for d={d}, k={k}. Continuing.")

    # Create job queue including d, k, and exp_id
    job_queue = mp.Queue()
    job_count = 0
    for d in d_values:
        for k in k_values:
            for P in P_values:
                for k0 in kappa_0_values:
                    for exp_id in range(hyperparams['num_exp']):
                        key = (P, f"{k0:.8f}", d, k, exp_id)
                        if key not in completed_jobs:
                            job_queue.put({
                                "P": P,
                                "kappa_0": float(k0),
                                "d": int(d),
                                "k": int(k),
                                "exp_id": int(exp_id),
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

    # Robust start method for CUDA multiprocessing
    mp.spawn(worker, args=(world_size, job_queue, hyperparams, base_save_dir), nprocs=world_size, join=True)

    print("\n--- All jobs completed ---")


if __name__ == '__main__':
    try:
        # 'spawn' is a robust start method, good for CUDA applications
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        print("Multiprocessing start method already set.")
        pass
    main()
