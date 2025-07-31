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
        self.phi = torch.relu

    def forward(self, x):
        pre_activation = x @ self.w
        post_activation = self.phi(pre_activation)
        output = post_activation @ self.a
        return output

def plot_and_save_weights(initial_w, final_w, initial_a, final_a, P, kappa_0, save_path):
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
    axes[0, 0].set_title(f"Initial 'w' Distribution (std: {initial_w.std():.4f})")
    axes[0, 0].set_xlabel("Weight Value")

    # Plot final 'w' distribution
    sns.histplot(final_w.flatten(), ax=axes[0, 1], kde=True, stat="density", bins=50)
    axes[0, 1].set_title(f"Final 'w' Distribution (std: {final_w.std():.4f})")
    axes[0, 1].set_xlabel("Weight Value")

    # Plot initial 'a' distribution
    sns.histplot(initial_a.flatten(), ax=axes[1, 0], kde=True, stat="density", bins=50)
    axes[1, 0].set_title(f"Initial 'a' Distribution (std: {initial_a.std():.4f})")
    axes[1, 0].set_xlabel("Weight Value")

    # Plot final 'a' distribution
    sns.histplot(final_a.flatten(), ax=axes[1, 1], kde=True, stat="density", bins=50)
    axes[1, 1].set_title(f"Final 'a' Distribution (std: {final_a.std():.4f})")
    axes[1, 1].set_xlabel("Weight Value")

    fig.suptitle(f'Weight Distributions for P={P}, κ₀={kappa_0}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig) # Close the figure to free up memory


def train_with_langevin(model, X_train, y_train, X_test, y_test, hyperparams, current_config, rank):
    """
    Trains the model using full-batch Langevin GD with mini-batch accumulation.
    >>> OPTIMIZED for H100 with bfloat16 and efficient logging. <<<
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
    
    print(f"GPU {rank} | Starting job (bfloat16): P={P_train}, N={N}, kappa_0={kappa_0:.3f}, T={T:.4f}")
    start_time = time.time()

    for epoch in range(epochs + 1):
        model.train()
        
        # --- Gradient Accumulation with bfloat16 ---
        model.zero_grad(set_to_none=True)
        
        # This will store the loss from the forward pass for efficient logging
        train_loss_for_logging = 0.0
        
        for i in range(0, P_train, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            
            # Use autocast with bfloat16 for the forward pass.
            # No GradScaler is needed.
            with autocast(dtype=torch.bfloat16):
                y_pred_batch = model(X_batch)
                batch_loss = loss_fn(y_pred_batch, y_batch)

            # Store the loss for logging later
            train_loss_for_logging += batch_loss.item()
            
            # Standard backward pass. Gradients are accumulated in float32.
            (batch_loss / P_train).backward()

        # --- Manual Langevin Update ---
        # Gradients are already in float32, no unscaling needed.
        with torch.no_grad():
            grad_a = uncompiled_model.a.grad
            if grad_a is not None:
                noise_a = torch.randn_like(uncompiled_model.a) * (2 * T * eta)**0.5
                decay_a = (T / (sigma_a ** 2)) * uncompiled_model.a
                delta_a = -eta * (decay_a + grad_a) + noise_a
                uncompiled_model.a.add_(delta_a)

            grad_w = uncompiled_model.w.grad
            if grad_w is not None:
                noise_w = torch.randn_like(uncompiled_model.w) * (2 * T * eta)**0.5
                decay_w = (T / (sigma_w ** 2)) * uncompiled_model.w
                delta_w = -eta * (decay_w + grad_w) + noise_w
                uncompiled_model.w.add_(delta_w)

        # --- Logging and Early Stopping Check ---
        if epoch % log_interval == 0:
            model.eval()
            with torch.no_grad():
                # Directly use the stored training loss, avoiding a redundant pass
                train_mse = train_loss_for_logging / P_train
                
                if np.isnan(train_mse):
                    print(f"GPU {rank} | P={P_train}, k0={kappa_0:.3f} | Training diverged with NaN loss. Stopping.")
                    return {
                        "train_mse": float('nan'), "test_mse": float('nan'),
                        "train_error_01": float('nan'), "test_error_01": float('nan'),
                    }
                
                # Still need one pass to get the 0-1 error on training data
                train_correct_accum = 0
                for i in range(0, P_train, batch_size):
                    X_batch = X_train[i:i+batch_size]
                    y_batch = y_train[i:i+batch_size]
                    with autocast(dtype=torch.bfloat16):
                         y_pred_batch = model(X_batch)
                    train_correct_accum += (torch.sign(y_pred_batch) == y_batch).float().sum().item()
                train_error_01 = 1.0 - (train_correct_accum / P_train)
                
                # Test evaluation in bfloat16 for speed
                with autocast(dtype=torch.bfloat16):
                    y_pred_test = model(X_test)
                test_mse = loss_fn(y_pred_test, y_test) / X_test.shape[0]
                test_error_01 = (torch.sign(y_pred_test) != y_test).float().mean()
                
                elapsed_time = time.time() - start_time
                print(
                    f"GPU {rank} | P={P_train}, k0={kappa_0:.3f} | Ep {epoch:6d} | "
                    f"Train MSE: {train_mse:.4f} | Test MSE: {test_mse.item():.4f} | "
                    f"Train Err: {train_error_01:.4f} | Test Err: {test_error_01.item():.4f} | "
                    f"Time: {elapsed_time:.2f}s"
                )

                if train_mse < early_stop_loss or train_error_01 < early_stop_error:
                    print(f"GPU {rank} | Early stopping triggered at epoch {epoch}.")
                    break
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        # Use the last computed training metrics
        final_train_mse = train_mse
        final_train_error_01 = train_error_01
        
        # Re-evaluate on test set for final numbers
        with autocast(dtype=torch.bfloat16):
            y_pred_test = model(X_test)
        final_test_mse = (loss_fn(y_pred_test, y_test) / X_test.shape[0]).item()
        final_test_error_01 = (torch.sign(y_pred_test) != y_test).float().mean().item()
    
    print(f"GPU {rank} | Finished job: P={P_train}, kappa_0={kappa_0:.3f}")

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

def worker(rank, world_size, job_queue, hyperparams, save_dir):
    """The main worker function for each GPU process."""
    torch.set_float32_matmul_precision('high')
    device = f'cuda:{rank}'
    print(f"Worker on GPU {rank} started.")

    X_test, y_test = generate_k_sparse_parity_data(
        hyperparams['P_test'], hyperparams['d'], hyperparams['k'], device=device
    )

    while not job_queue.empty():
        try:
            current_config = job_queue.get(timeout=1)
        except mp.queues.Empty:
            break

        P_train = current_config['P']
        kappa_0 = current_config['kappa_0']
        
        X_train, y_train = generate_k_sparse_parity_data(
            P_train, hyperparams['d'], hyperparams['k'], device=device
        )

        model = TwoLayerNet(
            d=hyperparams['d'], N=hyperparams['N'], g_w=hyperparams['g_w'],
            g_a=hyperparams['g_a'], gamma_scaling_exponent=hyperparams['gamma_scaling_exponent']
        ).to(device)
        
        # Capture initial weights before compiling
        initial_w = model.w.detach().clone()
        initial_a = model.a.detach().clone()

        model = torch.compile(model)
        uncompiled_model = model._orig_mod if hasattr(model, '_orig_mod') else model

        final_errors = train_with_langevin(
            model, X_train, y_train, X_test, y_test, hyperparams, current_config, rank
        )
        
        # Capture final weights
        final_w = uncompiled_model.w.detach().clone()
        final_a = uncompiled_model.a.detach().clone()

        # Save weight distribution plot
        plots_dir = save_dir / "weight_distributions"
        plots_dir.mkdir(exist_ok=True)
        plot_filename = f"weights_P_{P_train}_kappa_{kappa_0:.2f}.png"
        plot_path = plots_dir / plot_filename
        plot_and_save_weights(initial_w, final_w, initial_a, final_a, P_train, kappa_0, plot_path)
        
        full_result = {**current_config, **final_errors, "gpu": rank}
        save_result(full_result, save_dir / "training_results.json")

    print(f"Worker on GPU {rank} finished.")

def main():
    """Main function to set up and run the distributed experiment."""
    hyperparams = {
        "d": 30, "k": 4, "N": 1024, "g_w": 1.0, "g_a": 1.0,
        "gamma_scaling_exponent": 2.0, "eta": 5e-4, "epochs": 4000000,
        "log_interval": 10000, "P_test": 100000,
        "batch_size": 200000,
        "early_stop_loss": 0.05, "early_stop_error": 0.001,
    }
    
    # IMPORTANT: Make sure this path is correct for your system
    save_dir = Path("/home/goring/mean_field_langevin/Langevin_training/results/d30_k4_3107_grid")
    save_dir.mkdir(exist_ok=True)

    # --- Experiment Grid ---
    P_values = [10,100,1000,5000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,
                125000,150000,175000,200000,250000,300000,400000,500000,600000,700000,800000,900000,1000000]
    # A wider range might be needed for a full phase plot
    kappa_0_values = np.logspace(np.log10(1e-5), np.log10(1e5), 30)

    json_path = save_dir / "training_results.json"
    
    completed_jobs = set()
    if json_path.exists() and json_path.stat().st_size > 0:
        with open(json_path, 'r') as f:
            try:
                results = json.load(f)
                for r in results:
                    # Ensure keys exist before adding to set
                    if 'P' in r and 'kappa_0' in r:
                        # Use a tolerance for float comparison
                        completed_jobs.add((r['P'], f"{r['kappa_0']:.8f}"))
            except json.JSONDecodeError:
                print("Warning: Could not decode existing results.json. Starting fresh.")

    job_queue = mp.Queue()
    job_count = 0
    for P in P_values:
        for k0 in kappa_0_values:
            if (P, f"{k0:.8f}") not in completed_jobs:
                job_queue.put({"P": P, "kappa_0": k0})
                job_count += 1

    if job_count == 0:
        print("All experiments already completed. Exiting.")
        return

    print(f"Total jobs to run: {job_count}")
    
    with open(save_dir / "hyperparameters.json", 'w') as f:
        json.dump(hyperparams, f, indent=4)

    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("No CUDA devices found. This script requires a GPU.")
        return
    
    print(f"Found {world_size} GPUs. Starting workers...")
    mp.spawn(worker, args=(world_size, job_queue, hyperparams, save_dir), nprocs=world_size, join=True)
    
    print("\n--- All jobs completed ---")

if __name__ == '__main__':
    try:
        # 'spawn' is a robust start method, good for CUDA applications
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        print("Multiprocessing start method already set.")
        pass
    main()
