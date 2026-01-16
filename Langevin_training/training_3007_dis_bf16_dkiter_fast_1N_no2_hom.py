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
import queue as pyqueue  # for Empty exception in mp.Queue
from torch.amp import autocast

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
    f(x) = (1 / N^gamma) * sum_i a_i * phi(w_i^T x)
    where phi can be ReLU or sigmoid.
    Prior: a_i ~ N(0, g_a),  w_jk ~ N(0, g_w/d) independently.
    """
    def __init__(self, d, N, g_w, g_a, gamma_scaling_exponent: float, activation='relu'):
        super().__init__()
        self.d = d
        self.N = N
        self.gamma = float(gamma_scaling_exponent)

        sigma_w_sq = float(g_w) / d
        sigma_a_sq = float(g_a)
        self.sigma_w = math.sqrt(sigma_w_sq)
        self.sigma_a = math.sqrt(sigma_a_sq)

        self.w = nn.Parameter(torch.randn(d, N) * self.sigma_w)
        self.a = nn.Parameter(torch.randn(N, 1) * self.sigma_a)
        
        # Set activation function
        if activation == 'sigmoid':
            self.phi = torch.sigmoid
        elif activation == 'relu':
            self.phi = F.relu
        else:
            raise ValueError(f"Unknown activation: {activation}. Must be 'relu' or 'sigmoid'.")

    def forward(self, x):
        return (self.phi(x @ self.w) @ self.a) / (self.N ** self.gamma)

    # +++ NEW METHOD +++
    def homogeneity_loss(self):
        """
        Penalize variance of |w| within each neuron (column).

        self.w is shape (d, N).
        We take abs(w), compute variance across dim 0 (the d components) for each
        of the N neurons, then sum these N variances.
        """
        w_abs = self.w.abs()  # (d, N)
        var_per_neuron = torch.var(w_abs, dim=0, unbiased=False)
        return var_per_neuron.sum()

# -----------------------------
# LR schedule helper
# -----------------------------
def poly_decay_lr(epoch: int, eta_start: float, eta_final: float,
                  decay_steps: int, power: float = 2.0) -> float:
    """
    Polynomial decay from eta_start to eta_final over 'decay_steps' steps.
    Holds at eta_final afterwards.
    """
    if decay_steps <= 0:
        return eta_final
    tau = min(1.0, epoch / float(decay_steps))
    return eta_final + (eta_start - eta_final) * (1.0 - tau) ** power

# -----------------------------
# Custom Optimizer (SGLD/Langevin-GD) with foreach updates
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
            lr = float(group['lr'])
            T = float(group['T'])
            sigma_sq = group.get('sigma_sq', None)
            if sigma_sq is None:
                raise ValueError("Each param group must have 'sigma_sq' set.")

            # collect tensors that have grads
            ps = [p for p in group['params'] if p.grad is not None]
            if not ps:
                continue

            # decay = (T/sigma_sq) * p  (computed elementwise per param)
            decay_coeff = T / float(sigma_sq)
            decays = [p.mul(decay_coeff) for p in ps]

            # -lr * (decay + grad)
            drift_updates = [-(lr) * (d + p.grad) for p, d in zip(ps, decays)]

            # sqrt(2*T*lr) * N(0, I)  -- generate in fp32 then cast to param dtype
            noise_std = math.sqrt(2.0 * T * lr)
            noises = [torch.randn_like(p, dtype=torch.float32).mul_(noise_std).to(p.dtype) for p in ps]

            # combined updates: drift + noise
            updates = [du + nz for du, nz in zip(drift_updates, noises)]

            # foreach fused add
            torch._foreach_add_(ps, updates)

        return loss

def _cavity_constants(y_pred: torch.Tensor, y_true: torch.Tensor):
    yp = y_pred.to(torch.float32)
    yt = y_true.to(torch.float32)
    m_S = (yp * yt).mean().item()
    r = yp - m_S * yt
    noise_norm2 = (r * r).mean().item()
    err01_direct = (torch.sign(yp) != yt).float().mean().item()
    return m_S, noise_norm2, err01_direct

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
# Pruning success criterion (kept for reporting only)
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
# +++ MODIFIED FUNCTION +++
def train_with_langevin(model, X_train, y_train, X_test, y_test, hyperparams, current_config, device, use_full_batch):
    """
    Trains using Langevin GD.
    - If use_full_batch: single forward/backward per epoch over entire training set.
    - Else: mini-batch SGLD.

    Early stopping:
      * Default (unchanged): trigger when test MSE < 0.03, then continue +100k epochs and stop.
      * NEW SWITCH: if (P_train <= hyperparams['train_error_switch_P_below']) OR
                      (kappa_0 >= hyperparams['train_error_switch_kappa_above']),
                      trigger when TRAIN 0-1 error <= 0.01, then continue +100k epochs and stop.
    """
    epochs = hyperparams['epochs']
    log_interval = hyperparams['log_interval']
    batch_size = hyperparams['batch_size']

    # Final LR from grid (backwards compat), start LR from config or default to final
    eta_final = float(current_config['eta'])
    eta_start = float(current_config.get('eta_start', eta_final))
    lr_decay_steps = int(hyperparams.get('lr_decay_steps', 0))
    lr_power = float(hyperparams.get('lr_power', 2.0))

    P_train = int(X_train.shape[0])
    N = model.N
    sigma_a = float(model.sigma_a)
    sigma_w = float(model.sigma_w)

    kappa_0 = float(current_config['kappa_0'])
    gamma_scaling_exponent = float(current_config['gamma_scaling_exponent'])

    # --- IMPORTANT: use kappa as-is; T = 2*kappa^2
    kappa = kappa_0
    T = 2.0 * (kappa ** 2)

    loss_fn = nn.MSELoss(reduction='mean')  # mean == per-sample MSE

    d = int(current_config['d'])
    k = int(current_config['k'])
    exp_id = int(current_config['exp_id'])

    # +++ GET NEW HYPERPARAM +++
    lambda_h = float(hyperparams.get("homogeneity_penalty_weight", 0.0))
    if lambda_h > 0.0:
        print(f"[GPU {device}] Applying homogeneity penalty with lambda_h = {lambda_h:.2e}")

    # Optimizer with two param groups (so each carries its sigma^2)
    optimizer = LangevinGD(
        params=[
            {'params': [model.a], 'sigma_sq': sigma_a ** 2},
            {'params': [model.w], 'sigma_sq': sigma_w ** 2},
        ],
        lr=eta_start,  # will be overwritten each epoch by schedule
        T=T,
    )

    # -----------------------------
    # NEW: Decide early-stop mode based on thresholds
    # -----------------------------
    P_switch = hyperparams.get('train_error_switch_P_below', None)
    kappa_switch = hyperparams.get('train_error_switch_kappa_above', None)
    use_train_error_switch = (
        (P_switch is not None and P_train <= int(P_switch)) or
        (kappa_switch is not None and kappa_0 >= float(kappa_switch))
    )
    TRAIN_ERR_TARGET = 0.01  # as requested

    early_stop_mode_str = "train_err<=0.01" if use_train_error_switch else "test_mse<0.01"
    
    # Get activation for logging
    activation = current_config.get('activation', hyperparams.get('activation', 'relu'))

    print(f"[GPU {device}] Start (bf16): P={P_train}, d={d}, k={k}, exp={exp_id}, "
          f"N={N}, gamma={gamma_scaling_exponent}, k0={kappa_0:.3e}, "
          f"eta0={eta_start:.2e}, etaf={eta_final:.2e}, T={T:.4e}, "
          f"activation={activation}, full_batch={use_full_batch}, decay_steps={lr_decay_steps}, p={lr_power}, "
          f"early_stop={early_stop_mode_str}")

    start_time = time.time()
    epochs_run = 0
    stop_after_epoch = None
    stopped_early = False

    with torch.no_grad():
        with autocast("cuda", dtype=torch.bfloat16):
            y_pred_test0 = model(X_test)
    init_eval_mS, init_eval_noise_norm2, init_eval_err01_direct = _cavity_constants(y_pred_test0, y_test)

    # training loop
    for epoch in range(epochs + 1):
        # schedule the LR
        lr_t = poly_decay_lr(epoch, eta_start, eta_final, lr_decay_steps, lr_power)
        for g in optimizer.param_groups:
            g['lr'] = lr_t

        model.train()

        if use_full_batch:
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.bfloat16):
                y_pred_train = model(X_train)
                # 1. Calculate base MSE loss
                train_mse_loss = loss_fn(y_pred_train, y_train)

                # +++ 2. ADD HOMOGENEITY PENALTY +++
                if lambda_h > 0.0:
                    homog_loss = model.homogeneity_loss() * lambda_h
                    total_loss = train_mse_loss + homog_loss
                else:
                    total_loss = train_mse_loss
                # +++ END +++

            # 3. Backprop on the TOTAL loss
            total_loss.backward()
            optimizer.step()

            # Metrics on train
            # Log the un-penalized MSE
            train_mse = train_mse_loss.detach().item()
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
                with autocast("cuda", dtype=torch.bfloat16):
                    yb_pred = model(xb)
                    # 1. Calculate base MSE loss
                    batch_mse_loss = loss_fn(yb_pred, yb)

                    # +++ 2. ADD HOMOGENEITY PENALTY +++
                    # This is correct for SGLD: the regularizer's
                    # full gradient is added at each step.
                    if lambda_h > 0.0:
                        homog_loss = model.homogeneity_loss() * lambda_h
                        total_loss = batch_mse_loss + homog_loss
                    else:
                        total_loss = batch_mse_loss
                    # +++ END +++

                # 3. Backprop on the TOTAL loss
                total_loss.backward()
                optimizer.step()

                # Log the un-penalized MSE
                train_mse_sum += batch_mse_loss.detach().item() * xb.shape[0]
                train_correct_accum += (torch.sign(yb_pred.detach()) == yb).sum().item()

            train_mse = train_mse_sum / P_train
            train_error_01 = 1.0 - (train_correct_accum / P_train)

        # Logging & early stopping (only at log_interval)
        if epoch % log_interval == 0:
            model.eval()
            with torch.no_grad():
                with autocast("cuda", dtype=torch.bfloat16):
                    y_pred_test = model(X_test)
                test_mse = loss_fn(y_pred_test, y_test).item()
                test_error_01 = (torch.sign(y_pred_test) != y_test).float().mean().item()

            elapsed = time.time() - start_time
            print(f"[GPU {device}] P={P_train}, d={d}, k={k}, exp={exp_id}, "
                  f"gamma={gamma_scaling_exponent}, k0={kappa_0:.3e}, eta_now={lr_t:.2e} | "
                  f"Ep {epoch:>7} | Train MSE: {train_mse:.6f} | Test MSE: {test_mse:.6f} | "
                  f"Train Err: {train_error_01:.6f} | Test Err: {test_error_01:.6f} | "
                  f"T={T:.4e} | Time: {elapsed:.1f}s | ES:{early_stop_mode_str}")

            if np.isnan(train_mse) or np.isnan(test_mse):
                print(f"[GPU {device}] NaN detected. Stopping.")
                stopped_early = False
                epochs_run = epoch
                break

            # -----------------------------
            # NEW: Early stopping trigger (mode-dependent)
            # -----------------------------
            if stop_after_epoch is None:
                if use_train_error_switch:
                    if train_error_01 <= TRAIN_ERR_TARGET:
                        stop_after_epoch = epoch + 100_000
                        print(f"[GPU {device}] Early-stop trigger hit (train err <= {TRAIN_ERR_TARGET:.3f}). "
                              f"Continuing until epoch {stop_after_epoch}.")
                else:
                    # Original behavior: use test MSE threshold
                    if test_mse < 0.04:
                        stop_after_epoch = epoch + 100_000
                        print(f"[GPU {device}] Early-stop trigger hit (test MSE {test_mse:.3e}). "
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
        with autocast("cuda", dtype=torch.bfloat16):
            y_pred_train_final = model(X_train).clone()  # <-- clone here
            y_pred_test_final  = model(X_test)
        final_train_mse = loss_fn(y_pred_train_final, y_train).item()
        final_test_mse  = loss_fn(y_pred_test_final,  y_test ).item()

        final_train_error_01 = (torch.sign(y_pred_train_final) != y_train).float().mean().item()
        final_test_error_01 = (torch.sign(y_pred_test_final) != y_test).float().mean().item()
        final_eval_mS, final_eval_noise_norm2, final_eval_err01_direct = _cavity_constants(y_pred_test_final, y_test)

    print(f"[GPU {device}] Finished: P={P_train}, d={d}, k={k}, exp={exp_id}, "
          f"gamma={gamma_scaling_exponent}, k0={kappa_0:.3e}, eta0={eta_start:.2e}, etaf={eta_final:.2e}, "
          f"epochs_run={epochs_run}, stopped_early={stopped_early}, early_stop={early_stop_mode_str}")

    return {
        "train_mse": final_train_mse,
        "test_mse": final_test_mse,
        "train_error_01": final_train_error_01,
        "test_error_01": final_test_error_01,
        "stopped_early": stopped_early,
        "epochs_run": int(epochs_run),
        "init_eval_mS": init_eval_mS,
        "init_eval_noise_norm2": init_eval_noise_norm2,
        "init_eval_err01_direct": init_eval_err01_direct,
        # Final (post-training) eval constants
        "final_eval_mS": final_eval_mS,
        "final_eval_noise_norm2": final_eval_noise_norm2,
        "final_eval_err01_direct": final_eval_err01_direct,
        # LR schedule info
        "eta_start": eta_start,
        "eta_final": eta_final,
        "lr_decay_steps": lr_decay_steps,
        "lr_power": lr_power,
        # NEW: record which early-stop mode was active
        "early_stop_mode": early_stop_mode_str,
    }

# -----------------------------
# Worker (no pruning/skip logic)
# -----------------------------
def worker(global_rank,
           num_gpus,
           per_gpu_workers,
           job_queue,
           hyperparams,
           base_save_dir):
    """
    Each process pulls jobs from the queue and trains all of them. No skipping/pruning.
    Saves initial and final model state_dicts (.pt) with descriptive names.
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
        eta_final = float(current_config['eta'])
        eta_start = float(current_config.get('eta_start', eta_final))
        gamma_scaling_exponent = float(current_config['gamma_scaling_exponent'])

        # Build per-(d,k) test set for this job
        X_test, y_test = generate_k_sparse_parity_data(hyperparams['P_test'], d, k, device=device)

        # Seed per exp/d/k so initializations and data are reproducible across jobs
        if 'base_seed' in hyperparams and hyperparams['base_seed'] is not None:
            seed = int(hyperparams['base_seed'] + exp_id + 1315423911 * (d + 1) + 2654435761 * (k + 1))
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed % (2**32 - 1))

        X_train, y_train = generate_k_sparse_parity_data(P_train, d, k, device=device)

        # Get activation from config or hyperparams (config takes precedence)
        activation = current_config.get('activation', hyperparams.get('activation', 'relu'))
        
        model = TwoLayerNet(
            d=d, N=hyperparams['N'], g_w=hyperparams['g_w'],
            g_a=hyperparams['g_a'], gamma_scaling_exponent=gamma_scaling_exponent,
            activation=activation
        ).to(device)

        # compile the model (no logic change)
        #try:
        model = torch.compile(model, mode='max-autotune')
        #except Exception as e:
        #   print(f"[GPU {device}] torch.compile unavailable or failed: {e}. Proceeding without compile.")

        # -----------------------------
        # SAVE DIR + FILENAMES
        # -----------------------------
        save_dir = base_save_dir / f"d{d}_k{k}"
        save_dir.mkdir(exist_ok=True, parents=True)

        def smart_name(prefix):
            return (
                f"{prefix}_P_{P_train}_d_{d}_k_{k}_exp_{exp_id}"
                f"_kappa_{kappa_0:.6f}_eta0_{eta_start:.6e}_etaf_{eta_final:.6e}"
                f"_gamma_{gamma_scaling_exponent:.6f}.pt"
            )

        init_path = save_dir / smart_name("init_model")
        final_path = save_dir / smart_name("final_model")

        # -----------------------------
        # SAVE INITIAL SNAPSHOT *IMMEDIATELY* (FROZEN COPY)
        # -----------------------------
        init_state_frozen = {k_: v_.detach().cpu().clone() for k_, v_ in model.state_dict().items()}
        torch.save(init_state_frozen, init_path)

        # Conditional full-batch only if dataset fits in one batch
        use_full_batch = P_train <= hyperparams['batch_size']

        final_metrics = train_with_langevin(
            model, X_train, y_train, X_test, y_test, hyperparams, current_config, device, use_full_batch
        )

        # -----------------------------
        # SAVE FINAL SNAPSHOT (FROZEN COPY)
        # -----------------------------
        final_state_frozen = {k_: v_.detach().cpu().clone() for k_, v_ in model.state_dict().items()}
        torch.save(final_state_frozen, final_path)

        # Save results JSON
        json_path = save_dir / "training_results.json"
        full_result = {
            **current_config,
            **final_metrics,
            "gpu": device_idx,
            "worker_rank": global_rank,
            "N": hyperparams['N'],
            "g_w": hyperparams['g_w'],
            "g_a": hyperparams['g_a'],
            "epochs": hyperparams['epochs'],
            "batch_size": hyperparams['batch_size'],
            "P_test": hyperparams['P_test'],
            "status": "trained",
            "init_model_path": str(init_path),
            "final_model_path": str(final_path),
            # Keep success flag purely for reporting
            "success": is_success(final_metrics["train_error_01"], final_metrics["test_error_01"]),
        }
        save_result(full_result, json_path)

    print(f"Worker rank {global_rank} (device {device}) finished.")

# -----------------------------
# Main
# -----------------------------
def main():
    # --- Hyperparameters (fixed across jobs) ---
    # +++ MODIFIED DICTIONARY +++
    hyperparams = {
        # Model / data params
        "N": 512,
        "g_w": 1.0,
        "g_a": 1.0,
        
        # Activation function: 'relu' or 'sigmoid'
        "activation": "sigmoid",  # Change to 'relu' for ReLU activation

        # +++ NEW HYPERPARAMETER +++
        # Strength of the homogeneity penalty.
        # You will need to tune this value. Start small.
        "homogeneity_penalty_weight": 0.0,  # Example value, tune this!

        # Training params
        "epochs": 20_000_000,
        "log_interval": 250_000,
        "P_test": 100_000,
        "batch_size": 200_000,  # full-batch triggers when P_train <= batch_size
        "early_stop_loss": 1e-20,   # kept for JSON continuity
        "early_stop_error": 1e-20,  # kept for JSON continuity

        # LR schedule knobs
        "lr_decay_steps": 5_000_000,  # try 300_000 (aggressive) or 2_000_000 (conservative)
        "lr_power": 2.0,

        # Number of runs per unique (d, k, P, kappa_0, eta, gamma)
        "num_exp": 2,

        # Optional seed family
        "base_seed": 12345,

        # -----------------------------
        # NEW: Early-stop switching thresholds
        # -----------------------------
        "train_error_switch_P_below": 550,      # example: small-P jobs use train-error early stop
        "train_error_switch_kappa_above": 0.02, # example: large-kappa jobs use train-error early stop
    }

    # --- Save directory ---
    base_save_dir = Path("/home/goring/mean_field_langevin/Langevin_training/results_rebuttal/d35_k3_sig_full")
    base_save_dir.mkdir(exist_ok=True, parents=True)

    # --- Experiment Grids (ORDER MATTERS) ---
    d_values = [35]
    k_values = [3]  # k <= d

    # P descending (start from largest P)
    P_values = [10,100,500,1000,2333,5000,7666,10000]#[500,1000,10000,2133 ,10, 100, 750, 3666, 5000, 7500,20000,30000,50000]
    #P_values = sorted(P_values, reverse=True)  # descending

    # kappa ascending (start from smallest kappa
    kappa_0_values = [5e-3]
    kappa_0_values = sorted(kappa_0_values)   # ascending

    # gamma values
    gamma_values = [0.5]

    # LR grid (FINAL values)
    eta_values = [5e-4]
    # starting LR
    eta_start = 2e-3

    # Save hyperparams snapshot (+ the grids we sweep)
    snapshot = dict(hyperparams)
    snapshot.update({
        "P_values_desc": P_values,
        "kappa_0_values_asc": kappa_0_values,
        "gamma_values": gamma_values,
        "eta_values_final": eta_values,
        "eta_start": eta_start,
        "d_values": d_values,
        "k_values": k_values,
    })
    with open(base_save_dir / "hyperparameters.json", 'w') as f:
        json.dump(snapshot, f, indent=4)

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
                        required = ['P', 'kappa_0', 'd', 'k', 'exp_id', 'eta', 'gamma_scaling_exponent']
                        if all(key in r for key in required):
                            completed_jobs.add(
                                (int(r['P']),
                                 f"{float(r['kappa_0']):.8f}",
                                 int(r['d']), int(r['k']),
                                 int(r['exp_id']),
                                 f"{float(r['eta']):.8e}",
                                 f"{float(r['gamma_scaling_exponent']):.6f}")
                            )
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode existing results for d={d}, k={k}. Continuing.")

    # --- Build job queue (train ALL combos; no skipping) ---
    job_queue = mp.Queue()
    job_count = 0
    for d in d_values:
        for k in k_values:
            for gamma in gamma_values:
                for k0 in kappa_0_values:      # kappa ascending outer loop
                    for P in P_values:          # P descending inner loop
                        for eta in eta_values:  # FINAL LR
                            for exp_id in range(hyperparams['num_exp']):
                                key = (int(P),
                                       f"{float(k0):.8f}",
                                       int(d), int(k),
                                       int(exp_id),
                                       f"{float(eta):.8e}",
                                       f"{float(gamma):.6f}")
                                if key not in completed_jobs:
                                    job_queue.put({
                                        "P": int(P),
                                        "kappa_0": float(k0),
                                        "d": int(d),
                                        "k": int(k),
                                        "exp_id": int(exp_id),
                                        "eta": float(eta),             # FINAL lr
                                        "eta_start": float(eta_start), # START lr
                                        "gamma_scaling_exponent": float(gamma),
                                        "activation": hyperparams.get('activation', 'relu'),
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

    # --- 2 workers/GPU ---
    per_gpu_workers = 2
    nprocs = num_gpus * per_gpu_workers
    print(f"Launching {nprocs} workers ({per_gpu_workers} per GPU).")

    # Spawn (no shared pruning state needed)
    mp.spawn(
        worker,
        args=(
            num_gpus,
            per_gpu_workers,
            job_queue,
            hyperparams,
            base_save_dir,
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
