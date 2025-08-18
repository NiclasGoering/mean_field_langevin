import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
import os
import json

# --- 1.  Hyper-parameters & Model Setup ---
# These parameters are defined according to the provided theoretical text.
d, N = 20, 1024
gamma = 1.0

# Fundamental scaling parameters from the theory.
sigma_w = 1.0
sigma_a = 1.0

# --- Derived variances based on the paper's definitions ---
# Paper formula: g_w^2 = sigma_w / d
g_w_sq = sigma_w / d
g_w = np.sqrt(g_w_sq)

# Paper formula: g_a^2 = sigma_a / N^gamma
g_a_sq = sigma_a / (N**gamma)
g_a = np.sqrt(g_a_sq)

print(f"--- Parameters ---")
print(f"d={d}, N={N}, gamma={gamma}")
print(f"sigma_w={sigma_w:.2f}, sigma_a={sigma_a:.2f}")
print(f"g_w^2 = {g_w_sq:.4e}, g_a^2 = {g_a_sq:.4e}")
print(f"------------------")


# --- GPU/CPU Device Setup ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU found. Using CUDA.")
else:
    device = torch.device("cpu")
    print("No GPU found. Using CPU.")

# --- Activation Function Definitions ---
def relu(x):
    """Even activation function."""
    return torch.nn.functional.relu(x)

# --- 2.  Fast Boolean Sample Bank for Expectations (on GPU) ---
# This bank is used to compute expectations over the input data x.
X_bank = (2 * torch.randint(0, 2, size=(200_000, d), device=device) - 1).float()

def Sigma(w, activation_fn):
    """Computes Sigma(w) = E[phi(w^T x)^2] using the static bank."""
    z = X_bank @ w
    return torch.mean(activation_fn(z)**2)

def chi(A, X):
    """Computes the parity function chi_A(x) on a matrix of inputs."""
    return torch.prod(X[:, A], axis=1)

def J_A(w, A, activation_fn):
    """Computes J_A(w) = E[phi(w^T x) * chi_A(x)] using the static bank."""
    z = X_bank @ w
    return torch.mean(activation_fn(z) * chi(A, X_bank))

# --- 3.  Analytically Integrated Distributions (PyTorch version) ---
def log_p_w(w, m_S, kappa, S, activation_fn):
    """
    Calculates the log-probability of w, log p(w), which is -S_eff(w).
    This function implements the integrated-out action formula.
    """
    Sigma_w = Sigma(w, activation_fn)
    J_S_w = J_A(w, S, activation_fn)
    kappa_sq = kappa**2

    # Paper formula for the w prior term in the action: (d / (2 * sigma_w)) * ||w||^2
    s_prior_w = (d / (2.0 * sigma_w)) * torch.dot(w, w)

    # Paper formula for the coefficient of a^2 in the action: (N^gamma / sigma_a)
    prior_a_coeff = (N**gamma) / sigma_a
    
    denom_A = prior_a_coeff + (Sigma_w / (kappa_sq + 1e-20))

    # Paper formula for the log-determinant term in S_eff: 0.5 * log(...)
    log_det_term = 0.5 * torch.log(denom_A)

    numerator_B_base = J_S_w * (1.0 - m_S)
    
    # Paper formula for the final term in S_eff: - ( (1/kappa^4) * (J_Y-m*J_S)^2 ) / (4 * A)
    exp_term_numerator_B_sq = (numerator_B_base**2) / (kappa_sq * kappa_sq + 1e-20)
    exp_term = exp_term_numerator_B_sq / (4.0 * denom_A)

    # The log probability is the negative of the action: log p(w) = -S_eff(w)
    return -s_prior_w - log_det_term + exp_term

# --- 4.  Langevin Monte Carlo Sampler ---
def sample_w_langevin_then_a(m_S, kappa, n_samp, burn, step_size, S, activation_fn):
    """Samples w via Langevin Monte Carlo, then samples a from p(a|w)."""
    # Initialize w from its prior distribution: w ~ N(0, g_w^2 * I)
    w = torch.randn(d, device=device) * g_w
    
    W_samples, A_samples = [], []

    # Burn-in and sampling phase
    for t in range(burn + n_samp):
        # --- LMC Step ---
        w.requires_grad_(True)
        
        log_prob = log_p_w(w, m_S, kappa, S, activation_fn)
        
        log_prob.backward()
        grad = w.grad
        if t % 3000 == 0:
                print(f"t={t}, ||w||={w.norm():.2f}, ||grad||={grad.norm():.18e}")
        
        # --- Update and Sample without tracking gradients ---
        with torch.no_grad():
            if grad is not None:
                w.add_(grad, alpha=step_size / 2.0)
                w.grad.zero_()
            
            w.add_(torch.randn_like(w), alpha=np.sqrt(step_size))
            
            w = w.detach()

            # --- Collect samples after burn-in phase ---
            if t >= burn:
                Sigma_w = Sigma(w, activation_fn)
                J_S_w = J_A(w, S, activation_fn)
                
                prior_a_coeff = (N**gamma) / sigma_a
                var_a_inv = prior_a_coeff + (Sigma_w / (kappa**2 + 1e-12))
                var_a = 1.0 / var_a_inv
                
                mean_a_numerator = (J_S_w * (1.0 - m_S)) / (kappa**2 + 1e-12)
                mean_a = var_a * mean_a_numerator
                
                a = torch.randn(1, device=device) * torch.sqrt(var_a) + mean_a
                
                W_samples.append(w.clone())
                A_samples.append(a)
                # Inside the LMC for loop in sample_w_langevin_then_a
                 # Print every 500 steps
        
    return torch.stack(W_samples), torch.stack(A_samples)

# --- 5.  Self-Consistency Loop ---
def solve_kappa(kappa, S, activation_fn, m0=0.0, iters=320, damping=0.1):
    """
    Solves the self-consistency equation for m_S using the stable LMC sampler
    with a properly capped adaptive step size.
    """
    m_S = m0
    history = [m_S]
    
    # --- Adaptive LMC Step Size with a cap ---
    # The step size is scaled with kappa^2 but is now capped by the base value.
    # This prevents explosions when kappa is large.
    base_lmc_step_size = 5e-3
    scaled_step_size = base_lmc_step_size * (kappa**2)
    lmc_step_size = max(min(base_lmc_step_size, scaled_step_size), 1e-9)
    
    print(f"\n--- Solving for kappa = {kappa:.4e} (starting m_S = {m0:.4f}) ---")
    print(f"Using adaptive LMC step size: {lmc_step_size:.2e}")

    for i in range(iters):
        # 1. Sample (w, a) pairs using the robust LANGEVIN sampler.
        W_samples, A_samples = sample_w_langevin_then_a(
            m_S, kappa, 6000, 2000, lmc_step_size, S, activation_fn
        )
        
        # 2. Calculate the new order parameter m_S based on the samples.
        with torch.no_grad():
            J_S_w_all = torch.stack([J_A(w, S, activation_fn) for w in W_samples])
            m_S_new = N * torch.mean(A_samples * J_S_w_all)
            
            # 3. Update m_S with damping for stability.
            m_S_update = (1 - damping) * m_S + damping * m_S_new.item()
            
            # Safety clamp to prevent single-step explosions from rare bad samples.
            m_S = np.clip(m_S_update, -10, 10)
            
            history.append(m_S)
        
        print(f"Iter {i+1:3d}: m_S={m_S:.8f}")

        # Convergence criterion: break if m_S has stabilized.
        if i >= 150:
            with torch.no_grad():
                last_20 = np.array(history[-20:])
                mean_val = np.mean(last_20)
                std_val = np.std(last_20)
                
                converged_to_zero = abs(mean_val) < 1e-4 and std_val < 1e-4
                converged_to_nonzero = (abs(mean_val) > 1e-4 and (std_val / abs(mean_val)) < 0.02)

                if converged_to_zero or converged_to_nonzero:
                    print(f"CONVERGED: Value stabilized at m_S = {mean_val:.8f}")
                    m_S = mean_val
                    break
    
    return m_S, history

# --- 6.  Plotting Functions ---
def create_summary_plot(kappa_0_values, m_S_values, S_k, act_name, output_dir):
    """Generates a summary plot of m_S vs. kappa_0."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Phase Transition (Adaptive LMC Solver): $m_S$ vs. $\\kappa_0$ ({act_name.upper()}, |S|={S_k})", fontsize=18)

    ax1.plot(kappa_0_values, m_S_values, 'o-')
    ax1.set_title("Linear Scale")
    ax1.set_xlabel("Base Noise Parameter $\\kappa_0$")
    ax1.set_ylabel("Converged Order Parameter $m_S$")
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2.semilogx(kappa_0_values, m_S_values, 'o-')
    ax2.set_title("Logarithmic Scale")
    ax2.set_xlabel("Base Noise Parameter $\\kappa_0$")
    ax2.set_ylabel("Converged Order Parameter $m_S$")
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = f"summary_phase_transition_{act_name}_k{S_k}_ADAPTIVE_LMC.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=200)
    plt.show()

# --- 7.  Main Execution ---
if __name__ == "__main__":
    output_dir = "/home/goring/mean_field_langevin/MCMC_Pinf/results/3107_MCMCd40k4" 
    activation_choice = 'relu'
    S = tuple(range(2)) # Target parity on first 4 features, |S|=4
    kappa_0_values = [1] #np.logspace(5, -5, 25) # Extended range to test stability

    os.makedirs(output_dir, exist_ok=True)
    activation_fn = relu
    
    print(f"\nRunning with ADAPTIVE LANGEVIN SOLVER for {activation_choice.upper()}, |S|={len(S)}")
    
    converged_m_S_values = []
    m_S_guess = 0.01

    
    # Scan from high noise (m_S=0) to low noise (m_S>0) for stability.
    for kappa_0 in sorted(kappa_0_values, reverse=True):
        kappa = kappa_0 * (N**(1.0 - gamma))
        
        m_S, history = solve_kappa(kappa, S, activation_fn, m0=m_S_guess)
        converged_m_S_values.append(m_S)
        m_S_guess = m_S if np.isfinite(m_S) else 0.0
        
    final_m_S_values = list(reversed(converged_m_S_values))
    final_kappa_0s = sorted(kappa_0_values)

    results_data = {
        "S_k": len(S), "activation": activation_choice, "gamma": gamma, "d": d, "N": N,
        "sigma_w": sigma_w, "sigma_a": sigma_a,
        "kappa_0_values": final_kappa_0s, 
        "m_S_values": final_m_S_values
    }
    
    results_filename = f"results_data_{activation_choice}_k{len(S)}_ADAPTIVE_LMC.json"
    with open(os.path.join(output_dir, results_filename), 'w') as f:
        json.dump(results_data, f, indent=4)
    
    create_summary_plot(final_kappa_0s, final_m_S_values, 
                        S_k=len(S), act_name=activation_choice, output_dir=output_dir)
