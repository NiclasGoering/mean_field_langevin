import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import json

# --- 1. Hyper-parameters & Model Setup ---
# Using tractable parameters where a learning transition is expected.
d, N = 10, 1024
gamma = 1.0
S = tuple(range(2)) # Target feature: k=2 parity

# Fundamental scaling parameters from the theory.
sigma_w = 1.0
sigma_a = 1.0

# --- Derived variances based on the paper's definitions ---
g_w_sq = sigma_w / d
g_w = np.sqrt(g_w_sq)
g_a_sq = sigma_a / (N**gamma)
g_a = np.sqrt(g_a_sq)

print(f"--- Parameters ---")
print(f"d={d}, N={N}, k={len(S)}, gamma={gamma}")
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

# --- Activation Function Definition ---
def relu(x):
    """ReLU activation function."""
    return torch.nn.functional.relu(x)

# --- 2. Fast Boolean Sample Bank for Expectations (on GPU) ---
# This bank is used to compute expectations over the input data x.
X_bank = (2 * torch.randint(0, 2, size=(200_000, d), device=device) - 1).float()

def Sigma(w, activation_fn):
    """Computes Sigma(w) = E[phi(w^T x)^2] using the static bank."""
    z = X_bank @ w
    return torch.mean(activation_fn(z)**2)

def chi(A, X):
    """Computes the parity function chi_A(x) on a matrix of inputs."""
    if not A: # Handle empty tuple for chi_emptyset
        return torch.ones(X.shape[0], 1, device=X.device)
    return torch.prod(X[:, A], axis=1)

def J_A(w, A, activation_fn):
    """Computes J_A(w) = E[phi(w^T x) * chi_A(x)] using the static bank."""
    z = X_bank @ w
    return torch.mean(activation_fn(z) * chi(A, X_bank))

# --- 3. Log-Probability and Potential Energy ---
def log_p_w(w, m_S, kappa, S, activation_fn):
    """
    Calculates the log-probability of w, log p(w), which is -S_eff(w).
    This function implements the integrated-out action formula.
    """
    Sigma_w = Sigma(w, activation_fn)
    J_S_w = J_A(w, S, activation_fn)
    kappa_sq = kappa**2

    s_prior_w = (d / (2.0 * sigma_w)) * torch.dot(w, w)
    prior_a_coeff = (N**gamma) / sigma_a
    denom_A = prior_a_coeff + (Sigma_w / (kappa_sq + 1e-20))
    log_det_term = 0.5 * torch.log(denom_A)
    numerator_B_base = J_S_w * (1.0 - m_S)
    exp_term_numerator_B_sq = (numerator_B_base**2) / (kappa_sq * kappa_sq + 1e-20)
    exp_term = exp_term_numerator_B_sq / (4.0 * denom_A)

    return -s_prior_w - log_det_term + exp_term

def potential_energy(w, m_S, kappa, S, activation_fn):
    """ The potential energy is the negative log probability. """
    return -log_p_w(w, m_S, kappa, S, activation_fn)

# --- 4. Hamiltonian Monte Carlo (HMC) Sampler ---
def sample_w_hmc_then_a(m_S, kappa, n_samp, burn, S, activation_fn, step_size=1e-3, leapfrog_steps=10):
    """ Samples w via Hamiltonian Monte Carlo (HMC), then samples a from p(a|w). """
    w = torch.randn(d, device=device) * g_w
    W_samples, A_samples = [], []

    for t in range(burn + n_samp):
        w_current = w.clone()
        p = torch.randn_like(w_current)
        p_current = p.clone()

        # Leapfrog Integration
        w.requires_grad_(True)
        grad_U = torch.autograd.grad(potential_energy(w, m_S, kappa, S, activation_fn), w)[0]
        w.requires_grad_(False)
        
        p = p - (step_size / 2) * grad_U
        for _ in range(leapfrog_steps):
            w = w + step_size * p
            w.requires_grad_(True)
            grad_U = torch.autograd.grad(potential_energy(w, m_S, kappa, S, activation_fn), w)[0]
            w.requires_grad_(False)
            if _ != leapfrog_steps - 1:
                p = p - step_size * grad_U
        p = p - step_size * grad_U
        p = -p

        # Metropolis-Hastings Acceptance Step
        U_current = potential_energy(w_current, m_S, kappa, S, activation_fn)
        K_current = 0.5 * torch.dot(p_current, p_current)
        U_proposal = potential_energy(w, m_S, kappa, S, activation_fn)
        K_proposal = 0.5 * torch.dot(p, p)
        
        log_accept_prob = (U_current + K_current) - (U_proposal + K_proposal)

        if torch.log(torch.rand(1, device=device)) >= log_accept_prob:
            w = w_current # Reject: stay at the old state

        # Collect samples after burn-in
        if t >= burn:
            w_sample = w.clone()
            Sigma_w = Sigma(w_sample, activation_fn)
            J_S_w = J_A(w_sample, S, activation_fn)
            
            prior_a_coeff = (N**gamma) / sigma_a
            var_a_inv = prior_a_coeff + (Sigma_w / (kappa**2 + 1e-12))
            var_a = 1.0 / var_a_inv
            mean_a_numerator = (J_S_w * (1.0 - m_S)) / (kappa**2 + 1e-12)
            mean_a = var_a * mean_a_numerator
            
            a_sample = torch.randn(1, device=device) * torch.sqrt(var_a) + mean_a
            
            W_samples.append(w_sample)
            A_samples.append(a_sample)
            
    if not W_samples:
        return torch.empty(0, d, device=device), torch.empty(0, 1, device=device)

    return torch.stack(W_samples), torch.stack(A_samples)

# --- 5. Self-Consistency Loop with HMC ---
def solve_kappa_hmc(kappa, S, activation_fn, m0=0.0, iters=50, damping=0.1):
    """ Solves the self-consistency equation for m_S using the HMC sampler. """
    m_S = m0
    history = [m_S]
    
    print(f"\n--- Solving for kappa = {kappa:.4e} (starting m_S = {m0:.4f}) ---")
    print(f"Using Hamiltonian Monte Carlo (HMC) Solver")

    for i in range(iters):
        W_samples, A_samples = sample_w_hmc_then_a(
            m_S, kappa, n_samp=6000, burn=2000, S=S, activation_fn=activation_fn
        )
        
        with torch.no_grad():
            J_S_w_all = torch.stack([J_A(w, S, activation_fn) for w in W_samples])
            m_S_new = N * torch.mean(A_samples * J_S_w_all)
            
            m_S_update = (1 - damping) * m_S + damping * m_S_new.item()
            m_S = np.clip(m_S_update, -1.0, 1.0) # m_S should be in [-1, 1]
            history.append(m_S)
        
        print(f"Iter {i+1:3d}: m_S={m_S:.8f}")

        if i > 5 and abs(history[-1] - history[-2]) < 1e-6:
            print(f"CONVERGED: Value stabilized at m_S = {m_S:.8f}")
            break
            
    return m_S, history

# --- 6. Plotting Functions ---
def create_summary_plot(kappa_0_values, m_S_values, S_k, act_name, output_dir):
    """Generates a summary plot of m_S vs. kappa_0."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Phase Transition (HMC Solver): $m_S$ vs. $\\kappa_0$ ({act_name.upper()}, |S|={S_k}, d={d})", fontsize=18)

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
    filename = f"summary_phase_transition_{act_name}_k{S_k}_d{d}_HMC.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=200)
    plt.show()

# --- 7. Main Execution ---
if __name__ == "__main__":
    output_dir = "/home/goring/mean_field_langevin/MCMC_Pinf/results/3107_hmcd10k2" 
    activation_choice = 'relu'
    kappa_0_values =  [0.1]# np.logspace(2, -2, 20)

    os.makedirs(output_dir, exist_ok=True)
    activation_fn = relu
    
    print(f"\nRunning with HMC SOLVER for {activation_choice.upper()}, |S|={len(S)}")
    
    converged_m_S_values = []
    m_S_guess = 1

    # Scan from high noise (m_S=0) to low noise (m_S>0) for stability.
    for kappa_0 in sorted(kappa_0_values, reverse=True):
        kappa = kappa_0 * (N**(1.0 - gamma))
        
        m_S, history = solve_kappa_hmc(kappa, S, activation_fn, m0=m_S_guess)
        converged_m_S_values.append(m_S)
        # Use the converged value as the guess for the next, slightly different kappa
        m_S_guess = m_S if np.isfinite(m_S) else 0.0
        
    final_m_S_values = list(reversed(converged_m_S_values))
    final_kappa_0s = sorted(kappa_0_values)

    results_data = {
        "S_k": len(S), "d": d, "N": N, "gamma": gamma,
        "activation": activation_choice,
        "sigma_w": sigma_w, "sigma_a": sigma_a,
        "kappa_0_values": final_kappa_0s, 
        "m_S_values": final_m_S_values
    }
    
    results_filename = f"results_data_{activation_choice}_k{len(S)}_d{d}_HMC.json"
    with open(os.path.join(output_dir, results_filename), 'w') as f:
        json.dump(results_data, f, indent=4)
    
    create_summary_plot(final_kappa_0s, final_m_S_values, 
                        S_k=len(S), act_name=activation_choice, output_dir=output_dir)