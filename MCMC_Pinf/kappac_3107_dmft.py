import numpy as np
import math # Correct import for math functions
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm

class DMFT_Solver:
    """
    Solves the DMFT equations for a two-layer network learning a single parity mode.
    This version uses Monte Carlo sampling to compute J_S and Sigma,
    based on their exact definitions.
    """
    def __init__(self, params):
        self.params = params
        # Randomly choose k feature indices for the teacher
        self.S_indices = np.random.choice(params['d'], params['k'], replace=False)
        
        # MCMC state
        self.w_current = np.random.randn(params['d']) * np.sqrt(params['sigma_w'] / params['d'])
        self.log_prob_current = None # Will be computed on first call
        self.mcmc_acceptance_rate = 0.0

    def _phi(self, z):
        """ReLU activation function."""
        return np.maximum(0, z)

    def _sample_inputs(self, n_samples):
        """Generates a batch of random input vectors x in {-1, 1}^d."""
        return np.random.choice([-1, 1], size=(n_samples, self.params['d']))

    def _Sigma(self, w, x_samples):
        """Computes Sigma(w) = E_x[phi(w^T x)^2] via Monte Carlo."""
        pre_activations = x_samples @ w
        phi_values_sq = self._phi(pre_activations)**2
        return np.mean(phi_values_sq)

    def _J_S(self, w, x_samples):
        """Computes J_S(w) = E_x[phi(w^T x) chi_S(x)] via Monte Carlo."""
        # chi_S(x) = product of x_i for i in S
        chi_S_values = np.prod(x_samples[:, self.S_indices], axis=1)
        
        pre_activations = x_samples @ w
        phi_values = self._phi(pre_activations)
        
        return np.mean(phi_values * chi_S_values)

    def _log_prob_unnormalized(self, w, m_S):
        """
        Calculates the unnormalized log probability, which is -S_eff(w) 
        after integrating out 'a'.
        """
        p = self.params
        
        # Generate a common batch of x samples for this step
        x_samples = self._sample_inputs(p['n_samples_J_and_Sigma'])
        
        # Calculate key components using sampling
        Sigma_w = self._Sigma(w, x_samples)
        J_S_w = self._J_S(w, x_samples)
        
        # Common denominator term
        denom_term = (p['N']**p['gamma'] / p['sigma_a']) + (Sigma_w / p['kappa']**2)
        
        # Effective action terms from your paper
        log_term = 0.5 * np.log(denom_term)
        
        J_eff_w = (1 - m_S) * J_S_w
        exp_term = (J_eff_w**2 / p['kappa']**4) / (4 * denom_term)
        
        prior_term = (p['d'] / (2 * p['sigma_w'])) * np.sum(w**2)
        
        # Negative of the effective action
        return - (log_term - exp_term + prior_term)

    def _calculate_F(self, m_S):
        """
        Calculates the function F(m_S) using Metropolis-Hastings MCMC.
        """
        p = self.params
        samples = []
        n_accept = 0
        
        if self.log_prob_current is None:
            self.log_prob_current = self._log_prob_unnormalized(self.w_current, m_S)
            
        for _ in range(p['mcmc_samples']):
            w_proposal = self.w_current + np.random.randn(p['d']) * p['mcmc_step_size']
            log_prob_proposal = self._log_prob_unnormalized(w_proposal, m_S)
            
            log_alpha = log_prob_proposal - self.log_prob_current
            if np.log(np.random.rand()) < log_alpha:
                self.w_current = w_proposal
                self.log_prob_current = log_prob_proposal
                n_accept += 1
            
            samples.append(self.w_current)
        
        self.mcmc_acceptance_rate = n_accept / p['mcmc_samples']
        
        burn_in = p['mcmc_samples'] // 4
        valid_samples = samples[burn_in:]
        
        # --- Calculate Expectation using the generated MCMC samples of w ---
        expectation_sum = 0.0
        # Generate a single, larger batch of x_samples for the final expectation
        # to reduce noise in the estimate of F(m_S).
        x_samples_final = self._sample_inputs(p['n_samples_J_and_Sigma'] * 5)
        
        for w in valid_samples:
            Sigma_w = self._Sigma(w, x_samples_final)
            J_S_w = self._J_S(w, x_samples_final)
            numerator = J_S_w**2
            denominator = (p['N']**p['gamma'] / p['sigma_a']) + (Sigma_w / p['kappa']**2)
            expectation_sum += numerator / denominator
            
        expectation_avg = expectation_sum / len(valid_samples)
        
        F_mS = p['N'] * ((1 - m_S) / p['kappa']**2) * expectation_avg
        return F_mS

    def dmS_dtau(self, tau, m_S):
        """The right-hand side of the ODE: dmS/dtau = -mS + F(mS)."""
        F_mS = self._calculate_F(m_S[0]) # m_S is a list from solve_ivp
        
        self.pbar.set_description(
            f"τ={tau:.2f}, m_S={m_S[0]:.8f}, F(m_S)={F_mS:.8f}, "
            f"Acceptance={self.mcmc_acceptance_rate:.8%}"
        )
        self.pbar.update(1)
        
        return -m_S + F_mS

    def solve(self, m0, t_span, t_eval):
        """Solves the ODE."""
        num_steps = len(t_eval) if t_eval is not None else 100
        self.pbar = tqdm(total=num_steps, desc="Solving ODE")
        
        sol = solve_ivp(
            fun=self.dmS_dtau,
            t_span=t_span,
            y0=[m0],
            t_eval=t_eval,
            method='RK45'
        )
        self.pbar.close()
        return sol

# --- Main execution ---
if __name__ == '__main__':
    parameters = {
        'd': 40,
        'N': 1024,
        'k': 4,  # Now works for any k
        'kappa': 0.0001,
        'sigma_a': 1.0,
        'sigma_w': 1.0,
        'gamma': 1.0,
        'n_samples_J_and_Sigma': 1000, # Samples for inner E_x expectation.
        'mcmc_samples': 2000,          # MCMC samples for outer E_w expectation.
        'mcmc_step_size': 0.05
    }

    solver = DMFT_Solver(parameters)
    
    m_initial = 0.0
    t_simulation_span = [0, 30]
    t_evaluation_points = np.linspace(t_simulation_span[0], t_simulation_span[1], 50)

    print("Starting DMFT simulation with Monte Carlo estimation...")
    solution = solver.solve(m_initial, t_simulation_span, t_evaluation_points)
    print("Simulation finished.")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(solution.t, solution.y[0], 'o-', label=r'$m_S(\tau)$', color='b')
    ax.set_xlabel(r'Fictitious Time ($\tau$)', fontsize=14)
    ax.set_ylabel(r'Order Parameter ($m_S$)', fontsize=14)
    ax.set_title(f'DMFT Evolution for k={parameters["k"]} Parity', fontsize=16)
    
    fixed_point = solution.y[0, -1]
    ax.axhline(y=fixed_point, color='r', linestyle='--', label=f'Fixed Point ≈ {fixed_point:.4f}')
    
    ax.legend(fontsize=12)
    ax.grid(True)
    plt.tight_layout()
    plt.show()