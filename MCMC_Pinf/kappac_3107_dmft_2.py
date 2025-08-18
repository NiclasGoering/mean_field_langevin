import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

class DMFT_Solver:
    """
    Solves the DMFT equations for a two-layer network using a robust,
    fixed-point iteration scheme with a corrected MCMC sampler.
    """
    def __init__(self, params):
        self.params = params.copy() # Use a copy to allow for adaptive changes
        # Randomly choose k feature indices for the teacher
        self.S_indices = np.random.choice(params['d'], params['k'], replace=False)

        # MCMC state
        self.w_current = np.random.randn(params['d']) * np.sqrt(params['sigma_w'] / params['d'])
        self.log_prob_current = None
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
        chi_S_values = np.prod(x_samples[:, self.S_indices], axis=1)
        pre_activations = x_samples @ w
        phi_values = self._phi(pre_activations)
        return np.mean(phi_values * chi_S_values)

    def _log_prob_unnormalized(self, w, m_S, x_batch):
        """
        Calculates -S_eff(w) using a *fixed* batch of x_samples.
        """
        p = self.params
        Sigma_w = self._Sigma(w, x_batch)
        J_S_w = self._J_S(w, x_batch)
        denom_term = (p['N']**p['gamma'] / p['sigma_a']) + (Sigma_w / p['kappa']**2)
        log_term = 0.5 * np.log(denom_term)
        J_eff_w = (1 - m_S) * J_S_w
        exp_term = (J_eff_w**2 / p['kappa']**4) / (4 * denom_term)
        prior_term = (p['d'] / (2 * p['sigma_w'])) * np.sum(w**2)
        return - (log_term - exp_term + prior_term)

    def _calculate_F(self, m_S):
        """
        Calculates F(m_S) using a corrected Metropolis-Hastings MCMC sampler.
        """
        p = self.params
        
        # FIX 4: Draw ONE common batch of x_samples for this entire function call.
        x_batch_mcmc = self._sample_inputs(p['n_samples_J_and_Sigma'])
        
        # FIX 1: Recompute current log-prob for the *new* m_S.
        self.log_prob_current = self._log_prob_unnormalized(self.w_current, m_S, x_batch_mcmc)

        w_samples = []
        n_accept = 0
        
        for i in range(p['mcmc_samples']):
            w_proposal = self.w_current + np.random.randn(p['d']) * p['mcmc_step_size']
            lp_proposal = self._log_prob_unnormalized(w_proposal, m_S, x_batch_mcmc)
            
            if np.log(np.random.rand()) < lp_proposal - self.log_prob_current:
                self.w_current, self.log_prob_current = w_proposal, lp_proposal
                n_accept += 1

            if i >= p['mcmc_samples'] // 4:  # Burn-in period
                w_samples.append(self.w_current)
            
            # FIX 3: Adaptive MCMC step size
            if (i + 1) % 100 == 0:
                rate = n_accept / (i + 1)
                if rate > 0.5:
                    p['mcmc_step_size'] *= 1.05
                elif rate < 0.2:
                    p['mcmc_step_size'] *= 0.95
        
        self.mcmc_acceptance_rate = n_accept / p['mcmc_samples']
        
        # Estimate the expectation using the collected samples
        # Use a larger batch of x_samples for the final, more accurate estimate
        x_batch_final = self._sample_inputs(p['n_samples_J_and_Sigma'] * 5)
        
        # Vectorized calculation for performance
        Σs = np.array([self._Sigma(w, x_batch_final) for w in w_samples])
        Js = np.array([self._J_S(w, x_batch_final) for w in w_samples])
        
        denom = (p['N']**p['gamma'] / p['sigma_a']) + Σs / p['kappa']**2
        Fm = p['N'] * ((1 - m_S) / p['kappa']**2) * np.mean(Js**2 / denom)
        
        return Fm

    def solve_fixed_point(self, m0, max_iter=500, tol=1e-5):
        """
        FIX 2: Solves for m_S using damped fixed-point iteration.
        """
        m = m0
        p = self.params
        history = [m]
        print(f"{'Iteration':<10} | {'m_S':<10} | {'F(m_S)':<10} | {'Change':<10} | {'Accept%':<10}")
        print("-" * 59)

        for t in range(max_iter):
            Fm = self._calculate_F(m)
            m_new = (1 - p['damping']) * m + p['damping'] * Fm
            
            change = abs(m_new - m)
            print(f"{t:<10} | {m:<10.4f} | {Fm:<10.4f} | {change:<10.2e} | {self.mcmc_acceptance_rate:<9.2%}")
            
            history.append(m_new)
            if change < tol:
                print("\nConvergence reached.")
                return m_new, history
            
            m = m_new
            
        print("\nWarning: Maximum iterations reached without convergence.")
        return m, history

# --- Main execution ---
if __name__ == '__main__':
    parameters = {
        'd': 40,
        'N': 1024,
        'k': 4,
        'kappa': 0.000001,
        'sigma_a': 1.0,
        'sigma_w': 1.0,
        'gamma': 1.0,
        'n_samples_J_and_Sigma': 2000,
        'mcmc_samples': 5000,
        'mcmc_step_size': 0.05, # Initial guess, will be adapted
        'damping': 0.1 # Damping factor for fixed-point iteration
    }

    solver = DMFT_Solver(parameters)
    m_initial = 0.02
    
    print("Starting DMFT simulation with damped fixed-point iteration...")
    m_final, m_history = solver.solve_fixed_point(m_initial)
    print(f"\nFinal fixed-point: m_S ≈ {m_final:.5f}")

    # --- Plotting Results ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(m_history, 'o-', label=r'Iteration value of $m_S$', color='b')
    ax.axhline(y=m_final, color='r', linestyle='--', label=f'Final Fixed Point ≈ {m_final:.4f}')
    
    ax.set_xlabel('Iteration Step', fontsize=14)
    ax.set_ylabel(r'Order Parameter ($m_S$)', fontsize=14)
    ax.set_title('DMFT Fixed-Point Iteration', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True)
    plt.tight_layout()
    plt.show()