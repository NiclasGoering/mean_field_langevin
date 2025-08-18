import math
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpyro.distributions as dist
import matplotlib.pyplot as plt
import os
import json
from rich.console import Console
from rich.table import Table
import gc
import optax
from jax import value_and_grad
from scipy.optimize import brentq

# --- JAX and Console Configuration ---
jax.config.update("jax_enable_x64", True)
console = Console()

class DMFT_SaddlePoint_Solver:
    """
    Calculates DMFT quantities using a saddle-point approximation on the 1/N corrected action.
    """
    def __init__(self, params: dict, key: jax.random.PRNGKey):
        self.params = params
        self.key = key
        self.static_args = {
            "p": params,
            "x_data": self._sample_inputs(params["n_samples_J_and_Sigma"]),
            "S_indices": jax.random.permutation(key, jnp.arange(params["d"]))[:params["k"]]
        }
        console.print(f"Solver initialized. Target parity k={params['k']} on indices: {self.static_args['S_indices']}")
        
        opt_steps = self.params.get("optimization_steps", 2000)
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=1e-6, peak_value=params.get("learning_rate", 1e-3),
            warmup_steps=int(opt_steps * 0.1), decay_steps=opt_steps, end_value=1e-7
        )
        self.optimizer = optax.adam(learning_rate=lr_schedule)

    def _sample_inputs(self, n_samples: int) -> jnp.ndarray:
        self.key, subkey = jax.random.split(self.key)
        return jax.random.choice(subkey, jnp.array([-1.0, 1.0]), shape=(n_samples, self.params["d"]))

    @staticmethod
    @jax.jit
    def _phi(z: jnp.ndarray) -> jnp.ndarray:
        return jnp.maximum(0, z)

    @staticmethod
    @jax.jit
    def _calculate_expectations(w: jnp.ndarray, x_data: jnp.ndarray, S_indices: jnp.ndarray):
        pre_activation = jnp.dot(x_data, w)
        phi_values = DMFT_SaddlePoint_Solver._phi(pre_activation)
        Sigma_w = jnp.mean(phi_values**2)
        chi_S = jnp.prod(x_data[:, S_indices], axis=1)
        J_S_w = jnp.mean(phi_values * chi_S)
        return Sigma_w, J_S_w

    def calculate_saddle_point(self, m_S: float, chi_SS: float, w_init: jnp.ndarray = None):
        """Finds the optimal w* for a given m_S and chi_SS using the corrected action."""
        p = self.params

        def loss_fn(w):
            Sigma_w, J_S_w = self._calculate_expectations(w, self.static_args['x_data'], self.static_args['S_indices'])
            log_prior = dist.Normal(jnp.zeros(p['d']), jnp.sqrt(p['sigma_w'] / p['d'])).log_prob(w).sum()
            
            # --- 1/N Correction (Onsager Term) ---
            # The self-interaction term is now renormalized by the system's susceptibility.
            Sigma_corrected = Sigma_w - chi_SS * J_S_w**2
            
            # Define Gaussian parameters for 'a' using the corrected action
            alpha = (p["N"] ** p["gamma"] / p["sigma_a"]) + (Sigma_corrected / (p["kappa"] ** 2))
            beta = ((1.0 - m_S) * J_S_w) / (p["kappa"] ** 2)
            
            # The effective action after integrating out 'a'
            log_potential = -0.5 * jnp.log(alpha) + 0.5 * (beta**2 / alpha)
            log_posterior = log_prior + log_potential
            
            symm_break_strength = p.get("symm_break_strength", 0.0)
            symmetry_breaking_term = symm_break_strength * J_S_w
            
            return -log_posterior - symmetry_breaking_term

        loss_and_grad_fn = jax.jit(value_and_grad(loss_fn))
        
        if w_init is None:
            self.key, subkey = jax.random.split(self.key)
            w_init = jax.random.normal(subkey, (p['d'],)) * jnp.sqrt(p['sigma_w'] / p['d'])
        
        opt_state = self.optimizer.init(w_init)
        w_current = w_init
        
        opt_steps = p.get("optimization_steps", 2000)
        for step in range(opt_steps):
            loss, grads = loss_and_grad_fn(w_current)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            w_current = optax.apply_updates(w_current, updates)
            
        return w_current
    
    def calculate_susceptibility(self, m_S: float, chi_SS: float, w_star: jnp.ndarray):
        """Calculates the new susceptibility chi'_SS based on the current system state."""
        p = self.params
        Sigma_star, J_S_star = self._calculate_expectations(w_star, self.static_args['x_data'], self.static_args['S_indices'])

        # Corrected alpha term from the new effective action
        Sigma_corrected = Sigma_star - chi_SS * J_S_star**2
        alpha_prime = (p["N"] ** p["gamma"] / p["sigma_a"]) + (Sigma_corrected / (p["kappa"] ** 2))
        beta = ((1.0 - m_S) * J_S_star) / (p["kappa"] ** 2)

        # Variance and mean for the posterior of 'a'
        sigma_a_sq = 1.0 / alpha_prime
        mu_a = beta * sigma_a_sq
        
        # Calculate <a^2> = Var(a) + <_a_^2
        a_sq_avg = sigma_a_sq + mu_a**2
        
        # Definition of susceptibility
        chi_ss_next = (p["N"] / p["kappa"]**2) * a_sq_avg * J_S_star**2
        return chi_ss_next

    def calculate_F_map(self, m_S: float, chi_SS: float, w_star: jnp.ndarray):
        """Calculates the value of F(m_S) for the root-finder."""
        p = self.params
        Sigma_star, J_S_star = self._calculate_expectations(w_star, self.static_args['x_data'], self.static_args['S_indices'])
        
        # Note: We use the UNCORRECTED denominator for F(m_S) as per the original definition
        denom_star = (p["N"] ** p["gamma"] / p["sigma_a"]) + (Sigma_star / (p["kappa"] ** 2))
        final_expectation = jnp.where(denom_star > 1e-9, (J_S_star**2 / denom_star), 0.0)
        
        F_val = p["N"] * ((1.0 - m_S) / (p["kappa"] ** 2)) * final_expectation
        return float(F_val)


def solve_tap_equations(solver: DMFT_SaddlePoint_Solver, w_init_guess: jnp.ndarray = None):
    """
    Solves the nested self-consistency problem for m_S and chi_SS.
    """
    console.print(f"Solving TAP equations for κ={solver.params['kappa']}...")

    state = {'best_w': w_init_guess, 'best_chi': 1.0} # Start with a reasonable guess for chi_SS

    def H(m_S: float) -> float:
        """The function for the outer root-finding loop on m_S."""
        m_S_clipped = np.clip(m_S, 1e-9, 1.0 - 1e-9)
        console.print(f"  [bold]Outer Loop: Testing H(m_S = {m_S_clipped:.6f})[/bold]")

        # --- Inner Loop for chi_SS ---
        chi_ss_guess = state['best_chi'] # Use annealed guess
        for i in range(50): # Max 50 iterations for inner loop
            console.print(f"    [dim]Inner Loop Iter {i+1}: Guess chi_SS = {chi_ss_guess:.6f}[/dim]")
            # 1. Find the saddle point w* for the current (m_S, chi_SS)
            w_star = solver.calculate_saddle_point(m_S_clipped, chi_ss_guess, w_init=state['best_w'])
            
            # 2. Calculate the new susceptibility based on this w*
            chi_ss_next = solver.calculate_susceptibility(m_S_clipped, chi_ss_guess, w_star)
            
            # 3. Check for convergence
            residual = abs(chi_ss_next - chi_ss_guess)
            if residual < 1e-6:
                console.print(f"    [green]Inner loop converged. Final chi_SS = {chi_ss_next:.6f}[/green]")
                chi_ss_guess = chi_ss_next
                break
            
            # 4. Update guess with damping
            chi_ss_guess = 0.5 * chi_ss_guess + 0.5 * chi_ss_next
        
        state['best_w'] = w_star
        state['best_chi'] = chi_ss_guess
        
        # Calculate F(m_S) using the converged state
        F_val = solver.calculate_F_map(m_S_clipped, chi_ss_guess, w_star)
        
        residual_H = F_val - m_S_clipped
        console.print(f"  Result: F(m_S)={F_val:.6f} -> H(m_S) = {residual_H:.6f}")
        return residual_H

    try:
        a, b = 1e-6, 1.0 - 1e-6
        H_a, H_b = H(a), H(b)

        if np.sign(H_a) == np.sign(H_b):
            console.print("[yellow]Warning: H(a) and H(b) have the same sign. Root not bracketed.[/yellow]")
            return (0.0, state['best_w']) if H_a > 0 else (1.0, state['best_w'])

        solution_m_S = brentq(H, a, b, xtol=1e-8, rtol=1e-8)
        console.print(f"[bold green]Root-finder converged! Solution m_S = {solution_m_S:.8f}[/bold green]")
        return solution_m_S, state['best_w']

    except Exception as e:
        console.print(f"[bold red]Root-finding failed: {e}[/bold red]")
        return np.nan, None

if __name__ == "__main__":
    #check_gpu_capabilities()
    
    save_dir = f"/home/goring/mean_field_langevin/MCMC_Pinf_1N/results/0508_d30k4"
    os.makedirs(save_dir, exist_ok=True)
    console.print(f"Results will be saved in: '{os.path.abspath(save_dir)}'")
    
    base_parameters = {
        "d": 30, "N": 1024, "k": 4, "sigma_a": 1.0, "sigma_w": 1.0, "gamma": 1.0,
        "n_samples_J_and_Sigma": 8192, "learning_rate": 1e-3,
    }
    
    kappa_values = sorted([0.1, 0.05, 0.02, 0.01, 0.0075, 0.005, 0.0025, 0.001, 1e-4], reverse=True) 

    results_data = {}
    w_anneal_guess = None 
    master_key = jax.random.PRNGKey(42)

    for kappa in kappa_values:
        master_key, solver_key = jax.random.split(master_key)
        console.rule(f"[bold magenta]Running for κ = {kappa}[/bold magenta]")
        
        current_params = base_parameters.copy()
        current_params["kappa"] = kappa
        current_params["symm_break_strength"] = 0.2
        
        if kappa <= 0.01:
            current_params["optimization_steps"] = 30000
        else:
            current_params["optimization_steps"] = 10000

        solver = DMFT_SaddlePoint_Solver(current_params, solver_key)
        
        final_m_S, final_w = solve_tap_equations(solver, w_init_guess=w_anneal_guess)
        
        if final_w is not None and not np.isnan(final_m_S):
            results_data[str(kappa)] = {"mean": final_m_S, "std": 0.0, "all_results": [final_m_S]}
            w_anneal_guess = final_w 
            
            table = Table(title=f"Statistics for κ = {kappa}")
            table.add_column("Metric", style="cyan"); table.add_column("Value", style="magenta")
            table.add_row("Converged m_S", f"{final_m_S:.6f}")
            console.print(table)
        else:
            console.print(f"[bold red]Run failed for κ = {kappa}.[/bold red]")
            results_data[str(kappa)] = {"mean": np.nan, "std": np.nan, "all_results": []}
            w_anneal_guess = None 

        gc.collect()
    
    console.rule("[bold]All simulations finished", style="green")
    json_filepath = os.path.join(save_dir, "final_stats_vs_kappa.json")
    with open(json_filepath, 'w') as f:
        # Custom serializer
        def serializer(obj):
            if isinstance(obj, (jnp.ndarray, np.ndarray)): return obj.item()
            return float(obj)
        serializable_results = {k: {**v, 'all_results': [serializer(i) for i in v['all_results']]} for k, v in results_data.items()}
        json.dump(serializable_results, f, indent=4)
    console.print(f"Final results data saved to: [green]'{json_filepath}'[/green]")

    kappas_sorted = sorted([float(k) for k in results_data.keys()])
    means = [results_data[str(k)]['mean'] for k in kappas_sorted]
    
    if any(m is not None and not np.isnan(m) for m in means):
        plt.figure(figsize=(12, 7))
        plt.plot(kappas_sorted, means, 'o-', label=r'Converged $m_S$')
        plt.axhline(y=1.0, color='r', linestyle='--', label=r'$m_S=1$ (Perfect Learning)')
        plt.xscale('log')
        plt.xlabel(r"Noise Parameter $\kappa$")
        plt.ylabel(r"Order Parameter $m_S$")
        plt.title(f"DMFT Order Parameter vs. Noise (1/N Corrected TAP Model)")
        plt.grid(True, which="both", ls="--")
        plt.gca().invert_xaxis()
        plt.legend()
        plt.tight_layout()
        summary_plot_path = os.path.join(save_dir, "summary_m_s_vs_kappa.png")
        plt.savefig(summary_plot_path)
        console.print(f"Saved final summary plot to: [green]'{summary_plot_path}'[/green]")
        plt.show()