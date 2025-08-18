import math
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpyro.distributions as dist
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json
from rich.console import Console
from rich.table import Table
import gc
import optax
from jax import value_and_grad

# --- JAX and Console Configuration ---
jax.config.update("jax_enable_x64", True)
console = Console()

def check_gpu_capabilities():
    """Checks for JAX-visible GPUs and prints information."""
    try:
        devices = jax.devices()
        console.print("[bold green]JAX backend devices found:[/bold green]")
        for i, device in enumerate(devices):
            console.print(f"  Device {i}: [cyan]{device.platform.upper()} ({device.device_kind})[/cyan]")
        if not any(d.platform == 'gpu' for d in devices):
            console.print("[yellow]WARNING: No GPU detected by JAX. Running on CPU.[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error checking JAX devices: {e}[/bold red]")

class DMFT_SaddlePoint_Solver:
    """
    Solves the DMFT self-consistency equations using a saddle-point approximation.
    Instead of sampling the posterior with MCMC, it finds the mode (peak) of the
    posterior via gradient-based optimization. This is highly effective for low-kappa regimes.
    """
    def __init__(self, params: dict, key: jax.random.PRNGKey):
        self.params = params
        self.key = key
        # Static arguments for the model
        self.static_args = {
            "p": params,
            "x_data": self._sample_inputs(params["n_samples_J_and_Sigma"]),
            "S_indices": jax.random.permutation(key, jnp.arange(params["d"]))[:params["k"]]
        }
        console.print(f"Solver initialized. Target parity k={params['k']} on indices: {self.static_args['S_indices']}")
        
        # Initialize the optimizer once
        self.optimizer = optax.adam(learning_rate=params.get("learning_rate", 1e-3))

    def _sample_inputs(self, n_samples: int) -> jnp.ndarray:
        """Generates a batch of random input vectors x in {-1, 1}^d."""
        self.key, subkey = jax.random.split(self.key)
        return jax.random.choice(subkey, jnp.array([-1.0, 1.0]), shape=(n_samples, self.params["d"]))

    @staticmethod
    @jax.jit
    def _phi(z: jnp.ndarray) -> jnp.ndarray:
        """The ReLU activation function."""
        return jnp.maximum(0, z)

    @staticmethod
    @jax.jit
    def _calculate_expectations(w: jnp.ndarray, x_data: jnp.ndarray, S_indices: jnp.ndarray):
        """Calculates Sigma and J_S expectations for a single weight vector w."""
        pre_activation = jnp.dot(x_data, w)
        phi_values = DMFT_SaddlePoint_Solver._phi(pre_activation)
        Sigma_w = jnp.mean(phi_values**2)
        chi_S = jnp.prod(x_data[:, S_indices], axis=1)
        J_S_w = jnp.mean(phi_values * chi_S)
        return Sigma_w, J_S_w

    def calculate_F_saddlepoint(self, m_S: float, w_init: jnp.ndarray = None):
        """
        Calculates F(m_S) by finding the mode of the posterior p(w|m_S) via optimization.
        This replaces the MCMC sampling step.
        """
        p = self.params

        # 1. Define the loss function to be minimized: the negative log-posterior (effective action)
        def loss_fn(w):
            # Calculate expectations needed for the action
            Sigma_w, J_S_w = self._calculate_expectations(w, self.static_args['x_data'], self.static_args['S_indices'])
            
            # Log-prior term for the weights
            log_prior = dist.Normal(jnp.zeros(p['d']), jnp.sqrt(p['sigma_w'] / p['d'])).log_prob(w).sum()

            # Potential term from the effective action after integrating out 'a'
            denom = (p["N"] ** p["gamma"] / p["sigma_a"]) + (Sigma_w / (p["kappa"] ** 2))
            log_term = 0.5 * jnp.log(denom)
            J_eff = (1.0 - m_S) * J_S_w
            exp_term_raw = (J_eff**2 / p["kappa"]**4) / (2.0 * denom)
            exp_term = jnp.clip(exp_term_raw, a_max=120.0) # Keep stability clip
            
            log_potential = -log_term + exp_term
            
            # The total log posterior probability
            log_posterior = log_prior + log_potential
            
            # We want to MINIMIZE the negative log posterior
            return -log_posterior

        # 2. Set up and run the optimization
        loss_and_grad_fn = jax.jit(value_and_grad(loss_fn))
        
        # Initialize weights: either from scratch or from a provided starting point (for annealing)
        if w_init is None:
            self.key, subkey = jax.random.split(self.key)
            w_init = jax.random.normal(subkey, (p['d'],)) * jnp.sqrt(p['sigma_w'] / p['d'])
        
        opt_state = self.optimizer.init(w_init)
        
        # Optimization loop
        w_current = w_init
        console.print("Starting optimization to find posterior mode...")
        for _ in range(p.get("optimization_steps", 2000)):
            _, grads = loss_and_grad_fn(w_current)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            w_current = optax.apply_updates(w_current, updates)
        
        w_star = w_current  # The optimized weights at the posterior mode
        final_loss = loss_fn(w_star)
        console.print(f"Optimization finished. Final loss: {final_loss:.4f}")


        # 3. Calculate F(m_S) using the single optimal w_star
        Sigma_star, J_S_star = self._calculate_expectations(w_star, self.static_args['x_data'], self.static_args['S_indices'])
        
        denom_star = (p["N"] ** p["gamma"] / p["sigma_a"]) + (Sigma_star / (p["kappa"] ** 2))
        
        # Avoid division by zero if denom_star is somehow zero
        final_expectation = jnp.where(denom_star > 1e-9, (J_S_star**2 / denom_star), 0.0)

        # This is the self-consistency equation for m_S
        F_val = p["N"] * ((1.0 - m_S) / (p["kappa"] ** 2)) * final_expectation
        
        return float(F_val), w_star

def run_fixed_point_search(solver: DMFT_SaddlePoint_Solver, fp_config: dict, initial_m_S: float, w_init: jnp.ndarray = None):
    """Performs a stable fixed-point search using the saddle-point solver."""
    console.print(f"\n[bold green]----------- Starting Fixed-Point Search for κ={solver.params['kappa']} -----------[/bold green]")
    console.print(f"Initial guess m_S = {initial_m_S:.6f}")
    
    m_S_current = initial_m_S
    history = [m_S_current]
    w_current = w_init

    for i in range(fp_config["max_iterations"]):
        console.rule(f"Iteration {i+1}/{fp_config['max_iterations']}", style="blue")
        
        # STABILITY FIX: Clip m_S away from the exact boundary of 1.0 to prevent F(m_S) from becoming exactly 0.
        m_S_clipped = np.clip(m_S_current, -1.0, 1.0 - 1e-9)
        if m_S_clipped != m_S_current:
            console.print(f"[yellow]Warning: Clipped m_S from {m_S_current:.4f} to {m_S_clipped:.4f}[/yellow]")
            m_S_current = m_S_clipped
        
        # Use the saddle-point solver. Pass w_current for warm-starting the optimization.
        m_S_next_raw, w_star = solver.calculate_F_saddlepoint(m_S_current, w_init=w_current)
        w_current = w_star # Use the result for the next iteration's warm start
        
        if np.isnan(m_S_next_raw):
            console.print("[bold red]Solver returned NaN. Halting iteration.[/bold red]")
            return np.nan, history, None

        # Apply damping for stability
        change = m_S_next_raw - m_S_current
        m_S_next = m_S_current + fp_config["damping_alpha"] * change

        history.append(m_S_next)
        final_change = m_S_next - m_S_current
        
        console.print(f"m_S(old) = {m_S_current:.8f} -> F(m_S) = {m_S_next_raw:.8f} -> m_S(new) = {m_S_next:.8f} [Change = {final_change:+.8f}]")

        if abs(final_change) < fp_config["tolerance"]:
            console.print(f"\n[bold green]Convergence reached in {i+1} iterations![/bold green]")
            return m_S_next, history, w_star
        
        m_S_current = m_S_next
        gc.collect()

    console.print(f"\n[bold yellow]Maximum number of iterations ({fp_config['max_iterations']}) reached.[/bold yellow]")
    return m_S_current, history, w_star

def create_and_save_plot(history, final_m_S, kappa, save_dir, suffix=""):
    """Generates and saves a plot of the fixed-point iteration history."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history, 'o-', label='m_S value per iteration')
    if not np.isnan(final_m_S):
        ax.axhline(final_m_S, color='red', ls='--', label=f'Final value ≈ {final_m_S:.6f}')
    ax.set_xlabel("Iteration Step")
    ax.set_ylabel(r"Order Parameter $m_S$")
    ax.set_title(f"DMFT Fixed-Point Iteration History for κ = {kappa}{suffix}")
    ax.legend()
    ax.grid(True, which="both", ls="--")
    plt.tight_layout()
    plot_filepath = os.path.join(save_dir, f"iteration_history_kappa_{kappa}{suffix.replace('=', '').replace(' ', '_')}.png")
    plt.savefig(plot_filepath)
    plt.close(fig)
    console.print(f"Saved iteration history plot to: [green]'{plot_filepath}'[/green]")

if __name__ == "__main__":
    check_gpu_capabilities()
    
    # --- Configuration ---
    save_dir = f"/home/goring/mean_field_langevin/MCMC_Pinf/results/0408_opt1"
    os.makedirs(save_dir, exist_ok=True)
    console.print(f"Results will be saved in: '{os.path.abspath(save_dir)}'")
    
    base_parameters = {
        "d": 30, "N": 1024, "k": 4, "sigma_a": 1.0, "sigma_w": 1.0, "gamma": 1.0,
        "n_samples_J_and_Sigma": 20000,
        "learning_rate": 5e-4,
        "optimization_steps": 12000,
    }
    
    fixed_point_config = {
        "max_iterations": 100,
        "tolerance": 1e-6,
        # STABILITY FIX: Use a more conservative damping factor to prevent oscillations
        "damping_alpha": 0.2, 
    }
    
    kappa_values = sorted([10.0, 1.0, 0.5, 0.1, 0.05, 0.01, 0.001, 1e-4], reverse=True) 

    # --- Main Annealing Loop ---
    results_data = {}
    m_S_guess = 0.0 # Start from the no-learning solution at high kappa
    w_guess = None
    master_key = jax.random.PRNGKey(42)

    for kappa in kappa_values:
        master_key, solver_key = jax.random.split(master_key)
        
        current_params = base_parameters.copy()
        current_params["kappa"] = kappa
        
        # --- Special Check for Multiple Minima at Lowest Kappa ---
        if kappa == 1e-4:
            console.rule(f"[bold yellow]Performing multiple start check for κ = {kappa}", style="yellow")
            multi_start_results = {}
            # Test 10 different starting points for m_S
            for start_val in np.linspace(0.05, 0.95, 10):
                solver = DMFT_SaddlePoint_Solver(current_params, solver_key)
                final_m_S, history, final_w = run_fixed_point_search(solver, fixed_point_config, initial_m_S=start_val, w_init=w_guess)
                multi_start_results[f"start_{start_val:.2f}"] = final_m_S
                create_and_save_plot(history, final_m_S, kappa, save_dir, suffix=f" (start={start_val:.2f})")
            
            # Report results in a table
            table = Table(title=f"Multiple Start Point Results for κ = {kappa}")
            table.add_column("Initial m_S", justify="right", style="cyan")
            table.add_column("Converged m_S", justify="right", style="magenta")
            for start, end in multi_start_results.items():
                table.add_row(start.split('_')[1], f"{end:.6f}" if not np.isnan(end) else "[red]NaN[/red]")
            console.print(table)
            
            # Choose the mean of successful results to continue
            successful_runs = [val for val in multi_start_results.values() if not np.isnan(val)]
            final_m_S = np.mean(successful_runs) if successful_runs else np.nan
            
        else: # --- Standard Annealing for other kappas ---
            console.rule(f"[bold]Running for κ = {kappa}", style="magenta")
            solver = DMFT_SaddlePoint_Solver(current_params, solver_key)
            final_m_S, history, final_w = run_fixed_point_search(solver, fixed_point_config, initial_m_S=m_S_guess, w_init=w_guess)
            create_and_save_plot(history, final_m_S, kappa, save_dir)

        results_data[str(kappa)] = final_m_S
        
        # Use the converged value as the guess for the next, colder kappa
        if not np.isnan(final_m_S):
            m_S_guess = final_m_S
            w_guess = final_w # Anneal the weights as well!
        else:
            # If it failed, reset the guess for the next run
            m_S_guess = 0.5 
            w_guess = None
        
        gc.collect()
        
    console.rule("[bold]All simulations finished", style="green")
    json_filepath = os.path.join(save_dir, "final_roots_vs_kappa_saddlepoint.json")
    with open(json_filepath, 'w') as f: json.dump(results_data, f, indent=4)
    console.print(f"Final results data saved to: [green]'{json_filepath}'[/green]")

    # --- Final Summary Plot ---
    kappas_sorted = sorted([float(k) for k in results_data.keys() if results_data[k] is not None and not np.isnan(results_data[k])])
    m_s_values = [results_data[str(k)] for k in kappas_sorted]
    
    if m_s_values:
        plt.figure(figsize=(10, 6))
        plt.plot(kappas_sorted, m_s_values, 'o-', label=r'Converged $m_S$')
        plt.xscale('log')
        plt.xlabel(r"Noise Parameter κ")
        plt.ylabel(r"Order Parameter $m_S$")
        plt.title("DMFT Order Parameter vs. Noise (Saddle-Point Solver)")
        plt.grid(True, which="both", ls="--")
        plt.gca().invert_xaxis()
        plt.legend()
        plt.tight_layout()
        summary_plot_path = os.path.join(save_dir, "summary_m_s_vs_kappa.png")
        plt.savefig(summary_plot_path)
        console.print(f"Saved final summary plot to: [green]'{summary_plot_path}'[/green]")
        plt.show()
    else:
        console.print("[yellow]No data to plot for the final summary.[/yellow]")
