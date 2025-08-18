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
    Solves the DMFT self-consistency equations using a saddle-point approximation
    combined with an annealing strategy for robustness.
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
        
        # Use a learning rate schedule for the optimizer
        opt_steps = self.params.get("optimization_steps", 2000)
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=1e-6,
            peak_value=params.get("learning_rate", 1e-3),
            warmup_steps=int(opt_steps * 0.1),
            decay_steps=opt_steps,
            end_value=1e-7
        )
        self.optimizer = optax.adam(learning_rate=lr_schedule)

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
        """
        p = self.params

        def loss_fn(w):
            Sigma_w, J_S_w = self._calculate_expectations(w, self.static_args['x_data'], self.static_args['S_indices'])
            log_prior = dist.Normal(jnp.zeros(p['d']), jnp.sqrt(p['sigma_w'] / p['d'])).log_prob(w).sum()
            denom = (p["N"] ** p["gamma"] / p["sigma_a"]) + (Sigma_w / (p["kappa"] ** 2))
            log_term = 0.5 * jnp.log(denom)
            J_eff = (1.0 - m_S) * J_S_w
            exp_term_raw = (J_eff**2 / p["kappa"]**4) / (2.0 * denom)
            exp_term = jnp.clip(exp_term_raw, a_max=120.0)
            log_potential = -log_term + exp_term
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
        
        # Removed the tqdm progress bar for a cleaner log focused on m_S updates.
        for _ in range(p.get("optimization_steps", 2000)):
            loss, grads = loss_and_grad_fn(w_current)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            w_current = optax.apply_updates(w_current, updates)

        w_star = w_current

        Sigma_star, J_S_star = self._calculate_expectations(w_star, self.static_args['x_data'], self.static_args['S_indices'])
        denom_star = (p["N"] ** p["gamma"] / p["sigma_a"]) + (Sigma_star / (p["kappa"] ** 2))
        final_expectation = jnp.where(denom_star > 1e-9, (J_S_star**2 / denom_star), 0.0)
        F_val = p["N"] * ((1.0 - m_S) / (p["kappa"] ** 2)) * final_expectation
        
        return float(F_val), w_star

def run_fixed_point_search(solver: DMFT_SaddlePoint_Solver, fp_config: dict, initial_m_S: float, w_init: jnp.ndarray = None):
    """Performs a stable fixed-point search for a single run."""
    m_S_current = initial_m_S
    w_current = w_init

    for i in range(fp_config["max_iterations"]):
        console.print(f"--- Iteration {i+1}/{fp_config['max_iterations']} ---")
        m_S_clipped = np.clip(m_S_current, -1.0, 1.0 - 1e-9)
        m_S_current = m_S_clipped
        
        console.print(f"m_S(in) = {m_S_current:.8f}. Starting inner optimization...")
        m_S_next_raw, w_star = solver.calculate_F_saddlepoint(m_S_current, w_init=w_current)
        w_current = w_star
        console.print(f"Optimization finished. F(m_S) = {m_S_next_raw:.8f}")
        
        if np.isnan(m_S_next_raw):
            console.print(f"[bold red]Solver returned NaN for initial m_S={initial_m_S:.4f}.[/bold red]")
            return np.nan, None

        residual = abs(m_S_next_raw - m_S_current)
        
        if residual < fp_config["tolerance"]:
            console.print(f"[bold green]Run converged in {i+1} iterations for initial m_S={initial_m_S:.4f}, final m_S={m_S_current:.6f}[/bold green]")
            return m_S_current, w_star
        
        m_S_next = m_S_current + fp_config["damping_alpha"] * (m_S_next_raw - m_S_current)
        console.print(f"m_S(out) = {m_S_next:.8f} (Residual = {residual:.2e})")
        m_S_current = m_S_next

    console.print(f"[yellow]Run did not converge for initial m_S={initial_m_S:.4f}, final m_S={m_S_current:.6f}[/yellow]")
    return m_S_current, w_star

if __name__ == "__main__":
    check_gpu_capabilities()
    
    save_dir = f"/home/goring/mean_field_langevin/MCMC_Pinf/results/0408_opt_long1"
    os.makedirs(save_dir, exist_ok=True)
    console.print(f"Results will be saved in: '{os.path.abspath(save_dir)}'")
    
    base_parameters = {
        "d": 30, "N": 1024, "k": 4, "sigma_a": 1.0, "sigma_w": 1.0, "gamma": 1.0,
        "n_samples_J_and_Sigma": 8192,
        "learning_rate": 2e-3,
        "optimization_steps": 5000,
        "symm_break_strength": 0.0, # Default to zero
    }
    
    fixed_point_config = {
        "max_iterations": 60,
        "tolerance": 1e-7,
        "damping_alpha": 0.25, 
    }
    
    # --- Experiment Configuration ---
    NUM_INITS_PER_KAPPA = 1
    kappa_values = sorted([10.0, 1.0, 0.1, 0.05, 0.01,0.0075,0.005,0.0025, 0.001, 1e-4,1e-5], reverse=True) 

    # --- Main Experiment Loop ---
    results_data = {}
    w_anneal_guess = None 
    master_key = jax.random.PRNGKey(42)

    for kappa in kappa_values:
        master_key, solver_key, init_key = jax.random.split(master_key, 3)
        console.rule(f"[bold magenta]Running {NUM_INITS_PER_KAPPA} initializations for κ = {kappa}[/bold magenta]")
        
        current_params = base_parameters.copy()
        current_params["kappa"] = kappa
        
        # Adaptive strategy for critical regions
        if kappa <= 0.05:
            console.print("[yellow]Critical kappa detected. Using smaller damping, more opt steps, and symmetry breaking.[/yellow]")
            fixed_point_config["damping_alpha"] = 0.1
            current_params["optimization_steps"] = 10000
            current_params["symm_break_strength"] = 0.01 # Activate the guiding field
        else:
            fixed_point_config["damping_alpha"] = 0.25
            current_params["optimization_steps"] = 5000
            current_params["symm_break_strength"] = 0.0 # Turn it off when not needed

        solver = DMFT_SaddlePoint_Solver(current_params, solver_key)
        
        kappa_results_m_S = []
        kappa_results_w = []
        
        initial_m_S_values = jax.random.uniform(init_key, shape=(NUM_INITS_PER_KAPPA,))

        for i, m_S_start in enumerate(initial_m_S_values):
            console.print(f"\n--- Starting run {i+1}/{NUM_INITS_PER_KAPPA} for κ={kappa} with initial m_S = {m_S_start:.4f} ---")
            final_m_S, final_w = run_fixed_point_search(solver, fixed_point_config, initial_m_S=float(m_S_start), w_init=w_anneal_guess)
            
            if not np.isnan(final_m_S):
                kappa_results_m_S.append(final_m_S)
                kappa_results_w.append(final_w)

        if kappa_results_m_S:
            mean_m_S = np.mean(kappa_results_m_S)
            std_m_S = np.std(kappa_results_m_S)
            results_data[str(kappa)] = {
                "mean": mean_m_S,
                "std": std_m_S,
                "all_results": kappa_results_m_S
            }
            w_anneal_guess = jnp.mean(jnp.array(kappa_results_w), axis=0)
            
            table = Table(title=f"Statistics for κ = {kappa}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            table.add_row("Mean m_S", f"{mean_m_S:.6f}")
            table.add_row("Std Dev m_S", f"{std_m_S:.6f}")
            table.add_row("Successful Runs", f"{len(kappa_results_m_S)}/{NUM_INITS_PER_KAPPA}")
            console.print(table)

        else:
            console.print(f"[bold red]All runs failed for κ = {kappa}.[/bold red]")
            results_data[str(kappa)] = {"mean": np.nan, "std": np.nan, "all_results": []}
            w_anneal_guess = None

        gc.collect()
        
    console.rule("[bold]All simulations finished", style="green")
    json_filepath = os.path.join(save_dir, "final_stats_vs_kappa.json")
    with open(json_filepath, 'w') as f:
        serializable_results = {k: {**v, 'all_results': [float(i) for i in v['all_results']]} for k, v in results_data.items()}
        json.dump(serializable_results, f, indent=4)
    console.print(f"Final results data saved to: [green]'{json_filepath}'[/green]")

    kappas_sorted = sorted([float(k) for k in results_data.keys()])
    means = [results_data[str(k)]['mean'] for k in kappas_sorted]
    stds = [results_data[str(k)]['std'] for k in kappas_sorted]
    
    if any(not np.isnan(m) for m in means):
        plt.figure(figsize=(12, 7))
        plt.errorbar(kappas_sorted, means, yerr=stds, fmt='o-', capsize=5, label=r'Converged $m_S$ (mean ± std)')
        plt.xscale('log')
        plt.xlabel(r"Noise Parameter κ")
        plt.ylabel(r"Order Parameter $m_S$")
        plt.title(f"DMFT Order Parameter vs. Noise ({NUM_INITS_PER_KAPPA} runs per κ)")
        plt.grid(True, which="both", ls="--")
        plt.gca().invert_xaxis()
        plt.legend()
        plt.tight_layout()
        summary_plot_path = os.path.join(save_dir, "summary_m_s_vs_kappa_errorbar.png")
        plt.savefig(summary_plot_path)
        console.print(f"Saved final summary plot to: [green]'{summary_plot_path}'[/green]")
        plt.show()
    else:
        console.print("[yellow]No data to plot for the final summary.[/yellow]")
