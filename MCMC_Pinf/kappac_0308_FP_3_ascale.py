import math
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json
from rich.console import Console
from rich.table import Table
import gc
from datetime import datetime

# Configure JAX to use float64 for high precision calculations
jax.config.update("jax_enable_x64", True)
# Initialize Rich Console for beautiful terminal output
console = Console()

def check_gpu_capabilities():
    """Checks for JAX-visible GPUs and prints information."""
    try:
        # THIS IS THE LINE THAT ENABLES PARALLEL GPU EXECUTION
        numpyro.set_host_device_count(len(jax.devices()))
        devices = jax.devices()
        console.print("[bold green]JAX backend devices found:[/bold green]")
        for i, device in enumerate(devices):
            console.print(f"  Device {i}: [cyan]{device.platform.upper()} ({device.device_kind})[/cyan]")
        if not any(d.platform == 'gpu' for d in devices):
            console.print("[yellow]WARNING: No GPU detected by JAX. Running on CPU.[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error checking JAX devices: {e}[/bold red]")

class DMFT_NUTS_Solver:
    """
    Solves the DMFT self-consistency equations using the NUTS sampler in NumPyro.
    
    *** FINAL CORRECTED VERSION ***
    This version correctly implements BOTH the user-provided formulas with the 1/2 factors
    AND the numerically stable rescaling (a' = a/kappa). This is fast and accurate.
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
        phi_values = DMFT_NUTS_Solver._phi(pre_activation)
        Sigma_w = jnp.mean(phi_values**2)
        chi_S = jnp.prod(x_data[:, S_indices], axis=1)
        J_S_w = jnp.mean(phi_values * chi_S)
        return Sigma_w, J_S_w

    @staticmethod
    def numpyro_model(m_S: float, p: dict, x_data: jnp.ndarray, S_indices: jnp.ndarray):
        w = numpyro.sample("w", dist.Normal(jnp.zeros(p['d']), jnp.sqrt(p['sigma_w'] / p['d'])))
        Sigma_w, J_S_w = DMFT_NUTS_Solver._calculate_expectations(w, x_data, S_indices)
        kappa = p["kappa"]

        # Denominator for the log term from the stable derivation
        denom_log = (kappa**2 * p["N"]**p["gamma"] / (2.0 * p["sigma_a"])) + (Sigma_w / 2.0)
        log_term = 0.5 * jnp.log(denom_log)
        
        # Denominator for the exp term from the stable derivation
        denom_exp = (kappa**2 * p["N"]**p["gamma"] / p["sigma_a"]) + Sigma_w
        
        J_eff = (1.0 - m_S) * J_S_w
        exp_term_raw = (J_eff**2 / kappa**2) / denom_exp
        
        exp_term = jnp.clip(exp_term_raw, a_max=50.0)
        
        log_prob = -log_term + exp_term
        numpyro.factor("log_potential", log_prob)

    def calculate_F(self, m_S: float, mcmc_config: dict):
        self.key, subkey = jax.random.split(self.key)
        target_accept_prob = 0.85
        
        kernel = NUTS(self.numpyro_model, adapt_step_size=True, target_accept_prob=target_accept_prob)
        mcmc = MCMC(kernel, 
                      num_warmup=mcmc_config["num_warmup"], 
                      num_samples=mcmc_config["num_samples"], 
                      num_chains=mcmc_config["num_chains"],
                      progress_bar=True)

        mcmc.run(subkey, m_S=m_S, **self.static_args)

        w_samples = mcmc.get_samples()["w"]
        w_samples_flat = w_samples.reshape(-1, self.params['d'])
        console.print(f"MCMC finished. Shape of w samples: {w_samples_flat.shape}")

        vmap_expectations = jax.vmap(
            lambda w: self._calculate_expectations(w, self.static_args['x_data'], self.static_args['S_indices'])
        )
        Sigma_samples, J_S_samples = vmap_expectations(w_samples_flat)

        p = self.params
        kappa = p["kappa"]
        
        # Denominator for the F(m_S) update rule from stable derivation
        denom_F = (kappa**2 * p["N"]**p["gamma"] / (2.0 * p["sigma_a"])) + (Sigma_samples / 2.0)
        final_expectation = jnp.mean(J_S_samples**2 / denom_F)

        # The stable F_val update rule
        F_val = p["N"] * (1.0 - m_S) * final_expectation
        
        return float(F_val)

def run_fixed_point_search(solver: DMFT_NUTS_Solver, fp_config: dict, initial_m_S: float):
    console.print(f"\n[bold green]----------- Starting Fixed-Point Search for κ={solver.params['kappa']} -----------[/bold green]")
    console.print(f"Initial guess m_S = {initial_m_S:.6f}")
    
    m_S_current = initial_m_S
    history = [m_S_current]

    for i in range(fp_config["max_iterations"]):
        console.rule(f"Iteration {i+1}/{fp_config['max_iterations']}")
        
        m_S_clipped = np.clip(m_S_current, -1.0, 1.0)
        if m_S_clipped != m_S_current:
            console.print(f"[yellow]Warning: Clipped m_S from {m_S_current:.4f} to {m_S_clipped:.4f}[/yellow]")
            m_S_current = m_S_clipped
        
        m_S_next_raw = solver.calculate_F(m_S_current, fp_config["mcmc_config"])
        
        if np.isnan(m_S_next_raw):
            console.print("[bold red]Solver returned NaN. Halting iteration.[/bold red]")
            return np.nan, history

        change = m_S_next_raw - m_S_current
        m_S_next = m_S_current + fp_config["damping_alpha"] * change
        history.append(m_S_next)
        final_change = m_S_next - m_S_current
        
        console.print(f"m_S(old) = {m_S_current:.8f} -> F(m_S) = {m_S_next_raw:.8f} -> m_S(new) = {m_S_next:.8f} [Change = {final_change:+.8f}]")

        if abs(final_change) < fp_config["tolerance"]:
            console.print(f"\n[bold green]Convergence reached in {i+1} iterations![/bold green]")
            return m_S_next, history
        
        m_S_current = m_S_next
        gc.collect()

    console.print(f"\n[bold yellow]Maximum iterations ({fp_config['max_iterations']}) reached. Did not converge.[/bold yellow]")
    return m_S_current, history

def create_and_save_plot(history, final_m_S, kappa, save_dir):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history, 'o-', label='m_S value per iteration')
    if not np.isnan(final_m_S):
        ax.axhline(final_m_S, color='red', ls='--', label=f'Final value ≈ {final_m_S:.6f}')
    ax.set_xlabel("Iteration Step")
    ax.set_ylabel(r"Order Parameter $m_S$")
    ax.set_title(f"DMFT Fixed-Point Iteration History for κ = {kappa}")
    ax.legend()
    ax.set_ylim(-0.1, 1.1)
    plt.tight_layout()
    plot_filepath = os.path.join(save_dir, f"iteration_history_kappa_{kappa}.png")
    plt.savefig(plot_filepath)
    plt.close(fig)
    console.print(f"Saved iteration history plot to: [green]'{plot_filepath}'[/green]")


if __name__ == "__main__":
    check_gpu_capabilities()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./dmft_results_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    console.print(f"Results will be saved in: '{os.path.abspath(save_dir)}'")
    
    base_parameters = {
        "d": 30, "N": 1024, "k": 4, "sigma_a": 1.0, "sigma_w": 1.0, "gamma": 1.0,
        "n_samples_J_and_Sigma": 4096,
    }
    
    fixed_point_config = {
        "max_iterations": 30,
        "tolerance": 1e-5,
        "damping_alpha": 0.5,
        "mcmc_config": {
            "num_warmup": 1000,
            "num_samples": 4000,
            "num_chains": 3, 
        }
    }
    
    kappa_values = sorted([10.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001], reverse=False) 

    results_data = {}
    m_S_guess = 0.5
    master_key = jax.random.PRNGKey(42)

    for kappa in kappa_values:
        master_key, solver_key = jax.random.split(master_key)
        console.rule(f"[bold]Running for κ = {kappa}", style="magenta")
        
        current_params = base_parameters.copy()
        current_params["kappa"] = kappa
        
        solver = DMFT_NUTS_Solver(current_params, solver_key)
        
        if kappa < 0.05 and m_S_guess < 0.8:
            console.print(f"[yellow]Low kappa detected. Overriding guess from {m_S_guess:.4f} to 0.95 to find correct fixed point.[/yellow]")
            m_S_guess = 0.95

        final_m_S, history = run_fixed_point_search(solver, fixed_point_config, initial_m_S=m_S_guess)
        
        create_and_save_plot(history, final_m_S, kappa, save_dir)
        results_data[str(kappa)] = final_m_S
        
        if not np.isnan(final_m_S):
            m_S_guess = final_m_S
        
        gc.collect()
        
        table = Table(title=f"Summary for κ = {kappa}")
        table.add_column("Converged Fixed Point (m_S)", justify="right", style="cyan")
        table.add_row(f"{final_m_S:.6f}" if not np.isnan(final_m_S) else "[red]NaN[/red]")
        console.print(table)
        
    console.rule("[bold]All simulations finished", style="green")
    json_filepath = os.path.join(save_dir, "final_roots_vs_kappa.json")
    with open(json_filepath, 'w') as f: json.dump(results_data, f, indent=4)
    console.print(f"Final results data saved to: [green]'{json_filepath}'[/green]")

    kappas_sorted = sorted([float(k) for k in results_data.keys() if not np.isnan(results_data[k])])
    m_s_values = [results_data[str(k)] for k in kappas_sorted]
    
    if m_s_values:
        plt.figure(figsize=(10, 6))
        plt.plot(kappas_sorted, m_s_values, 'o-', label=r'Converged $m_S$')
        plt.xscale('log')
        plt.xlabel(r"Noise Parameter κ")
        plt.ylabel(r"Order Parameter $m_S$")
        plt.title("DMFT Order Parameter vs. Noise (Numerically Stable NUTS Solver)")
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