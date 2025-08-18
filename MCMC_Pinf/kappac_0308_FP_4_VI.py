import math
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json
from rich.console import Console
from rich.table import Table
import gc
import optax

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

class DMFT_VI_Solver:
    """
    Solves the DMFT self-consistency equations using Variational Inference (VI).
    This version uses an advanced learning rate schedule and a symmetry-breaking field
    to handle difficult optimization landscapes at low kappa.
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
        
        vi_steps = self.params.get("vi_steps", 5000)
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=1e-6,
            peak_value=params.get("learning_rate", 1e-3),
            warmup_steps=int(vi_steps * 0.1),
            decay_steps=vi_steps,
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
        phi_values = DMFT_VI_Solver._phi(pre_activation)
        Sigma_w = jnp.mean(phi_values**2)
        chi_S = jnp.prod(x_data[:, S_indices], axis=1)
        J_S_w = jnp.mean(phi_values * chi_S)
        return Sigma_w, J_S_w

    @staticmethod
    def model(m_S: float, p: dict, x_data: jnp.ndarray, S_indices: jnp.ndarray):
        """
        The probabilistic model (defines the true posterior).
        """
        w_prior = dist.Independent(dist.Normal(jnp.zeros(p['d']), jnp.sqrt(p['sigma_w'] / p['d'])), 1)
        w = numpyro.sample("w", w_prior)
        
        Sigma_w, J_S_w = DMFT_VI_Solver._calculate_expectations(w, x_data, S_indices)
        denom = (p["N"] ** p["gamma"] / p["sigma_a"]) + (Sigma_w / (p["kappa"] ** 2))
        log_term = 0.5 * jnp.log(denom)
        J_eff = (1.0 - m_S) * J_S_w
        exp_term_raw = (J_eff**2 / p["kappa"]**4) / (2.0 * denom)
        exp_term = jnp.clip(exp_term_raw, a_max=120.0)
        
        # STABILITY FIX: Add a symmetry-breaking field to guide the optimizer away from the trivial w=0 solution.
        # This provides a small, explicit reward for aligning with the target function.
        symm_break_strength = p.get("symm_break_strength", 0.0)
        symmetry_breaking_term = symm_break_strength * J_S_w
        
        log_potential = -log_term + exp_term + symmetry_breaking_term
        numpyro.factor("log_potential", log_potential)

    def calculate_F_vi(self, m_S: float, svi_params_init: dict = None):
        """
        Calculates F(m_S) by running Variational Inference to approximate the posterior.
        """
        p = self.params
        self.key, svi_key, sample_key = jax.random.split(self.key, 3)

        def guide(m_S, p, x_data, S_indices):
            w_loc_init = svi_params_init['w_loc'] if svi_params_init else jnp.zeros(p['d'])
            w_scale_tril_init = svi_params_init['w_scale_tril'] if svi_params_init else jnp.identity(p['d']) * 0.1
            
            w_loc = numpyro.param("w_loc", w_loc_init)
            w_scale_tril = numpyro.param("w_scale_tril", w_scale_tril_init)
            numpyro.sample("w", dist.MultivariateNormal(loc=w_loc, scale_tril=w_scale_tril))

        svi = SVI(self.model, guide, self.optimizer, loss=Trace_ELBO())
        svi_state = svi.init(svi_key, m_S=m_S, **self.static_args)
        
        console.print("Starting Variational Inference optimization...")
        @jax.jit
        def svi_step(svi_state):
            svi_state, loss = svi.update(svi_state, m_S=m_S, **self.static_args)
            return svi_state, loss

        for _ in tqdm(range(p.get("vi_steps", 5000))):
            svi_state, _ = svi_step(svi_state)
        
        final_svi_params = svi.get_params(svi_state)
        console.print(f"VI optimization finished.")

        guide_dist = dist.MultivariateNormal(loc=final_svi_params["w_loc"], scale_tril=final_svi_params["w_scale_tril"])
        w_samples = guide_dist.sample(sample_key, (p.get("vi_num_samples", 10000),))

        vmap_expectations = jax.vmap(
            lambda w: self._calculate_expectations(w, self.static_args['x_data'], self.static_args['S_indices'])
        )
        Sigma_samples, J_S_samples = vmap_expectations(w_samples)

        denom_samples = (p["N"] ** p["gamma"] / p["sigma_a"]) + (Sigma_samples / (p["kappa"] ** 2))
        final_expectation = jnp.mean(J_S_samples**2 / denom_samples)

        F_val = p["N"] * ((1.0 - m_S) / (p["kappa"] ** 2)) * final_expectation
        
        return float(F_val), final_svi_params

def run_fixed_point_search(solver: DMFT_VI_Solver, fp_config: dict, initial_m_S: float, svi_params_init: dict = None):
    """Performs a stable fixed-point search using the VI solver."""
    console.print(f"\n[bold green]----------- Starting Fixed-Point Search for κ={solver.params['kappa']} -----------[/bold green]")
    console.print(f"Initial guess m_S = {initial_m_S:.6f}")
    
    m_S_current = initial_m_S
    history = [m_S_current]
    svi_params_current = svi_params_init

    for i in range(fp_config["max_iterations"]):
        console.rule(f"Iteration {i+1}/{fp_config['max_iterations']}", style="blue")
        
        m_S_clipped = np.clip(m_S_current, -1.0, 1.0 - 1e-9)
        if m_S_clipped != m_S_current:
            console.print(f"[yellow]Warning: Clipped m_S from {m_S_current:.4f} to {m_S_clipped:.4f}[/yellow]")
            m_S_current = m_S_clipped
        
        m_S_next_raw, final_svi_params = solver.calculate_F_vi(m_S_current, svi_params_init=svi_params_current)
        svi_params_current = final_svi_params
        
        if np.isnan(m_S_next_raw):
            console.print("[bold red]Solver returned NaN. Halting iteration.[/bold red]")
            return np.nan, history, None

        change = m_S_next_raw - m_S_current
        m_S_next = m_S_current + fp_config["damping_alpha"] * change
        history.append(m_S_next)
        final_change = m_S_next - m_S_current
        
        console.print(f"m_S(old) = {m_S_current:.8f} -> F(m_S) = {m_S_next_raw:.8f} -> m_S(new) = {m_S_next:.8f} [Change = {final_change:+.8f}]")

        if abs(final_change) < fp_config["tolerance"]:
            console.print(f"\n[bold green]Convergence reached in {i+1} iterations![/bold green]")
            return m_S_next, history, final_svi_params
        
        m_S_current = m_S_next
        gc.collect()

    console.print(f"\n[bold yellow]Maximum number of iterations ({fp_config['max_iterations']}) reached.[/bold yellow]")
    return m_S_current, history, svi_params_current

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
    
    save_dir = f"./results/0408_VI_Test_SymmBreak"
    os.makedirs(save_dir, exist_ok=True)
    console.print(f"Results will be saved in: '{os.path.abspath(save_dir)}'")
    
    base_parameters = {
        "d": 30, "N": 1024, "k": 4, "sigma_a": 1.0, "sigma_w": 1.0, "gamma": 1.0,
        "n_samples_J_and_Sigma": 8192,
        "learning_rate": 2e-3, # Peak learning rate
        "vi_steps": 5000,
        "vi_num_samples": 20000,
        "symm_break_strength": 0.0, # Default to zero, will be turned on for low kappa
    }
    
    fixed_point_config = {
        "max_iterations": 40,
        "tolerance": 1e-6,
        "damping_alpha": 0.2, 
    }
    
    kappa_values = sorted([10.0, 1.0, 0.5, 0.1, 0.05, 0.01, 0.001, 1e-4], reverse=True) 

    results_data = {}
    m_S_guess = 0.0
    svi_params_guess = None
    master_key = jax.random.PRNGKey(42)

    for kappa in kappa_values:
        master_key, solver_key = jax.random.split(master_key)
        
        current_params = base_parameters.copy()
        current_params["kappa"] = kappa

        # STABILITY FIX: Use more optimization steps and a symmetry-breaking field for the hardest problems.
        if kappa <= 0.01:
            console.print(f"[yellow]Low kappa ({kappa}) detected. Increasing VI steps and adding symmetry-breaking field.[/yellow]")
            current_params["vi_steps"] = 10000
            current_params["symm_break_strength"] = 0.1
        
        solver = DMFT_VI_Solver(current_params, solver_key)
        final_m_S, history, final_svi_params = run_fixed_point_search(solver, fixed_point_config, initial_m_S=m_S_guess, svi_params_init=svi_params_guess)
        create_and_save_plot(history, final_m_S, kappa, save_dir)

        results_data[str(kappa)] = final_m_S
        
        if not np.isnan(final_m_S):
            m_S_guess = final_m_S
            svi_params_guess = final_svi_params
        else:
            m_S_guess = 0.5 
            svi_params_guess = None
        
        gc.collect()
        
    console.rule("[bold]All simulations finished", style="green")
    json_filepath = os.path.join(save_dir, "final_roots_vs_kappa_vi.json")
    with open(json_filepath, 'w') as f: json.dump(results_data, f, indent=4)
    console.print(f"Final results data saved to: [green]'{json_filepath}'[/green]")

    kappas_sorted = sorted([float(k) for k in results_data.keys() if results_data[k] is not None and not np.isnan(results_data[k])])
    m_s_values = [results_data[str(k)] for k in results_data[k]]
    
    if m_s_values:
        plt.figure(figsize=(10, 6))
        plt.plot(kappas_sorted, m_s_values, 'o-', label=r'Converged $m_S$')
        plt.xscale('log')
        plt.xlabel(r"Noise Parameter κ")
        plt.ylabel(r"Order Parameter $m_S$")
        plt.title("DMFT Order Parameter vs. Noise (Variational Inference Solver)")
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
