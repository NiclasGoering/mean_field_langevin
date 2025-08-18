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
    Calculates the DMFT update F(m_S) by finding the mode of p(w|m_S).
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
            init_value=1e-6,
            peak_value=params.get("learning_rate", 1e-3),
            warmup_steps=int(opt_steps * 0.1),
            decay_steps=opt_steps,
            end_value=1e-7
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

    def calculate_F_saddlepoint(self, m_S: float, w_init: jnp.ndarray = None):
        """
        This version now includes diagnostic printing for the inner optimization loop.
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
        
        opt_steps = p.get("optimization_steps", 2000)
        print_interval = max(1, opt_steps // 10) # Print ~10 times

        for step in range(opt_steps):
            loss, grads = loss_and_grad_fn(w_current)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            w_current = optax.apply_updates(w_current, updates)

            # --- DIAGNOSTIC PRINTING ---
            if step % print_interval == 0 or step == opt_steps - 1:
                console.print(f"      [dim]Inner Opt Step {step:>{len(str(opt_steps))}}/{opt_steps} | Loss: {loss:.6f}[/dim]")

        w_star = w_current
        
        # --- ADDED DIAGNOSTIC ---
        Sigma_star, J_S_star = self._calculate_expectations(w_star, self.static_args['x_data'], self.static_args['S_indices'])
        ratio = p["N"]*(J_S_star**2) / (Sigma_star + 1e-9)
        ratio = ratio/(1+ratio)
        console.print(f"      [cyan]Diagnostic: J_S^2 / Sigma_w = {ratio:.6f}[/cyan]")
        
        denom_star = (p["N"] ** p["gamma"] / p["sigma_a"]) + (Sigma_star / (p["kappa"] ** 2))
        final_expectation = jnp.where(denom_star > 1e-9, (J_S_star**2 / denom_star), 0.0)
        F_val = p["N"] * ((1.0 - m_S) / (p["kappa"] ** 2)) * final_expectation
        
        return float(F_val), w_star

def find_fixed_point_with_root_finder(solver: DMFT_SaddlePoint_Solver, w_init_guess: jnp.ndarray = None):
    """
    Finds the fixed point by solving H(m_S) = F(m_S) - m_S = 0.
    """
    console.print(f"Searching for fixed point for κ={solver.params['kappa']} using a root-finder.")

    state = {'best_w': w_init_guess}

    def H(m_S: float) -> float:
        """The function whose root we want to find: H(m_S) = F(m_S) - m_S"""
        # --- FIX 1: Corrected the clipping range for m_S ---
        m_S_clipped = np.clip(m_S, 1e-9, 1.0 - 1e-9)
        console.print(f"  [bold]Testing H(m_S = {m_S_clipped:.6f})[/bold]")
        F_val, w_star = solver.calculate_F_saddlepoint(m_S_clipped, w_init=state['best_w'])
        state['best_w'] = w_star
        residual = F_val - m_S_clipped
        console.print(f"  Result: F(m_S)={F_val:.6f} -> H(m_S) = {residual:.6f}")
        return residual

    try:
        a, b = 1e-6, 1.0 - 1e-6
        H_a, H_b = H(a), H(b)

        if np.sign(H_a) == np.sign(H_b):
            console.print("[yellow]Warning: H(a) and H(b) have the same sign. Root not bracketed.[/yellow]")
            if H_a > 0:
                 console.print("[yellow]Trivial solution m_S=0 is likely stable.[/yellow]")
                 return 0.0, state['best_w']
            else:
                 return 1.0, state['best_w']

        solution_m_S = brentq(H, a, b, xtol=1e-8, rtol=1e-8)
        console.print(f"[bold green]Root-finder converged! Solution m_S = {solution_m_S:.8f}[/bold green]")
        _, final_w = solver.calculate_F_saddlepoint(solution_m_S, w_init=state['best_w'])
        return solution_m_S, final_w

    except Exception as e:
        console.print(f"[bold red]Root-finding failed: {e}[/bold red]")
        return np.nan, None

if __name__ == "__main__":
    check_gpu_capabilities()
    
    save_dir = f"./results/0805_final_fix"
    os.makedirs(save_dir, exist_ok=True)
    console.print(f"Results will be saved in: '{os.path.abspath(save_dir)}'")
    
    base_parameters = {
        "d": 30, "N": 1024, "k": 4, "sigma_a": 1.0, "sigma_w": 1.0, "gamma": 1.0,
        "n_samples_J_and_Sigma": 8192,
        "learning_rate": 1e-3,
    }
    
    # Run for just one kappa value as requested
    kappa_values = [1,1e-2,1e-3,1e-4] 

    results_data = {}
    w_anneal_guess = None 
    master_key = jax.random.PRNGKey(42)

    for kappa in kappa_values:
        master_key, solver_key = jax.random.split(master_key)
        console.rule(f"[bold magenta]Running for κ = {kappa}[/bold magenta]")
        
        current_params = base_parameters.copy()
        current_params["kappa"] = kappa

        current_params["symm_break_strength"] = 0.2
        console.print(f"[bold cyan]Symmetry breaking strength = {current_params['symm_break_strength']:.4f}[/bold cyan]")

        if kappa <= 0.01:
            current_params["optimization_steps"] = 20000 
            console.print(f"[yellow]Critical regime: Using {current_params['optimization_steps']} optimization steps.[/yellow]")
        elif kappa <= 0.1:
            current_params["optimization_steps"] = 15000
            console.print(f"[yellow]Intermediate regime: Using {current_params['optimization_steps']} optimization steps.[/yellow]")
        else:
            current_params["optimization_steps"] = 5000

        solver = DMFT_SaddlePoint_Solver(current_params, solver_key)
        
        final_m_S, final_w = find_fixed_point_with_root_finder(solver, w_init_guess=w_anneal_guess)
        
        if final_w is not None and not np.isnan(final_m_S):
            results_data[str(kappa)] = {"mean": final_m_S, "std": 0.0, "all_results": [final_m_S]}
            w_anneal_guess = final_w 
            
            table = Table(title=f"Statistics for κ = {kappa}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            table.add_row("Converged m_S", f"{final_m_S:.6f}")
            console.print(table)
        else:
            console.print(f"[bold red]Run failed for κ = {kappa}.[/bold red]")
            results_data[str(kappa)] = {"mean": np.nan, "std": np.nan, "all_results": []}
            w_anneal_guess = None 

        gc.collect()
    
    # --- FIX 2: Corrected JSON saving logic ---
    console.rule("[bold]All simulations finished", style="green")
    json_filepath = os.path.join(save_dir, "final_stats_vs_kappa.json")
    with open(json_filepath, 'w') as f:
        # Custom serializer to handle JAX arrays and other numpy types
        def serializer(obj):
            if isinstance(obj, (jnp.ndarray, np.ndarray)):
                return obj.item()
            return float(obj)
        
        serializable_results = {}
        for k, v in results_data.items():
            serializable_results[k] = {
                'mean': serializer(v['mean']),
                'std': serializer(v['std']),
                'all_results': [serializer(i) for i in v['all_results']]
            }
        json.dump(serializable_results, f, indent=4)
    console.print(f"Final results data saved to: [green]'{json_filepath}'[/green]")

    kappas_sorted = sorted([float(k) for k in results_data.keys()])
    means = [results_data[str(k)]['mean'] for k in kappas_sorted]
    
    if any(m is not None and not np.isnan(m) for m in means):
        plt.figure(figsize=(12, 7))
        plt.errorbar(kappas_sorted, means, yerr=0, fmt='o-', capsize=5, label=r'Converged $m_S$')
        plt.axhline(y=1.0, color='r', linestyle='--', label=r'$m_S=1$ (Perfect Learning)')
        if len(kappas_sorted) > 1:
            plt.xscale('log')
            plt.gca().invert_xaxis()
        plt.xlabel(r"Noise Parameter $\kappa$")
        plt.ylabel(r"Order Parameter $m_S$")
        plt.title(f"DMFT Order Parameter vs. Noise (Root-Finder with Diagnostics)")
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.tight_layout()
        summary_plot_path = os.path.join(save_dir, "summary_m_s_vs_kappa_errorbar.png")
        plt.savefig(summary_plot_path)
        console.print(f"Saved final summary plot to: [green]'{summary_plot_path}'[/green]")
        plt.show()
    else:
        console.print("[yellow]No data to plot for the final summary.[/yellow]")