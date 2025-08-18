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
from jax import value_and_grad, lax
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

    def calculate_saddle_point(self, m_S: float, chi_SS: float, w_init: jnp.ndarray = None, num_steps: int = -1):
        p = self.params
        opt_steps = p.get("optimization_steps", 2000) if num_steps == -1 else num_steps

        def loss_fn(w):
            Sigma_w, J_S_w = self._calculate_expectations(w, self.static_args['x_data'], self.static_args['S_indices'])
            log_prior = dist.Normal(jnp.zeros(p['d']), jnp.sqrt(p['sigma_w'] / p['d'])).log_prob(w).sum()
            
            # --- CORRECTED 1/N Onsager Term ---
            Sigma_corrected = Sigma_w - (chi_SS / p["N"]) * J_S_w**2
            
            alpha = (p["N"] ** p["gamma"] / p["sigma_a"]) + (Sigma_corrected / (p["kappa"] ** 2))
            beta = ((1.0 - m_S) * J_S_w) / (p["kappa"] ** 2)
            
            alpha = jnp.maximum(alpha, 1e-9)
            log_potential = -0.5 * jnp.log(alpha) + 0.5 * (beta**2 / alpha)
            log_posterior = log_prior + log_potential
            return -log_posterior

        if w_init is None:
            self.key, subkey = jax.random.split(self.key)
            w_init = jax.random.normal(subkey, (p['d'],)) * jnp.sqrt(p['sigma_w'] / p['d'])
        
        @jax.jit
        def optimization_step(_, state):
            w_current, opt_state = state
            grads = value_and_grad(loss_fn)(w_current)[1]
            updates, new_opt_state = self.optimizer.update(grads, opt_state)
            new_w = optax.apply_updates(w_current, updates)
            return new_w, new_opt_state

        initial_state = (w_init, self.optimizer.init(w_init))
        w_star, _ = lax.fori_loop(0, opt_steps, optimization_step, initial_state)
        return w_star
    
    def calculate_susceptibility(self, m_S: float, chi_SS: float, w_star: jnp.ndarray):
        p = self.params
        Sigma_star, J_S_star = self._calculate_expectations(w_star, self.static_args['x_data'], self.static_args['S_indices'])
        
        # --- CORRECTED 1/N Onsager Term ---
        Sigma_corrected = Sigma_star - (chi_SS / p["N"]) * J_S_star**2

        alpha_prime = (p["N"] ** p["gamma"] / p["sigma_a"]) + (Sigma_corrected / (p["kappa"] ** 2))
        alpha_prime = np.maximum(alpha_prime, 1e-9)
        beta = ((1.0 - m_S) * J_S_star) / (p["kappa"] ** 2)
        sigma_a_sq = 1.0 / alpha_prime
        mu_a = beta * sigma_a_sq
        a_sq_avg = sigma_a_sq + mu_a**2
        return (p["N"] / p["kappa"]**2) * a_sq_avg * J_S_star**2

    def calculate_F_map(self, m_S: float, w_star: jnp.ndarray):
        p = self.params
        Sigma_star, J_S_star = self._calculate_expectations(w_star, self.static_args['x_data'], self.static_args['S_indices'])
        denom_star = (p["N"] ** p["gamma"] / p["sigma_a"]) + (Sigma_star / (p["kappa"] ** 2))
        final_expectation = jnp.where(denom_star > 1e-9, (J_S_star**2 / denom_star), 0.0)
        return float(p["N"] * ((1.0 - m_S) / (p["kappa"] ** 2)) * final_expectation)

def solve_tap_equations(solver: DMFT_SaddlePoint_Solver, w_init_guess: jnp.ndarray = None):
    console.print(f"Solving TAP equations for κ={solver.params['kappa']}...")
    state = {'best_w': w_init_guess, 'best_chi': 1.0}

    def H(m_S: float) -> float:
        m_S_clipped = np.clip(m_S, 1e-9, 1.0 - 1e-9)
        console.print(f"  [bold]Outer Loop: Testing H(m_S = {m_S_clipped:.6f})[/bold]")
        chi_ss_guess = state['best_chi']
        damping_alpha = 0.1
        max_inner_steps = 200

        for i in range(max_inner_steps):
            w_star = solver.calculate_saddle_point(m_S_clipped, chi_ss_guess, w_init=state['best_w'])
            chi_ss_next = solver.calculate_susceptibility(m_S_clipped, chi_ss_guess, w_star)
            residual = abs(chi_ss_next - chi_ss_guess)
            if residual < 1e-7:
                console.print(f"    [green]Inner loop converged after {i+1} iterations. Final chi_SS = {chi_ss_next:.6f}[/green]")
                chi_ss_guess = chi_ss_next
                break
            
            chi_ss_guess = (1.0 - damping_alpha) * chi_ss_guess + damping_alpha * chi_ss_next
            chi_ss_guess = max(0, chi_ss_guess)

        if i == max_inner_steps - 1:
             console.print(f"    [yellow]Warning: Inner loop for chi_SS did not converge (residual={residual:.2e}).[/yellow]")
        
        state['best_w'], state['best_chi'] = w_star, chi_ss_guess
        
        F_val = solver.calculate_F_map(m_S_clipped, w_star)
        console.print(f"  Result: F(m_S)={F_val:.6f} -> H(m_S) = {F_val - m_S_clipped:.6f}")
        return F_val - m_S_clipped

    try:
        a, b = 1e-6, 1.0 - 1e-6
        H_a, H_b = H(a), H(b)
        
        if np.sign(H_a) == np.sign(H_b):
            console.print("[yellow]Warning: H(a) and H(b) have the same sign. Root not bracketed.[/yellow]")
            return (0.0, state['best_w']) if H_a < 0 and H_b < 0 else (1.0, state['best_w'])
        
        solution_m_S = brentq(H, a, b, xtol=1e-8, rtol=1e-8)
        console.print(f"[bold green]Root-finder converged! Solution m_S = {solution_m_S:.8f}[/bold green]")
        return solution_m_S, state['best_w']
    except Exception as e:
        console.print(f"[bold red]Root-finding failed: {e}[/bold red]")
        return np.nan, None

if __name__ == "__main__":
    check_gpu_capabilities()
    
    save_dir = "./results/tap_corrected_final"
    os.makedirs(save_dir, exist_ok=True)
    console.print(f"Results will be saved in: '{os.path.abspath(save_dir)}'")
    
    base_parameters = {
        "d": 30, "N": 1024, "k": 4, "sigma_a": 1.0, "sigma_w": 1.0, "gamma": 1.0,
        "n_samples_J_and_Sigma": 8192, "learning_rate": 1e-3,
    }
    
    #kappa_values = sorted([10.0, 1.0, 0.5, 0.1, 0.05, 0.02, 0.01, 0.0075, 0.005, 0.0025, 0.001, 1e-4], reverse=True)
    kappa_values = sorted([ 1e-4], reverse=True)
    results_data = {}
    w_anneal_guess = None 
    master_key = jax.random.PRNGKey(42)
    
    # --- JIT WARM-UP ---
    console.rule("[bold cyan]JIT Compilation Warm-up[/bold cyan]")
    console.print("Performing a single, cheap optimization run to trigger JAX compilation.")
    console.print("This may take several minutes, please be patient...")
    warmup_params = base_parameters.copy()
    warmup_params["kappa"] = 1.0
    warmup_key, master_key = jax.random.split(master_key)
    warmup_solver = DMFT_SaddlePoint_Solver(warmup_params, warmup_key)
    warmup_solver.calculate_saddle_point(m_S=0.5, chi_SS=1.0, num_steps=2).block_until_ready()
    console.print("[green]JIT compilation finished.[/green]")
    # --- END WARM-UP ---

    for kappa in kappa_values:
        master_key, solver_key = jax.random.split(master_key)
        console.rule(f"[bold magenta]Running for κ = {kappa}[/bold magenta]")
        
        current_params = base_parameters.copy()
        current_params["kappa"] = kappa
        
        if kappa <= 0.01:
            current_params["optimization_steps"] = 40000
        else:
            current_params["optimization_steps"] = 15000

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
        def serializer(obj):
            if isinstance(obj, (jnp.ndarray, np.ndarray)): return obj.item()
            return float(obj)
        serializable_results = {k: {**v, 'all_results': [serializer(i) for i in v['all_results']]} for k, v in results_data.items()}
        json.dump(serializable_results, f, indent=4)
    console.print(f"Final results data saved to: [green]'{json_filepath}'[/green]")

    kappas_sorted = sorted([float(k) for k in results_data.keys() if results_data[k]['mean'] is not None])
    means = [results_data[str(k)]['mean'] for k in kappas_sorted]
    
    if any(m is not None and not np.isnan(m) for m in means):
        plt.figure(figsize=(12, 7))
        plt.plot(kappas_sorted, means, 'o-', label=r'Converged $m_S$')
        plt.axhline(y=1.0, color='r', linestyle='--', label=r'$m_S=1$ (Perfect Learning)')
        plt.xscale('log')
        plt.xlabel(r"Noise Parameter $\kappa$")
        plt.ylabel(r"Order Parameter $m_S$")
        plt.title(f"DMFT Order Parameter vs. Noise (Corrected TAP Model)")
        plt.grid(True, which="both", ls="--")
        plt.gca().invert_xaxis()
        plt.legend()
        plt.tight_layout()
        summary_plot_path = os.path.join(save_dir, "summary_m_s_vs_kappa.png")
        plt.savefig(summary_plot_path)
        console.print(f"Saved final summary plot to: [green]'{summary_plot_path}'[/green]")
        plt.show()