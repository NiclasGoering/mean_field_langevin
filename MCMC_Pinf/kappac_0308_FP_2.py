import math
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json
from rich.console import Console
from rich.table import Table
import gc

console = Console()

def check_gpu_capabilities():
    if not torch.cuda.is_available():
        console.print("[yellow]WARNING: CUDA is not available. Running on CPU.[/yellow]")
        return torch.device("cpu")
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    console.print(f"--- GPU Information ---\nUsing device: [cyan]{props.name}[/cyan]\nCompute Capability: [cyan]{props.major}.{props.minor}[/cyan]\nTotal Memory: [cyan]{props.total_memory / 1e9:.2f} GB[/cyan]\n-----------------------")
    return device

class DMFT_Solver:
    def __init__(self, params: dict, device: torch.device):
        self.params = params
        self.device = device
        self.dtype = torch.float32
        self.num_chains = params["mcmc_chains"]
        self.S_indices = torch.randperm(params["d"], device=self.device)[:params["k"]]
        self.w_current = torch.randn(
            (self.num_chains, params["d"]), device=self.device, dtype=self.dtype
        ) * math.sqrt(params["sigma_w"] / params["d"])
        self.mcmc_step_size = torch.tensor(params.get("mcmc_step_size", 1e-6), device=self.device)
        self.mcmc_acceptance_rate = 0.0

    @staticmethod
    def _phi(z: torch.Tensor) -> torch.Tensor:
        return torch.relu(z)

    def _sample_inputs(self, n_samples: int) -> torch.Tensor:
        return (torch.randint(0, 2, (n_samples, self.params["d"]), device=self.device, dtype=self.dtype) * 2 - 1)

    def _calculate_Sigma_expectation(self, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        total_sum_sq = torch.zeros(w.shape[0], device=self.device, dtype=w.dtype)
        num_total_samples = x.shape[0]
        for x_batch in torch.split(x, self.params["data_batch_size"]):
            pre_activation = torch.einsum('cd,sd->cs', w, x_batch)
            phi_values = self._phi(pre_activation)
            total_sum_sq += torch.sum(phi_values**2, dim=1)
        return total_sum_sq / num_total_samples

    def _calculate_J_S_expectation(self, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        total_sum_js = torch.zeros(w.shape[0], device=self.device, dtype=w.dtype)
        num_total_samples = x.shape[0]
        for x_batch in torch.split(x, self.params["data_batch_size"]):
            pre_activation = torch.einsum('cd,sd->cs', w, x_batch)
            phi_values = self._phi(pre_activation)
            chi_S = torch.prod(x_batch[:, self.S_indices], dim=1)
            total_sum_js += torch.sum(phi_values * chi_S.unsqueeze(0), dim=1)
        return total_sum_js / num_total_samples

    def _log_prob_and_grad_batch(self, w: torch.Tensor, m_S: float, x: torch.Tensor):
        p = self.params
        use_float64 = p['kappa'] < 0.01
        if use_float64:
            w_calc = w.double()
            x_calc = x.double()
        else:
            w_calc = w
            x_calc = x
        w_calc.requires_grad_(True)
        
        Sigma_w = self._calculate_Sigma_expectation(w_calc, x_calc)
        J_S_w = self._calculate_J_S_expectation(w_calc, x_calc)
        
        denom = (p["N"] ** p["gamma"] / p["sigma_a"]) + (Sigma_w / p["kappa"] ** 2) + 1e-12
        log_term = 0.5 * torch.log(denom)
        J_eff = (1.0 - m_S) * J_S_w
        exp_term = (J_eff**2 / p["kappa"]**4) / (4.0 * denom)
        prior_term = (p["d"] / (2.0 * p["sigma_w"])) * torch.sum(w_calc**2, dim=1)
        
        log_p_per_chain = -(log_term - exp_term + prior_term)
        total_log_p = torch.sum(log_p_per_chain)
        (grad,) = torch.autograd.grad(total_log_p, w_calc, allow_unused=True)
        return log_p_per_chain.to(self.dtype), grad.to(self.dtype) if grad is not None else None

    def _run_mcmc_and_get_expectation(self, m_S: float, num_mcmc_samples: int, x_mcmc: torch.Tensor):
        p = self.params
        burn_in_steps = num_mcmc_samples // 4
        log_prob_current, grad_current = self._log_prob_and_grad_batch(self.w_current, m_S, x_mcmc)
        if grad_current is None:
            console.print("[bold red]FATAL: Initial gradient is None. MCMC cannot proceed.[/bold red]")
            return torch.tensor(torch.nan, device=self.device)
        log_prob_current = log_prob_current.detach()
        grad_current = grad_current.detach()
        
        n_accept = 0
        expectation_accumulator = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        post_burn_in_count = 0

        pbar = tqdm(range(num_mcmc_samples), desc=f"MCMC Sampling for m_S={m_S:.4f}", leave=False)
        for i in pbar:
            with torch.no_grad():
                epsilon = self.mcmc_step_size
                noise = torch.randn_like(self.w_current) * torch.sqrt(2 * epsilon)
                w_prop = self.w_current + epsilon * grad_current
            
            log_prob_prop, grad_prop = self._log_prob_and_grad_batch(w_prop, m_S, x_mcmc)
            if grad_prop is None:
                log_prob_prop = -torch.inf * torch.ones_like(log_prob_current)

            with torch.no_grad():
                log_q_forward = -torch.sum((w_prop - self.w_current - epsilon * grad_current)**2, dim=1) / (4 * epsilon)
                log_q_backward = -torch.sum((self.w_current - w_prop - epsilon * grad_prop)**2, dim=1) / (4 * epsilon)
                log_alpha = (log_prob_prop + log_q_backward) - (log_prob_current + log_q_forward)
                accepted = (torch.rand(self.num_chains, device=self.device).log() < log_alpha)
                accepted_reshaped = accepted.view(-1, 1)

            self.w_current = torch.where(accepted_reshaped, w_prop, self.w_current).detach()
            log_prob_current = torch.where(accepted, log_prob_prop, log_prob_current).detach()
            grad_current = torch.where(accepted_reshaped, grad_prop, grad_current).detach()

            if i >= burn_in_steps:
                n_accept += torch.sum(accepted).item()
                post_burn_in_count += 1
                with torch.no_grad():
                    current_sigma = self._calculate_Sigma_expectation(self.w_current, x_mcmc)
                    current_js = self._calculate_J_S_expectation(self.w_current, x_mcmc)
                    current_denom = (p["N"] ** p["gamma"] / p["sigma_a"]) + (current_sigma / p["kappa"] ** 2) + 1e-9
                    current_term_estimate = torch.mean(current_js**2 / current_denom)
                    expectation_accumulator += (current_term_estimate - expectation_accumulator) / post_burn_in_count
        
        total_post_burn_in = self.num_chains * (num_mcmc_samples - burn_in_steps)
        self.mcmc_acceptance_rate = n_accept / total_post_burn_in if total_post_burn_in > 0 else 0
        return expectation_accumulator

    def calculate_F(self, m_S: float, num_mcmc_samples: int):
        """
        Performs one high-precision MCMC run to calculate F(m_S).
        This is the core of the fixed-point iteration step.
        """
        p = self.params
        x_mcmc_fixed = self._sample_inputs(p["n_samples_J_and_Sigma"])
        
        # We don't need to re-initialize w_current here, because we want the MCMC
        # state to carry over between iterations, providing a warm start.
        
        expectation = self._run_mcmc_and_get_expectation(m_S, num_mcmc_samples, x_mcmc_fixed)
        
        if torch.isnan(expectation):
            return np.nan
        
        F_val = p["N"] * ((1.0 - m_S) / p["kappa"] ** 2) * expectation.item()
        return F_val

if __name__ == "__main__":
    device = check_gpu_capabilities()
    save_dir = f"/home/goring/mean_field_langevin/MCMC_Pinf/results/3107_hmcd10k2"
    os.makedirs(save_dir, exist_ok=True)
    console.print(f"Results will be saved in: '{os.path.abspath(save_dir)}'")
    
    # --- Configuration for Fixed-Point Iteration ---
    base_parameters = {
        "d": 40, "N": 1024, "k": 4, "sigma_a": 1.0, "sigma_w": 1.0, "gamma": 1.0,
        "mcmc_chains": 8192, # Can increase to leverage H100
        "n_samples_J_and_Sigma": 8000,
        "mcmc_step_size": 1e-7,
        "data_batch_size": 1000,
    }

    # "Leveraging H100" means we can use a large number of samples per step
    # to get a very precise estimate of F(m_S) in each iteration.
    MCMC_SAMPLES_PER_ITERATION = 50000 

    # --- Iteration Control ---
    ITERATION_TOLERANCE = 1e-6
    MAX_ITERATIONS = 50
    DIVERGENCE_THRESHOLD = 10.0 # Stop if the change grows too large

    # --- Simulation Setup ---
    # kappa = 1e-05
    # initial_m_S_guess = 0.99 # Start where you think the solution might be

    kappa = 10.0
    initial_m_S_guess = 0.1 # A different starting point

    console.rule(f"[bold]Starting Fixed-Point Iteration for κ = {kappa}", style="magenta")
    console.print(f"Initial Guess for m_S: {initial_m_S_guess}")
    console.print(f"MCMC Samples per Iteration: {MCMC_SAMPLES_PER_ITERATION}")

    current_params = base_parameters.copy()
    current_params["kappa"] = kappa
    solver = DMFT_Solver(current_params, device=device)
    
    m_S_current = initial_m_S_guess
    m_S_history = [m_S_current]
    last_change = np.inf

    # --- The Main Iteration Loop ---
    for i in range(MAX_ITERATIONS):
        console.rule(f"Iteration {i+1}/{MAX_ITERATIONS}")
        
        # Calculate the next m_S value based on the current one
        m_S_next = solver.calculate_F(m_S_current, MCMC_SAMPLES_PER_ITERATION)
        
        if np.isnan(m_S_next):
            console.print("[bold red]Solver returned NaN. Halting iteration.[/bold red]")
            break

        change = m_S_next - m_S_current
        console.print(f"m_S(old) = {m_S_current:.8f}  -->  m_S(new) = {m_S_next:.8f}  [Change = {change:+.8f}]")
        console.print(f"MCMC acceptance rate for this step: {solver.mcmc_acceptance_rate:.2%}")
        
        m_S_history.append(m_S_next)
        
        # --- Check for Convergence or Divergence ---
        if abs(change) < ITERATION_TOLERANCE:
            console.print(f"\n[bold green]Convergence reached in {i+1} iterations![/bold green]")
            m_S_current = m_S_next
            break
            
        # Safety check for divergence
        if abs(change) > abs(last_change) and abs(change) > DIVERGENCE_THRESHOLD * ITERATION_TOLERANCE:
             console.print(f"\n[bold yellow]WARNING: Iteration appears to be diverging. Halting.[/bold yellow]")
             console.print("This may happen if the fixed point is unstable. Try a different initial guess.")
             break
        
        m_S_current = m_S_next
        last_change = change
        gc.collect()
        torch.cuda.empty_cache()

    else: # This 'else' belongs to the 'for' loop, it runs if the loop finishes without 'break'
        console.print(f"\n[bold yellow]Maximum number of iterations ({MAX_ITERATIONS}) reached. Process did not converge.[/bold yellow]")

    console.rule("[bold]Final Results", style="green")
    console.print(f"Final converged value m_S* = [cyan]{m_S_current:.8f}[/cyan]")
    console.print(f"The solver object now contains MCMC chains (`solver.w_current`) sampling from P(w | m_S*).")
    
    # --- Plotting the convergence history ---
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(m_S_history, 'o-', label='m_S value per iteration')
    ax.axhline(m_S_current, color='red', ls='--', label=f'Final value ≈ {m_S_current:.6f}')
    ax.set_xlabel("Iteration Step")
    ax.set_ylabel(r"Order Parameter $m_S$")
    ax.set_title(f"Fixed-Point Iteration History for κ = {kappa}")
    ax.legend()
    plt.tight_layout()
    plot_filepath = os.path.join(save_dir, f"iteration_history_kappa_{kappa}.png")
    plt.savefig(plot_filepath)
    console.print(f"Saved iteration history plot to: '{plot_filepath}'")
    plt.show()