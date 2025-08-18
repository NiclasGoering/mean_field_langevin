import math
import numpy as np
import torch
from scipy.optimize import brentq
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

    # =========================================================================
    # CORRECTED FUNCTION FOR TYPE HANDLING
    # =========================================================================
    def _log_prob_and_grad_batch(self, w: torch.Tensor, m_S: float, x: torch.Tensor):
        p = self.params
        use_float64 = p['kappa'] < 0.01

        # FIX: Cast both w and x to double if high precision is needed
        if use_float64:
            w_calc = w.double()
            x_calc = x.double()
        else:
            w_calc = w
            x_calc = x
        
        w_calc.requires_grad_(True)
        
        # Now pass the correctly-typed tensors
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
    # =========================================================================

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

        for i in range(num_mcmc_samples):
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

    def get_F_robust(self, m_S: float, num_mcmc_samples: int, num_runs: int):
        p = self.params
        F_values = []
        x_mcmc_fixed = self._sample_inputs(p["n_samples_J_and_Sigma"])
        
        for _ in range(num_runs):
            self.w_current = torch.randn(
                (self.num_chains, p["d"]), device=self.device, dtype=self.dtype
            ) * math.sqrt(p["sigma_w"] / p["d"])
            
            expectation = self._run_mcmc_and_get_expectation(m_S, num_mcmc_samples, x_mcmc_fixed)
            if not torch.isnan(expectation):
                F_val = p["N"] * ((1.0 - m_S) / p["kappa"] ** 2) * expectation.item()
                F_values.append(F_val)
        
        if not F_values:
            return np.nan, np.nan
            
        mean_F = np.mean(F_values)
        std_err_F = np.std(F_values, ddof=1) / np.sqrt(len(F_values)) if len(F_values) > 1 else 0
        
        return mean_F, std_err_F

def run_phase1_exploration(solver: DMFT_Solver, p1_config: dict):
    console.print("\n[bold green]----------- Phase 1: Exploration -----------[/bold green]")
    m_grid = np.linspace(p1_config["m_min"], p1_config["m_max"], p1_config["grid_points"])
    F_means, F_stderrs = [], []
    pbar = tqdm(m_grid, desc="Phase 1 Scan")
    for m_S in pbar:
        mean_F, stderr_F = solver.get_F_robust(
            m_S, num_mcmc_samples=p1_config["mcmc_samples"], num_runs=p1_config["num_runs"]
        )
        F_means.append(mean_F); F_stderrs.append(stderr_F)
        pbar.set_postfix({
            "m_S": f"{m_S:.3f}", "F(m_S)": f"{mean_F:.3f} ± {stderr_F:.3f}",
            "Accept": f"{solver.mcmc_acceptance_rate:.2%}"
        })
    return {"m_grid": m_grid, "F_means": np.array(F_means), "F_stderrs": np.array(F_stderrs)}

def run_phase2_refinement(solver: DMFT_Solver, p2_config: dict, p1_data: dict):
    console.print("\n[bold blue]----------- Phase 2: Refinement -----------[/bold blue]")
    m_grid, F_means = p1_data["m_grid"], p1_data["F_means"]
    H_means = m_grid - F_means
    brackets = []
    for i in range(len(m_grid) - 1):
        if np.sign(H_means[i]) != np.sign(H_means[i+1]):
            brackets.append((m_grid[i], m_grid[i+1]))
    if not brackets:
        console.print("[yellow]No zero-crossings found in Phase 1 data. Cannot proceed.[/yellow]")
        return []
    console.print(f"Found {len(brackets)} potential root bracket(s): {brackets}")
    def H_precise(m_S_scalar):
        console.print(f"  [Phase 2] High-precision evaluation at m_S = {m_S_scalar:.6f}...")
        mean_F, stderr_F = solver.get_F_robust(
            m_S_scalar, num_mcmc_samples=p2_config["mcmc_samples"], num_runs=p2_config["num_runs"]
        )
        residual = m_S_scalar - mean_F
        console.print(f"  [Phase 2] -> F(m_S)={mean_F:.6f}, H(m_S)={residual:.6f} (stderr: {stderr_F:.2e})")
        if np.isnan(residual): return np.inf
        return residual
    found_roots = []
    for a, b in brackets:
        try:
            root = brentq(H_precise, a, b, xtol=p2_config["tolerance"])
            found_roots.append(root)
            console.print(f"[bold green]Successfully found root: {root:.6f} in bracket [{a:.3f}, {b:.3f}][/bold green]")
        except (ValueError, RuntimeError):
            console.print(f"[bold red]Root finding failed in bracket [{a:.3f}, {b:.3f}].[/bold red]")
    return found_roots

def create_and_save_plot(p1_data, roots, kappa, save_dir):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(
        p1_data["m_grid"], p1_data["F_means"], yerr=p1_data["F_stderrs"],
        fmt='o', capsize=4, label=r'Estimate of $F(m_S)$', zorder=10
    )
    ax.plot(p1_data["m_grid"], p1_data["m_grid"], 'k--', label=r'$y = m_S$')
    for i, root in enumerate(roots):
        ax.axvline(root, color='red', ls='--', lw=2, label=f'Found Root ≈ {root:.4f}' if i == 0 else None)
        ax.axhline(root, color='red', ls='--', lw=2)
    ax.set_xlabel(r"Order Parameter $m_S$"); ax.set_ylabel(r"$F(m_S)$")
    ax.set_title(f"DMFT Fixed Point Analysis for κ = {kappa}")
    ax.legend(); plt.tight_layout()
    plot_filepath = os.path.join(save_dir, f"fixed_point_analysis_kappa_{kappa}.png")
    plt.savefig(plot_filepath); plt.close(fig)
    console.print(f"Saved analysis plot to: [green]'{plot_filepath}'[/green]")

if __name__ == "__main__":
    device = check_gpu_capabilities()
    save_dir = f"/home/goring/mean_field_langevin/MCMC_Pinf/results/3107_d40k4_mala09_v2"
    os.makedirs(save_dir, exist_ok=True)
    console.print(f"Results will be saved in: '{os.path.abspath(save_dir)}'")
    
    base_parameters = {
        "d": 40, "N": 1024, "k": 4, "sigma_a": 1.0, "sigma_w": 1.0, "gamma": 1.0,
        "mcmc_chains": 4096, "n_samples_J_and_Sigma": 8000, "mcmc_step_size": 1e-7,
        "data_batch_size": 10000,
    }
    p1_config = {"m_min": 0.01, "m_max": 0.99, "grid_points": 20, "mcmc_samples": 2000, "num_runs": 5}
    p2_config = {"mcmc_samples": 20000, "num_runs": 10, "tolerance": 1e-6}
    
    kappa_values = sorted([0.05, 1e-05]) 

    results_data = {}
    for kappa in kappa_values:
        console.rule(f"[bold]Running for κ = {kappa}", style="magenta")
        current_params = base_parameters.copy()
        current_params["kappa"] = kappa
        solver = DMFT_Solver(current_params, device=device)
        
        p1_data = run_phase1_exploration(solver, p1_config)
        found_roots = run_phase2_refinement(solver, p2_config, p1_data)
        create_and_save_plot(p1_data, found_roots, kappa, save_dir)
        results_data[str(kappa)] = found_roots
        
        gc.collect()
        torch.cuda.empty_cache()
        
        table = Table(title=f"Summary for κ = {kappa}")
        table.add_column("Fixed Point (m_S)", justify="right", style="cyan")
        for root in found_roots: table.add_row(f"{root:.6f}")
        console.print(table)
        
    console.rule("[bold]All simulations finished", style="green")
    json_filepath = os.path.join(save_dir, "final_roots_vs_kappa.json")
    with open(json_filepath, 'w') as f: json.dump(results_data, f, indent=4)
    console.print(f"Final results data saved to: [green]'{json_filepath}'[/green]")