import math
import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.optimize import brentq # <<< NEW: Import root-finding solver
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json

PRINT_DIAGNOSTICS = True 

def check_gpu_capabilities():
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. Running on CPU.")
        return torch.device("cpu"), False
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(f"--- GPU Information ---\nUsing device: {props.name}\nCompute Capability: {props.major}.{props.minor}\nTotal Memory: {props.total_memory / 1e9:.2f} GB")
    has_bf16_support = torch.cuda.is_bf16_supported()
    print(f"BFloat16 Support: {has_bf16_support}\n-----------------------")
    return device, has_bf16_support

class DMFT_Solver:
    def __init__(self, params: dict, device: torch.device, use_bf16: bool, initial_w: torch.Tensor = None):
        self.params = params
        self.device = device
        self.dtype = torch.float32
        print(f"INFO: Base data type is {self.dtype}. Conditional high-precision will be used.")
        self.num_chains = params["mcmc_chains"]
        print(f"Using {self.num_chains} parallel MCMC chains.")
        self.S_indices = torch.randperm(params["d"], device=self.device)[: params["k"]]
        if initial_w is not None:
            self.w_current = initial_w.clone().to(device, dtype=self.dtype)
        else:
            print("No warm start provided, initializing w_current randomly (cold start).")
            self.w_current = torch.randn((self.num_chains, params["d"]), device=self.device, dtype=self.dtype) * math.sqrt(params["sigma_w"] / params["d"])
        self.mcmc_step_size = torch.tensor(params.get("mcmc_step_size", 1e-6), device=self.device)
        self.mcmc_acceptance_rate: float = 0.0
        self.last_tau_for_pbar = 0.0

    @staticmethod
    def _phi(z: torch.Tensor) -> torch.Tensor: return torch.relu(z)

    def _sample_inputs(self, n_samples: int) -> torch.Tensor:
        return (torch.randint(0, 2, (n_samples, self.params["d"]), device=self.device, dtype=self.dtype) * 2 - 1)

    @staticmethod
    def _Sigma_batch(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        pre_activation = torch.einsum('cd,sd->cs', w, x)
        phi_values = DMFT_Solver._phi(pre_activation)
        return torch.mean(phi_values**2, dim=1)

    @staticmethod
    def _J_S_batch(w: torch.Tensor, x: torch.Tensor, S_indices) -> torch.Tensor:
        pre_activation = torch.einsum('cd,sd->cs', w, x)
        phi_values = DMFT_Solver._phi(pre_activation)
        chi_S = torch.prod(x[:, S_indices], dim=1)
        return torch.mean(phi_values * chi_S, dim=1)

    @staticmethod
    def _log_prob_and_grad_batch(w: torch.Tensor, m_S: float, x: torch.Tensor, S_indices, params):
        p = params
        precision_threshold = 0.01 
        use_float64 = p['kappa'] < precision_threshold

        if use_float64:
            with torch.amp.autocast('cuda', enabled=False):
                w_calc = w.double(); x_calc = x.double(); m_S_calc = float(m_S)
                epsilon_stabilizer = 1e-20
        else:
            w_calc = w; x_calc = x; m_S_calc = m_S
            epsilon_stabilizer = 1e-9

        w_calc.requires_grad_(True)
        Sigma_w = DMFT_Solver._Sigma_batch(w_calc, x_calc)
        J_S_w = DMFT_Solver._J_S_batch(w_calc, x_calc, S_indices)
        denom = (p["N"] ** p["gamma"] / p["sigma_a"]) + (Sigma_w / p["kappa"] ** 2) + epsilon_stabilizer
        log_term = 0.5 * torch.log(denom)
        J_eff = (1.0 - m_S_calc) * J_S_w
        exp_term_raw = (J_eff**2 / p["kappa"] ** 4) / (4.0 * denom)
        exp_term = torch.clamp(exp_term_raw, max=150.0)
        prior = (p["d"] / (2.0 * p["sigma_w"])) * torch.sum(w_calc**2, dim=1)
        
        global PRINT_DIAGNOSTICS
        if PRINT_DIAGNOSTICS:
            print(f"\n[DIAGNOSTIC k={p['kappa']}] Using float64: {use_float64}. Max exp_term: {torch.max(exp_term_raw).item():.2e}")
            PRINT_DIAGNOSTICS = False

        log_p_per_chain = -(log_term - exp_term + prior)
        total_log_p = torch.sum(log_p_per_chain)
        (grad,) = torch.autograd.grad(total_log_p, w_calc, allow_unused=True)
        
        final_log_p = log_p_per_chain.to(w.dtype)
        final_grad = grad.to(w.dtype) if grad is not None else None
        
        return final_log_p, final_grad

    def _get_final_expectation(self, m_S: float, x_mcmc: torch.Tensor):
        p = self.params; n_accept = 0
        burn_in_steps = p["mcmc_samples"] // 4
        target_acceptance = 0.65; adapt_rate = 0.05
        running_avg_accumulator = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        post_burn_in_count = 0
        running_avg_window = p.get("running_avg_window", 10)
        
        w = self.w_current.clone().detach()
        global PRINT_DIAGNOSTICS
        PRINT_DIAGNOSTICS = True
        log_prob_current, grad_current = self._log_prob_and_grad_batch(w, m_S, x_mcmc, self.S_indices, self.params)

        if grad_current is None:
            print("WARNING: Initial gradient is None. MCMC cannot proceed.")
            return torch.tensor(0.0)
        
        log_prob_current = log_prob_current.detach()
        grad_current = grad_current.detach()

        for i in range(p["mcmc_samples"]):
            epsilon = self.mcmc_step_size
            noise = torch.randn_like(w) * torch.sqrt(2 * epsilon)
            w_prop = w + epsilon * grad_current + noise
            log_prob_prop, grad_prop = self._log_prob_and_grad_batch(w_prop, m_S, x_mcmc, self.S_indices, self.params)

            if grad_prop is None: log_alpha = -torch.inf * torch.ones(self.num_chains, device=self.device)
            else:
                log_q_forward = -torch.sum((w_prop - w - epsilon * grad_current)**2, dim=1) / (4 * epsilon)
                log_q_backward = -torch.sum((w - w_prop - epsilon * grad_prop.detach())**2, dim=1) / (4 * epsilon)
                log_alpha = (log_prob_prop.detach() + log_q_backward) - (log_prob_current + log_q_forward)

            log_alpha = torch.nan_to_num(log_alpha, nan=-torch.inf)
            accepted = (torch.rand(self.num_chains, device=self.device).log() < log_alpha)
            accepted_reshaped = accepted.view(-1, 1).expand_as(w)
            
            next_w = torch.where(accepted_reshaped, w_prop, w)
            next_log_prob = torch.where(accepted, log_prob_prop, log_prob_current)
            next_grad = torch.where(accepted_reshaped, grad_prop, grad_current) if grad_prop is not None else grad_current
            
            w = next_w.detach()
            log_prob_current = next_log_prob.detach()
            grad_current = next_grad.detach()

            if i < burn_in_steps:
                acceptance_prob = torch.mean(torch.exp(torch.min(torch.zeros_like(log_alpha), log_alpha)))
                if not torch.isnan(acceptance_prob): self.mcmc_step_size *= torch.exp(adapt_rate * (acceptance_prob - target_acceptance))
            else:
                n_accept += torch.sum(accepted).item()
                post_burn_in_count += 1
                current_sigma = self._Sigma_batch(w, x_mcmc)
                current_js = self._J_S_batch(w, x_mcmc, self.S_indices)
                current_denom = (p["N"] ** p["gamma"] / p["sigma_a"]) + (current_sigma / p["kappa"] ** 2) + 1e-9
                current_term_estimate = torch.mean(current_js**2 / current_denom)
                running_avg_accumulator += (current_term_estimate - running_avg_accumulator) / min(running_avg_window, post_burn_in_count)

        self.w_current = w.detach()
        total_samples_for_rate = self.num_chains * (p["mcmc_samples"] - burn_in_steps)
        if total_samples_for_rate > 0: self.mcmc_acceptance_rate = n_accept / total_samples_for_rate
        
        return running_avg_accumulator if post_burn_in_count > 0 else torch.tensor(0.0)
    
    def dmS_dtau(self, tau: float, m_S_vec, x_mcmc: torch.Tensor):
        m_S = float(m_S_vec[0])
        # Clip m_S to prevent it from going into unphysical regimes which can cause instability
        m_S = np.clip(m_S, -1.0, 1.0) 
        
        expectation = self._get_final_expectation(m_S, x_mcmc)
        F_mS = self.params["N"] * ((1.0 - m_S) / self.params["kappa"] ** 2) * expectation.item()
        
        self.pbar.update(tau - self.last_tau_for_pbar)
        self.last_tau_for_pbar = tau
        step_str = f"{self.mcmc_step_size.item():.2e}" if not torch.isnan(self.mcmc_step_size) else "nan"
        self.pbar.set_description(f"τ={tau:.3f}, m_S={m_S:.6f}, F(m_S)={F_mS:.6f}, Accept={self.mcmc_acceptance_rate:.2%}, Step={step_str}")
        
        if np.isnan(F_mS):
            print("\nFATAL ERROR: F(m_S) is NaN. Halting ODE for this kappa.")
            return np.nan
        
        # --- IMPLEMENT DAMPING ---
        alpha = self.params.get("damping_alpha", 1.0) # Default to 1.0 (no damping)
        return [alpha * (-m_S + F_mS)]

    def solve_ode(self, m0: float, t_span, t_eval, rtol, atol):
        """Solves the DMFT equation using a traditional ODE integrator."""
        print("Using Damped ODE Solver...")
        self.last_tau_for_pbar = t_span[0]
        x_mcmc_fixed = self._sample_inputs(self.params["n_samples_J_and_Sigma"])
        fun_with_fixed_x = lambda t, y: self.dmS_dtau(t, y, x_mcmc_fixed)
        self.pbar = tqdm(total=t_span[1], initial=t_span[0], unit="τ", desc="Solving ODE")
        sol = solve_ivp(fun=fun_with_fixed_x, t_span=t_span, y0=[m0], t_eval=t_eval, method="RK45", rtol=rtol, atol=atol)
        self.pbar.n = t_span[1]; self.pbar.refresh(); self.pbar.close()
        return sol

    def solve_root(self, bracket=[0.01, 0.99], xtol=1e-4):
        """Solves the DMFT equation by finding the root of H(m) = m - F(m)."""
        print("Using Root-Finding Solver...")
        x_mcmc_fixed = self._sample_inputs(self.params["n_samples_J_and_Sigma"])

        def H(m_S: float) -> float:
            # The function whose root we want to find
            print(f"  Testing m_S = {m_S:.5f}...")
            expectation = self._get_final_expectation(m_S, x_mcmc_fixed)
            F_mS = self.params["N"] * ((1.0 - m_S) / self.params["kappa"] ** 2) * expectation.item()
            residual = m_S - F_mS
            print(f"    -> F(m_S) = {F_mS:.5f}, Residual (m-F(m)) = {residual:.5f}")
            return residual
        
        try:
            root = brentq(H, a=bracket[0], b=bracket[1], xtol=xtol)
            return root
        except ValueError as e:
            print(f"\nERROR: Root finding failed. The signs of H(a) and H(b) might be the same. {e}")
            return np.nan


if __name__ == "__main__":
    # --- CHOOSE YOUR SOLVER ---
    # solver_method = "ode"
    solver_method = "ode"

    device, use_bf16 = check_gpu_capabilities()
    save_dir = f"/home/goring/mean_field_langevin/MCMC_Pinf/results/3107_d40k4_mala09_v2"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved in: '{os.path.abspath(save_dir)}'")
    
    mcmc_chains_config = 4096 
    # For root finding with a noisy function, more samples are better.
    mcmc_samples_per_chain_config = 10000 
    n_samples_config = 8000
    mcmc_step_size_config = 1e-7
    
    kappa_values = sorted([0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0])
    
    base_parameters = {
        "d": 40, "N": 1024, "k": 4, "sigma_a": 1.0, "sigma_w": 1.0, "gamma": 1.0,
        "mcmc_chains": mcmc_chains_config, 
        "n_samples_J_and_Sigma": n_samples_config, 
        "mcmc_samples": mcmc_samples_per_chain_config, 
        "mcmc_step_size": mcmc_step_size_config,
        "running_avg_window": 25,
        "damping_alpha": 0.1 # Damping factor for the ODE solver
    }
    
    results_data = {}
    for kappa in kappa_values:
        print(f"\n{'='*25} Running for kappa = {kappa} {'='*25}")
        current_params = base_parameters.copy()
        current_params["kappa"] = kappa
        
        solver = DMFT_Solver(current_params, device=device, use_bf16=False, initial_w=None)
        
        if solver_method == "ode":
            m0_test = 0.7
            t_span = [0, 15]; t_eval = np.linspace(t_span[0], t_span[1], 100)
            sol = solver.solve_ode(m0_test, t_span, t_eval, rtol=1e-4, atol=1e-5)
            
            if sol.status != 0:
                print(f"ODE solver failed for kappa = {kappa}. Status: {sol.status}, Message: {sol.message}")
                final_m_S = np.nan
            else:
                final_m_S = sol.y[0, -1]
        
        elif solver_method == "root":
            final_m_S = solver.solve_root()
            # For plotting, we create a dummy solution object
            sol = type('solution', (), {})()
            sol.t = [0, 1]
            sol.y = [[final_m_S, final_m_S]]


        print(f"Done with kappa = {kappa}. Final m_S = {final_m_S}")
        results_data[str(kappa)] = final_m_S
        
        if not np.isnan(final_m_S):
            plt.style.use("seaborn-v0_8-whitegrid")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(sol.t, sol.y[0], "o-", label=f"$m_S$ values")
            ax.axhline(final_m_S, color='red', ls="--", label=f"Fixed point ≈ {final_m_S:.4f}")
            ax.set_xlabel(r"Fictitious time $\tau$ (if applicable)"); ax.set_ylabel(r"Order parameter $m_S$")
            ax.set_title(f"Final MALA DMFT (κ={kappa}, Solver: {solver_method})"); ax.legend(); plt.tight_layout()
            plot_filename = f"m_s_evolution_kappa_{kappa}.png"; plot_filepath = os.path.join(save_dir, plot_filename)
            plt.savefig(plot_filepath); plt.close(fig)
            print(f"Saved evolution plot to: {plot_filepath}")

    json_filename = "final_m_s_vs_kappa.json"
    json_filepath = os.path.join(save_dir, json_filename)
    clean_results = {k: v for k, v in results_data.items() if not np.isnan(v)}
    if clean_results:
        sorted_results = {str(k): clean_results.get(str(k)) for k in sorted([float(k) for k in clean_results.keys()])}
        with open(json_filepath, 'w') as f: json.dump(sorted_results, f, indent=4)
        print(f"\nAll simulations finished. Final results data saved to: {json_filepath}")
        
        plt.figure(figsize=(8, 5))
        kappas_sorted = [float(k) for k in sorted_results.keys()]
        m_s_sorted = [sorted_results[str(k)] for k in kappas_sorted]
        plt.plot(kappas_sorted, m_s_sorted, 'o-')
        plt.xscale('log'); plt.xlabel("κ (kappa)"); plt.ylabel("Final Order Parameter $m_S$")
        plt.title(f"Final $m_S$ as a function of κ (Solver: {solver_method})"); plt.grid(True, which="both", ls="--")
        plt.tight_layout(); summary_plot_path = os.path.join(save_dir, "summary_m_s_vs_kappa.png")
        plt.savefig(summary_plot_path); print(f"Saved summary plot to: {summary_plot_path}"); plt.show()