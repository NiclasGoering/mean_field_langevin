# (Imports and check_gpu_capabilities are the same)
import math
import numpy as np
import torch
from scipy.integrate import solve_ivp
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json

def check_gpu_capabilities():
    if not torch.cuda.is_available(): print("WARNING: CUDA is not available. Running on CPU."); return torch.device("cpu"), False
    device = torch.device("cuda"); props = torch.cuda.get_device_properties(device)
    print(f"--- GPU Information ---\nUsing device: {props.name}\nCompute Capability: {props.major}.{props.minor}\nTotal Memory: {props.total_memory / 1e9:.2f} GB")
    has_bf16_support = torch.cuda.is_bf16_supported()
    print(f"BFloat16 Support: {has_bf16_support}\n-----------------------")
    if not has_bf16_support: print("WARNING: GPU does not support BFloat16. Falling back to float32.")
    return device, has_bf16_support

class DMFT_Solver:
    def __init__(self, params: dict, device: torch.device, use_bf16: bool, initial_w: torch.Tensor = None):
        self.params = params
        self.device = device
        self.use_bf16 = use_bf16
        self.dtype = torch.bfloat16 if self.use_bf16 else torch.float32
        self.num_chains = params["mcmc_chains"]
        print(f"Using data type: {self.dtype}, running {self.num_chains} parallel MCMC chains.")

        self.S_indices = torch.randperm(params["d"], device=self.device)[: params["k"]]
        
        # Initial w_current is now a BATCH of chains: (num_chains, d)
        if initial_w is not None:
            self.w_current = initial_w.clone().to(device, dtype=self.dtype)
        else:
            self.w_current = torch.randn(
                (self.num_chains, params["d"]), device=self.device, dtype=self.dtype
            ) * math.sqrt(params["sigma_w"] / params["d"])
        
        self.mcmc_step_size = torch.tensor(params.get("mcmc_step_size", 1e-6), device=self.device)
        self.mcmc_acceptance_rate: float = 0.0
        self.last_tau_for_pbar = 0.0

    @staticmethod
    def _phi(z: torch.Tensor) -> torch.Tensor: return torch.relu(z)
    def _sample_inputs(self, n_samples: int) -> torch.Tensor: return (torch.randint(0, 2, (n_samples, self.params["d"]), device=self.device, dtype=self.dtype) * 2 - 1)
    
    # --- Vectorized Methods ---
    def _Sigma_batch(self, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # w: (num_chains, d), x: (n_samples, d) -> Einsum -> (num_chains, n_samples)
        pre_activation = torch.einsum('cd,sd->cs', w, x)
        phi_values = self._phi(pre_activation)
        # Average over n_samples dimension
        return torch.mean(phi_values**2, dim=1) # Returns shape (num_chains,)

    def _J_S_batch(self, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # w: (num_chains, d), x: (n_samples, d) -> Einsum -> (num_chains, n_samples)
        pre_activation = torch.einsum('cd,sd->cs', w, x)
        phi_values = self._phi(pre_activation)
        # chi_S has shape (n_samples,)
        chi_S = torch.prod(x[:, self.S_indices], dim=1)
        # Multiply and average over n_samples dimension
        return torch.mean(phi_values * chi_S, dim=1) # Returns shape (num_chains,)

    def _log_prob_and_grad_batch(self, w: torch.Tensor, m_S: float, x: torch.Tensor):
        w.requires_grad_(True)
        p = self.params; epsilon_stabilizer = 1e-9
        
        Sigma_w = self._Sigma_batch(w, x) # (num_chains,)
        J_S_w = self._J_S_batch(w, x)   # (num_chains,)
        
        denom = (p["N"] ** p["gamma"] / p["sigma_a"]) + (Sigma_w / p["kappa"] ** 2) + epsilon_stabilizer
        log_term = 0.5 * torch.log(denom)
        J_eff = (1.0 - m_S) * J_S_w
        exp_term = (J_eff**2 / p["kappa"] ** 4) / (4.0 * denom)
        
        # Prior needs to be summed over d, then result has shape (num_chains,)
        prior = (p["d"] / (2.0 * p["sigma_w"])) * torch.sum(w**2, dim=1)
        
        # Summing over chains to get a single scalar loss for autograd
        total_log_p = torch.sum(-(log_term - exp_term + prior))

        grad, = torch.autograd.grad(total_log_p, w)
        
        # We also need the per-chain log_p for the MH step
        log_p_per_chain = -(log_term - exp_term + prior)

        if w.requires_grad: w.requires_grad_(False)
        return log_p_per_chain.detach(), grad.detach()

    def _get_final_expectation(self, m_S: float):
        p = self.params; n_accept = 0
        x_mcmc = self._sample_inputs(p["n_samples_J_and_Sigma"])
        term_accumulator = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        burn_in_steps = p["mcmc_samples"] // 4
        target_acceptance = 0.574
        epsilon_stabilizer = 1e-9

        w = self.w_current.clone() # Shape (num_chains, d)
        
        with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_bf16):
            log_prob_current, grad_current = self._log_prob_and_grad_batch(w, m_S, x_mcmc)

            for i in range(p["mcmc_samples"]):
                epsilon = self.mcmc_step_size
                
                noise = torch.randn_like(w) * torch.sqrt(2 * epsilon)
                w_prop = w + epsilon * grad_current + noise

                log_prob_prop, grad_prop = self._log_prob_and_grad_batch(w_prop, m_S, x_mcmc)

                # All calculations are now batched over the chains
                log_q_forward = -torch.sum((w_prop - w - epsilon * grad_current)**2, dim=1) / (4 * epsilon)
                log_q_backward = -torch.sum((w - w_prop - epsilon * grad_prop)**2, dim=1) / (4 * epsilon)
                log_alpha = (log_prob_prop + log_q_backward) - (log_prob_current + log_q_forward)
                
                # Acceptance mask for all chains
                accepted = (torch.rand(self.num_chains, device=self.device).log() < log_alpha)
                # Update all chains that accepted the proposal
                w[accepted] = w_prop[accepted]
                log_prob_current[accepted] = log_prob_prop[accepted]
                grad_current[accepted] = grad_prop[accepted]

                if i >= burn_in_steps:
                    n_accept += torch.sum(accepted).item()
                
                if i < burn_in_steps:
                    acceptance_prob = torch.mean(torch.exp(torch.min(torch.zeros_like(log_alpha), log_alpha)))
                    self.mcmc_step_size *= torch.exp(0.01 * (acceptance_prob - target_acceptance))
                
                if i >= burn_in_steps:
                    # 'w' now contains the current samples for all chains
                    J_S_w_batch = self._J_S_batch(w, x_mcmc)
                    Sigma_w_batch = self._Sigma_batch(w, x_mcmc)
                    denom = (p["N"] ** p["gamma"] / p["sigma_a"]) + (Sigma_w_batch / p["kappa"] ** 2) + epsilon_stabilizer
                    # Accumulate the mean of the terms from all chains
                    term_accumulator += torch.mean((J_S_w_batch**2) / denom)

        self.w_current = w
        total_samples = self.num_chains * (p["mcmc_samples"] - burn_in_steps)
        if total_samples > 0:
            self.mcmc_acceptance_rate = n_accept / total_samples
        
        post_burn_in_samples = p["mcmc_samples"] - burn_in_steps
        return term_accumulator / post_burn_in_samples if post_burn_in_samples > 0 else torch.tensor(0.0)

    # (dmS_dtau and solve are unchanged)
    def dmS_dtau(self, tau: float, m_S_vec):
        m_S = float(m_S_vec[0]); p = self.params
        expectation = self._get_final_expectation(m_S)
        F_mS = p["N"] * ((1.0 - m_S) / p["kappa"] ** 2) * expectation.item()
        progress_update = tau - self.last_tau_for_pbar; self.pbar.update(progress_update); self.last_tau_for_pbar = tau
        self.pbar.set_description(f"τ={tau:.3f}, m_S={m_S:.6f}, F(m_S)={F_mS:.6f}, Accept={self.mcmc_acceptance_rate:.2%}, Step={self.mcmc_step_size.item():.2e}")
        return [-m_S + F_mS]
    def solve(self, m0: float, t_span, t_eval, rtol, atol):
        self.last_tau_for_pbar = t_span[0]
        self.pbar = tqdm(total=t_span[1], initial=t_span[0], unit="τ", desc="Solving ODE")
        sol = solve_ivp(fun=self.dmS_dtau, t_span=t_span, y0=[m0], t_eval=t_eval, method="RK45", rtol=rtol, atol=atol)
        self.pbar.n = t_span[1]; self.pbar.refresh(); self.pbar.close()
        return sol


if __name__ == "__main__":
    device, use_bf16 = check_gpu_capabilities()
    save_dir = "/home/goring/mean_field_langevin/MCMC_Pinf/results/3107_d40k4_mala01"
    os.makedirs(save_dir, exist_ok=True)
    
    # --- TUNING PARAMETERS ---
    # We can use fewer steps per chain, but run many chains in parallel
    mcmc_chains_config = 1024 # Number of parallel walkers
    mcmc_samples_per_chain_config = 4000 # Steps per walker
    n_samples_config = 2000
    mcmc_step_size_config = 1e-4
    ode_rtol = 5e-3; ode_atol = 5e-4

    kappa_values = sorted([0.00001,0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0])
    
    base_parameters = { "d": 40, "N": 1024, "k": 4, "sigma_a": 1.0, "sigma_w": 1.0, "gamma": 1.0,
        "mcmc_chains": mcmc_chains_config,
        "n_samples_J_and_Sigma": n_samples_config, 
        "mcmc_samples": mcmc_samples_per_chain_config,
        "mcmc_step_size": mcmc_step_size_config, 
    }
    
    results_data = {}
    w_final_state = None 

    for kappa in kappa_values:
        print(f"\n{'='*25} Running for kappa = {kappa} {'='*25}")
        current_params = base_parameters.copy()
        current_params["kappa"] = kappa
        
        solver = DMFT_Solver(current_params, device=device, use_bf16=use_bf16, initial_w=w_final_state)
        
        m0_test = 0.7
        t_span = [0, 5]
        t_eval = np.linspace(t_span[0], t_span[1], 50)
        
        sol = solver.solve(m0_test, t_span, t_eval, rtol=ode_rtol, atol=ode_atol)
        print(f"Done with kappa = {kappa}.")
        
        # Save the final w state for the next iteration. We just use the first chain's state as a representative.
        # --- FIX ---
# Correctly save the state of ALL chains for the warm start.
        w_final_state = solver.w_current.clone()
        
        # (The rest of the plotting/saving loop is identical)
        final_m_S = sol.y[0, -1]
        results_data[str(kappa)] = final_m_S
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sol.t, sol.y[0], "o-", label=f"$m_S(\\tau)$ starting from {m0_test}")
        ax.axhline(final_m_S, color='red', ls="--", label=f"Fixed point ≈ {final_m_S:.4f}")
        ax.set_xlabel(r"Fictitious time $\tau$"); ax.set_ylabel(r"Order parameter $m_S$")
        ax.set_title(f"Vectorized MALA DMFT (κ={kappa})")
        ax.legend(); plt.tight_layout()
        plot_filename = f"m_s_evolution_kappa_{kappa}.png"; plot_filepath = os.path.join(save_dir, plot_filename)
        plt.savefig(plot_filepath); plt.close(fig)
        print(f"Saved evolution plot to: {plot_filepath}")
    
    # (Final saving/plotting is identical)
    json_filename = "final_m_s_vs_kappa.json"; json_filepath = os.path.join(save_dir, json_filename)
    with open(json_filepath, 'w') as f: json.dump(results_data, f, indent=4)
    print(f"\nAll simulations finished. Final results data saved to: {json_filepath}")
    plt.figure(figsize=(8, 5)); kappas_sorted = sorted([float(k) for k in results_data.keys()]); m_s_sorted = [results_data[str(k)] for k in results_data.keys()]
    plt.plot(kappas_sorted, m_s_sorted, 'o-'); plt.xscale('log'); plt.xlabel("κ (kappa)"); plt.ylabel("Final Order Parameter $m_S$"); plt.title("Final $m_S$ vs. κ (Vectorized MALA)"); plt.grid(True, which="both", ls="--"); plt.tight_layout()
    summary_plot_path = os.path.join(save_dir, "summary_m_s_vs_kappa.png"); plt.savefig(summary_plot_path)
    print(f"Saved summary plot to: {summary_plot_path}"); plt.show()