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
    """
    Final "gold-standard" implementation using a robust and theoretically correct
    Metropolis-Adjusted Langevin Algorithm (MALA) with adaptive step size.
    This balances speed and accuracy.
    """
    def __init__(self, params: dict, device: torch.device, use_bf16: bool):
        self.params = params
        self.device = device
        self.use_bf16 = use_bf16
        self.dtype = torch.bfloat16 if self.use_bf16 else torch.float32
        print(f"Using data type: {self.dtype}")

        self.S_indices = torch.randperm(params["d"], device=self.device)[: params["k"]]
        self.w_current = torch.randn(
            params["d"], device=self.device, dtype=self.dtype
        ) * math.sqrt(params["sigma_w"] / params["d"])
        
        self.mcmc_step_size = torch.tensor(params.get("mcmc_step_size", 1e-6), device=self.device)
        self.mcmc_acceptance_rate: float = 0.0

    @staticmethod
    def _phi(z: torch.Tensor) -> torch.Tensor: return torch.relu(z)
    def _sample_inputs(self, n_samples: int) -> torch.Tensor: return (torch.randint(0, 2, (n_samples, self.params["d"]), device=self.device, dtype=self.dtype) * 2 - 1)
    def _Sigma_single_w(self, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor: return torch.mean(self._phi(x @ w) ** 2)
    def _J_S_single_w(self, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor: return torch.mean(self._phi(x @ w) * torch.prod(x[:, self.S_indices], dim=1))

    def _log_prob_and_grad(self, w: torch.Tensor, m_S: float, x: torch.Tensor):
        w.requires_grad_(True)
        p = self.params; epsilon_stabilizer = 1e-9
        
        Sigma_w = self._Sigma_single_w(w, x)
        J_S_w = self._J_S_single_w(w, x)
        denom = (p["N"] ** p["gamma"] / p["sigma_a"]) + (Sigma_w / p["kappa"] ** 2) + epsilon_stabilizer
        log_term = 0.5 * torch.log(denom)
        J_eff = (1.0 - m_S) * J_S_w
        exp_term = (J_eff**2 / p["kappa"] ** 4) / (4.0 * denom)
        prior = (p["d"] / (2.0 * p["sigma_w"])) * torch.sum(w**2)
        log_p = -(log_term - exp_term + prior)

        grad, = torch.autograd.grad(log_p, w)
        if w.requires_grad: w.requires_grad_(False)
        return log_p.detach(), grad.detach()

    def _get_final_expectation(self, m_S: float):
        p = self.params; n_accept = 0
        x_mcmc = self._sample_inputs(p["n_samples_J_and_Sigma"])
        term_accumulator = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        burn_in_steps = p["mcmc_samples"] // 4
        target_acceptance = 0.574 # Theoretical optimum for MALA
        epsilon_stabilizer = 1e-9

        w = self.w_current.clone()
        
        with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_bf16):
            log_prob_current, grad_current = self._log_prob_and_grad(w, m_S, x_mcmc)

            for i in range(p["mcmc_samples"]):
                epsilon = self.mcmc_step_size
                
                # Propose a new state using the Langevin step
                noise = torch.randn_like(w) * torch.sqrt(2 * epsilon)
                w_prop = w + epsilon * grad_current + noise

                # Calculate log-probability and gradient at the proposed state
                log_prob_prop, grad_prop = self._log_prob_and_grad(w_prop, m_S, x_mcmc)

                # Calculate the Metropolis-Hastings acceptance probability
                # Forward proposal probability q(w_prop | w)
                log_q_forward = -torch.sum((w_prop - w - epsilon * grad_current)**2) / (4 * epsilon)
                # Backward proposal probability q(w | w_prop)
                log_q_backward = -torch.sum((w - w_prop - epsilon * grad_prop)**2) / (4 * epsilon)
                
                log_alpha = (log_prob_prop + log_q_backward) - (log_prob_current + log_q_forward)
                
                acceptance_prob = torch.exp(torch.min(torch.tensor(0.0, device=self.device), log_alpha))

                if torch.rand(1, device=self.device).log() < log_alpha:
                    w = w_prop
                    log_prob_current = log_prob_prop
                    grad_current = grad_prop
                    if i >= burn_in_steps:
                        n_accept += 1
                
                # Adapt step size during burn-in to meet target acceptance rate
                if i < burn_in_steps:
                    self.mcmc_step_size *= torch.exp(0.01 * (acceptance_prob - target_acceptance))

                # Accumulate statistics after burn-in
                if i >= burn_in_steps:
                    current_J_S_w = self._J_S_single_w(w, x_mcmc)
                    current_Sigma_w = self._Sigma_single_w(w, x_mcmc)
                    denom = (p["N"] ** p["gamma"] / p["sigma_a"]) + (current_Sigma_w / p["kappa"] ** 2) + epsilon_stabilizer
                    term_accumulator += (current_J_S_w**2) / denom

        self.w_current = w
        
        post_burn_in_samples = p["mcmc_samples"] - burn_in_steps
        if post_burn_in_samples > 0:
            self.mcmc_acceptance_rate = n_accept / post_burn_in_samples
            return term_accumulator / post_burn_in_samples
        else:
            return torch.tensor(0.0, device=self.device)

    def dmS_dtau(self, tau: float, m_S_vec):
        m_S = float(m_S_vec[0]); p = self.params
        expectation = self._get_final_expectation(m_S)
        F_mS = p["N"] * ((1.0 - m_S) / p["kappa"] ** 2) * expectation.item()
        self.pbar.set_description(f"τ={tau:.3f}, m_S={m_S:.6f}, F(m_S)={F_mS:.6f}, Accept={self.mcmc_acceptance_rate:.2%}, Step={self.mcmc_step_size.item():.2e}")
        self.pbar.update(1)
        return [-m_S + F_mS]

    def solve(self, m0: float, t_span, t_eval):
        if t_eval is not None: n_steps = len(t_eval)
        else: n_steps = int((t_span[1] - t_span[0]) * 20)
        self.pbar = tqdm(total=n_steps, desc="Solving ODE")
        sol = solve_ivp(fun=self.dmS_dtau, t_span=t_span, y0=[m0], t_eval=t_eval, method="RK45")
        self.pbar.n = n_steps; self.pbar.refresh(); self.pbar.close()
        return sol

if __name__ == "__main__":
    device, use_bf16 = check_gpu_capabilities()
    save_dir = "/home/goring/mean_field_langevin/MCMC_Pinf/results/3107_d40k4_mala01"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved in: '{os.path.abspath(save_dir)}'")
    kappa_values = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    base_parameters = { "d": 40, "N": 1024, "k": 4, "sigma_a": 1.0, "sigma_w": 1.0, "gamma": 1.0,
        "n_samples_J_and_Sigma": 8000, 
        "mcmc_samples": 10000,
        # Start with a small step size; the adaptive logic will increase it.
        "mcmc_step_size": 1e-5, 
    }
    # ... (The rest of the main loop is identical to the previous version)
    results_data = {}
    for kappa in kappa_values:
        print(f"\n{'='*25} Running for kappa = {kappa} {'='*25}")
        current_params = base_parameters.copy()
        current_params["kappa"] = kappa
        solver = DMFT_Solver(current_params, device=device, use_bf16=use_bf16)
        m0_test = 0.9
        t_span = [0, 10]
        t_eval = np.linspace(t_span[0], t_span[1], 100)
        sol = solver.solve(m0_test, t_span, t_eval)
        print(f"Done with kappa = {kappa}.")
        final_m_S = sol.y[0, -1]
        results_data[str(kappa)] = final_m_S
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sol.t, sol.y[0], "o-", label=f"$m_S(\\tau)$ starting from {m0_test}")
        ax.axhline(final_m_S, color='red', ls="--", label=f"Fixed point ≈ {final_m_S:.4f}")
        ax.set_xlabel(r"Fictitious time $\tau$"); ax.set_ylabel(r"Order parameter $m_S$")
        ax.set_title(f"MALA DMFT (κ={kappa})")
        ax.legend(); plt.tight_layout()
        plot_filename = f"m_s_evolution_kappa_{kappa}.png"; plot_filepath = os.path.join(save_dir, plot_filename)
        plt.savefig(plot_filepath); plt.close(fig)
        print(f"Saved evolution plot to: {plot_filepath}")
    sorted_results = {str(k): results_data[str(k)] for k in sorted([float(k) for k in results_data.keys()])}
    json_filename = "final_m_s_vs_kappa.json"; json_filepath = os.path.join(save_dir, json_filename)
    with open(json_filepath, 'w') as f: json.dump(sorted_results, f, indent=4)
    print(f"\nAll simulations finished. Final results data saved to: {json_filepath}")
    plt.figure(figsize=(8, 5)); kappas_sorted = sorted([float(k) for k in sorted_results.keys()]); m_s_sorted = [sorted_results[str(k)] for k in sorted_results.keys()]
    plt.plot(kappas_sorted, m_s_sorted, 'o-'); plt.xscale('log'); plt.xlabel("κ (kappa)"); plt.ylabel("Final Order Parameter $m_S$"); plt.title("Final $m_S$ vs. κ (MALA)"); plt.grid(True, which="both", ls="--"); plt.tight_layout()
    summary_plot_path = os.path.join(save_dir, "summary_m_s_vs_kappa.png"); plt.savefig(summary_plot_path)
    print(f"Saved summary plot to: {summary_plot_path}"); plt.show()