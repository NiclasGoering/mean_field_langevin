import math
import numpy as np
import torch
from scipy.integrate import solve_ivp
from tqdm import tqdm
import matplotlib.pyplot as plt


class DMFT_Solver:
    """
    Final version of the DMFT solver featuring a robust,
    adaptive Metropolis-Adjusted Langevin Algorithm (MALA) for MCMC sampling.
    """

    def __init__(self, params: dict):
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.S_indices = torch.randperm(params["d"], device=self.device)[: params["k"]]
        self.w_current = torch.randn(params["d"], device=self.device) * math.sqrt(
            params["sigma_w"] / params["d"]
        )
        self.mcmc_acceptance_rate: float = 0.0
        # Set an initial guess for the step size; the algorithm will adapt it.
        self.mcmc_step_size = params.get("mcmc_step_size", 0.001)

        if hasattr(torch, "compile"):
            print("PyTorch 2.x detected. Compiling key functions...")
            self._Sigma = torch.compile(self._Sigma)
            self._J_S = torch.compile(self._J_S)
            self._log_prob_unnormalized = torch.compile(self._log_prob_unnormalized)

    # ------------------ Core math (unchanged) ------------------ #
    @staticmethod
    def _phi(z: torch.Tensor) -> torch.Tensor:
        return torch.relu(z)

    def _sample_inputs(self, n_samples: int) -> torch.Tensor:
        return (
            torch.randint(
                0, 2, (n_samples, self.params["d"]), device=self.device, dtype=torch.float32
            ) * 2 - 1
        )

    def _Sigma(self, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(self._phi(x @ w) ** 2)

    def _J_S(self, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(self._phi(x @ w) * torch.prod(x[:, self.S_indices], dim=1))

    def _log_prob_unnormalized(self, w: torch.Tensor, m_S: float, x: torch.Tensor) -> torch.Tensor:
        p = self.params
        Sigma_w = self._Sigma(w, x)
        J_S_w = self._J_S(w, x)
        denom = (p["N"] ** p["gamma"] / p["sigma_a"]) + (Sigma_w / p["kappa"] ** 2)
        log_term = 0.5 * torch.log(denom)
        J_eff = (1.0 - m_S) * J_S_w
        exp_term = (J_eff**2 / p["kappa"] ** 4) / (4.0 * denom)
        prior = (p["d"] / (2.0 * p["sigma_w"])) * torch.sum(w**2)
        return -(log_term - exp_term + prior)

    # ------------------ Gradient Calculation for MALA ------------------ #
    def _grad_log_prob(self, w: torch.Tensor, m_S: float, x: torch.Tensor) -> torch.Tensor:
        w.requires_grad_()
        log_p = self._log_prob_unnormalized(w, m_S, x)
        grad, = torch.autograd.grad(log_p, w)
        w.requires_grad_(False)
        return grad
    
    # ------------------ UPGRADED: ADAPTIVE MALA Sampler ------------------ #
    def _calculate_F(self, m_S: float) -> float:
        p = self.params
        samples: list[torch.Tensor] = []
        n_accept = 0

        x_mcmc = self._sample_inputs(p["n_samples_J_and_Sigma"])
        
        burn_in_steps = p["mcmc_samples"] // 4
        # The optimal acceptance rate for MALA is theoretically ~57.4%
        target_acceptance = 0.574

        for i in range(p["mcmc_samples"]):
            grad_current = self._grad_log_prob(self.w_current, m_S, x_mcmc)
            
            epsilon = self.mcmc_step_size
            noise = torch.randn_like(self.w_current) * math.sqrt(2 * epsilon)
            w_prop = self.w_current + epsilon * grad_current + noise

            grad_prop = self._grad_log_prob(w_prop, m_S, x_mcmc)
            
            log_prob_current = self._log_prob_unnormalized(self.w_current, m_S, x_mcmc)
            log_prob_prop = self._log_prob_unnormalized(w_prop, m_S, x_mcmc)

            log_q_forward = -torch.sum((w_prop - self.w_current - epsilon * grad_current)**2) / (4 * epsilon)
            log_q_backward = -torch.sum((self.w_current - w_prop - epsilon * grad_prop)**2) / (4 * epsilon)
            
            log_alpha = (log_prob_prop + log_q_backward) - (log_prob_current + log_q_forward)
            
            acceptance_prob = math.exp(min(0.0, log_alpha.item()))

            if math.log(np.random.rand()) < log_alpha.item():
                self.w_current = w_prop
                if i >= burn_in_steps: # Only count acceptances after burn-in
                    n_accept += 1
            
            # --- Adaptive step size logic during burn-in phase --- #
            if i < burn_in_steps:
                self.mcmc_step_size *= math.exp(0.01 * (acceptance_prob - target_acceptance))
            
            samples.append(self.w_current)
        
        post_burn_in_samples = p["mcmc_samples"] - burn_in_steps
        if post_burn_in_samples > 0:
            self.mcmc_acceptance_rate = n_accept / post_burn_in_samples
        else: # Handle case where mcmc_samples < 4
            self.mcmc_acceptance_rate = 0.0

        W = torch.stack(samples[burn_in_steps:], dim=0)

        x_big = self._sample_inputs(p["n_samples_J_and_Sigma"] * 5)
        pre = x_big @ W.T
        phi_pre = self._phi(pre)
        Sigma_w = torch.mean(phi_pre**2, dim=0)
        chi_S = torch.prod(x_big[:, self.S_indices], dim=1, keepdim=True)
        J_S_w = torch.mean(phi_pre * chi_S, dim=0)
        denom = (p["N"] ** p["gamma"] / p["sigma_a"]) + (Sigma_w / p["kappa"] ** 2)
        expectation = torch.mean((J_S_w**2) / denom)
        F_mS = p["N"] * ((1.0 - m_S) / p["kappa"] ** 2) * expectation
        return F_mS.item()

    # ------------------ ODE RHS and Solver (unchanged) ------------------ #

    # Corrected dmS_dtau function
    def dmS_dtau(self, tau: float, m_S_vec):
        m_S = float(m_S_vec[0])
        F_mS = self._calculate_F(m_S) # Variable is defined here as F_mS

        self.pbar.set_description(
            # The typo was here: F_m_S should be F_mS
            f"τ={tau:.3f}, m_S={m_S:.6f}, F(m_S)={F_mS:.6f}, "
            f"Accept={self.mcmc_acceptance_rate:.2%}, Step={self.mcmc_step_size:.2e}"
        )
        self.pbar.update(1)
        return [-m_S + F_mS]


    def solve(self, m0: float, t_span, t_eval):
        n_steps = len(t_eval) if t_eval is not None else 100
        self.pbar = tqdm(total=n_steps, desc="Solving ODE")
        sol = solve_ivp(
            fun=self.dmS_dtau,
            t_span=t_span,
            y0=[m0],
            t_eval=t_eval,
            method="RK45", # User found this faster for their current non-stiff case
        )
        self.pbar.close()
        return sol


# ------------------ Example run with Adaptive MALA ------------------ #
if __name__ == "__main__":
    parameters = {
        "d": 40,
        "N": 1024,
        "k": 4,
        "kappa": 0.0001, # A small, but not "frozen" kappa
        "sigma_a": 1.0,
        "sigma_w": 1.0,
        "gamma": 1.0,
        "n_samples_J_and_Sigma": 8000,
        "mcmc_samples": 20000,
        "mcmc_step_size": 1e-3, # Initial guess, will be adapted automatically
    }

    solver = DMFT_Solver(parameters)

    m0_test = 0.5
    t_span = [0, 10]
    t_eval = np.linspace(t_span[0], t_span[1], 50)

    print("Running simulation with adaptive MALA...")
    sol = solver.solve(m0_test, t_span, t_eval)
    print("Done.")

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(10, 6))
    plt.plot(sol.t, sol.y[0], "o-", label=f"$m_S(\\tau)$ starting from {m0_test}")
    
    final_m_S = sol.y[0, -1]
    plt.axhline(final_m_S, color='red', ls="--", label=f"Fixed point ≈ {final_m_S:.4f}")

    plt.xlabel(r"Fictitious time $\tau$")
    plt.ylabel(r"Order parameter $m_S$")
    plt.title(f"Adaptive MALA-DMFT Evolution (κ={parameters['kappa']})")
    plt.legend()
    plt.tight_layout()
    plt.show()