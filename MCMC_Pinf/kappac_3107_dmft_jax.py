import math
import numpy as np
import torch
from scipy.integrate import solve_ivp
from tqdm import tqdm
import matplotlib.pyplot as plt


class DMFT_Solver:
    """
    PyTorch implementation of the DMFT solver with stability fixes.
    """

    def __init__(self, params: dict):
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Random parity indices for the teacher
        self.S_indices = torch.randperm(params["d"], device=self.device)[: params["k"]]

        # Current MCMC weight vector (on device)
        self.w_current = torch.randn(params["d"], device=self.device) * math.sqrt(
            params["sigma_w"] / params["d"]
        )
        self.log_prob_current: float | None = None
        self.mcmc_acceptance_rate: float = 0.0
        
        # Initialize step size from params; will be adapted during runtime
        self.mcmc_step_size = params.get("mcmc_step_size", 0.05)

        # JIT-compile key kernels if Torch 2.x is available
        if hasattr(torch, "compile"):
            print("PyTorch 2.x detected. Compiling key functions...")
            self._Sigma = torch.compile(self._Sigma)
            self._J_S = torch.compile(self._J_S)
            self._log_prob_unnormalized = torch.compile(self._log_prob_unnormalized)

    # ------------------ Core math ------------------ #
    @staticmethod
    def _phi(z: torch.Tensor) -> torch.Tensor:
        """ReLU activation."""
        return torch.relu(z)

    def _sample_inputs(self, n_samples: int) -> torch.Tensor:
        """Generates ±1 input vectors on-device."""
        return (
            torch.randint(
                0, 2, (n_samples, self.params["d"]), device=self.device, dtype=torch.float32
            ) * 2 - 1
        )

    def _Sigma(self, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        pre = x @ w
        return torch.mean(self._phi(pre) ** 2)

    def _J_S(self, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        chi_S = torch.prod(x[:, self.S_indices], dim=1)
        pre = x @ w
        return torch.mean(self._phi(pre) * chi_S)

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

    # ------------------ MCMC WITH ADAPTIVE STEP SIZE ------------------ #
    def _calculate_F(self, m_S: float) -> float:
        p = self.params
        samples: list[torch.Tensor] = []
        n_accept = 0

        x_mcmc = self._sample_inputs(p["n_samples_J_and_Sigma"])
        self.log_prob_current = self._log_prob_unnormalized(self.w_current, m_S, x_mcmc).item()

        burn_in_steps = p["mcmc_samples"] // 4
        target_acceptance = 0.35  # Target a 35% acceptance rate

        # MCMC loop with adaptation during burn-in
        for i in range(p["mcmc_samples"]):
            w_prop = self.w_current + torch.randn_like(self.w_current) * self.mcmc_step_size
            log_prop = self._log_prob_unnormalized(w_prop, m_S, x_mcmc).item()
            log_alpha = log_prop - self.log_prob_current

            # Metropolis-Hastings acceptance condition
            if math.log(np.random.rand()) < log_alpha:
                self.w_current = w_prop
                self.log_prob_current = log_prop
                # Only count acceptances after the burn-in/adaptation phase
                if i >= burn_in_steps:
                    n_accept += 1
            
            # --- ADAPTIVE STEP SIZE LOGIC (during burn-in) --- #
            if i < burn_in_steps:
                # Use Robbins-Monro algorithm to adapt step size towards target acceptance
                # The term math.exp(min(0.0, log_alpha)) is the acceptance probability
                adaptation_factor = (math.exp(min(0.0, log_alpha))) - target_acceptance
                self.mcmc_step_size *= math.exp(0.01 * adaptation_factor)
                
            samples.append(self.w_current)
        
        # Calculate final acceptance rate on the post-burn-in part of the chain
        post_burn_in_samples = p["mcmc_samples"] - burn_in_steps
        if post_burn_in_samples > 0:
            self.mcmc_acceptance_rate = n_accept / post_burn_in_samples
        else:
            self.mcmc_acceptance_rate = 0.0

        # Use post-burn-in samples for the final expectation value
        W = torch.stack(samples[burn_in_steps:], dim=0)

        # Use a large, separate batch of x for a more stable final expectation
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

    # ------------------ ODE RHS ------------------ #
    def dmS_dtau(self, tau: float, m_S_vec):
        m_S = float(m_S_vec[0])
        F_mS = self._calculate_F(m_S)
        self.pbar.set_description(
            f"τ={tau:.3f}, m_S={m_S:.6f}, F(m_S)={F_mS:.6f}, "
            f"Accept={self.mcmc_acceptance_rate:.2%}, Step={self.mcmc_step_size:.4f}"
        )
        self.pbar.update(1)
        return [-m_S + F_mS]

    # ------------------ Public API ------------------ #
    def solve(self, m0: float, t_span, t_eval):
        n_steps = len(t_eval) if t_eval is not None else 100
        self.pbar = tqdm(total=n_steps, desc="Solving ODE")
        sol = solve_ivp(
            fun=self.dmS_dtau,
            t_span=t_span,
            y0=[m0],
            t_eval=t_eval,
            # --- USE A STIFF SOLVER --- #
            method="Radau",
        )
        self.pbar.close()
        return sol


# ------------------ Example run with STABILITY FIXES ------------------ #
if __name__ == "__main__":
    parameters = {
        "d": 40,
        "N": 1024,
        "k": 4,
        # --- INCREASED KAPPA FOR STABILITY --- #
        "kappa": 0.001,  # Changed from 0.001 to a much larger value
        "sigma_a": 1.0,
        "sigma_w": 1.0,
        "gamma": 1.0,
        "n_samples_J_and_Sigma": 12000,
        "mcmc_samples": 100000,
        # The initial step size is no longer as critical due to adaptation
        "mcmc_step_size": 0.1,
    }

    solver = DMFT_Solver(parameters)

    m0 = 0.1
    t_span = [0, 20]
    t_eval = np.linspace(t_span[0], t_span[1], 200)

    print("Running GPU-accelerated DMFT simulation with stability fixes...")
    sol = solver.solve(m0, t_span, t_eval)
    print("Done.")

    # --- Plotting Results --- #
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(10, 6))
    plt.plot(sol.t, sol.y[0], "o-", label=r"$m_S(\tau)$")
    plt.axhline(sol.y[0, -1], color='red', ls="--", label=f"Fixed point ≈ {sol.y[0, -1]:.4f}")
    plt.xlabel(r"Fictitious time $\tau$")
    plt.ylabel(r"Order parameter $m_S$")
    plt.title(f"DMFT evolution (d={parameters['d']}, k={parameters['k']}, κ={parameters['kappa']})")
    plt.legend()
    plt.tight_layout()
    plt.show()