#!/usr/bin/env python3
# --------------------------------------------------------------
#  DMFT / TAP fixed-point solver  –  κ-sweep
#  (Final version with progress bar for user feedback)
# --------------------------------------------------------------
import math, os, gc, json
import numpy as np
import jax, jax.numpy as jnp
from jax import value_and_grad, lax
import numpyro.distributions as dist
import optax
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table  import Table
from rich.progress import Progress # --- FIX: Added missing import for Progress bar ---

jax.config.update("jax_enable_x64", True)
console = Console()

def check_gpu():
    console.rule("[bold cyan]JAX devices[/bold cyan]")
    for i, d in enumerate(jax.devices()):
        console.print(f"  {i}: {d.platform.upper()}  ({d.device_kind})")
    if not any(d.platform == "gpu" for d in jax.devices()):
        console.print("[yellow]Warning: running on CPU.[/yellow]")

class DMFTSolver:
    def __init__(self, p: dict, key: jax.random.PRNGKey, S_idx: jnp.ndarray):
        self.p, self.key, self.S = p, key, S_idx
        self.X = self._sample_X(p["n_samples"])
        # Moved print to the main loop for cleaner progress bar output
    def _sample_X(self, n):
        self.key, sub = jax.random.split(self.key)
        return jax.random.choice(sub, jnp.array([-1., 1.]), shape=(n, self.p['d']))
    @staticmethod
    @jax.jit
    def _phi(z): return jnp.maximum(0., z)
    @staticmethod
    @jax.jit
    def _expectations(w, X, S_idx):
        phi = DMFTSolver._phi(X @ w)
        Σ   = jnp.mean(phi ** 2)
        J_S = jnp.mean(phi * jnp.prod(X[:, S_idx], axis=1))
        return Σ, J_S

    def _adam(self):
        return optax.adam(self.p['lr'])

    def saddle_point(self, m_S, χ_SS, w0=None):
        p, X, S, opt = self.p, self.X, self.S, self._adam()
        @jax.jit
        def opt_step(_, state):
            w_curr, opt_state = state
            def loss_fn(w):
                Σ, J = DMFTSolver._expectations(w, X, S)
                Σ_corr = Σ - (χ_SS / p['N']) * J**2
                α = p['N']**p['γ'] / p['σa'] + Σ_corr / p['κ']**2
                α_clipped = jnp.maximum(α, 1e-9)
                β = (1. - m_S) * J / p['κ']**2
                logP = dist.Normal(0., math.sqrt(p['σw']/p['d'])).log_prob(w).sum()
                logP += -0.5*jnp.log(α_clipped) + 0.5*β**2/α_clipped
                return -logP
            grads = value_and_grad(loss_fn)(w_curr)[1]
            updates, new_opt_state = opt.update(grads, opt_state, w_curr)
            return optax.apply_updates(w_curr, updates), new_opt_state
        if w0 is None:
            self.key, sub = jax.random.split(self.key)
            w0 = jax.random.normal(sub, (p['d'],)) * math.sqrt(p['σw']/p['d'])
        initial_state = (w0, opt.init(w0))
        w_star, _ = lax.fori_loop(0, p['opt_steps'], opt_step, initial_state)
        Σ, J = DMFTSolver._expectations(w_star, X, S)
        return w_star, float(Σ), float(J)

    def χ_next(self, m_S, χ_SS, Σ, J):
        p = self.p
        Σ_corr = Σ - (χ_SS / p['N']) * J**2
        α = p['N']**p['γ'] / p['σa'] + Σ_corr / p['κ']**2
        α = max(α, 1e-9)
        σ_a2, μ_a = 1./α, (1.-m_S)*J/p['κ']**2 * (1./α)
        a2 = σ_a2 + μ_a**2
        return (p['N']/p['κ']**2) * a2 * J**2
    
    def F(self, m_S, Σ, J, χ_SS):
        p = self.p
        Σ_TAP = Σ - (χ_SS / p['N']) * J**2
        denom = p['N']**p['γ']/p['σa'] + Σ_TAP / p['κ']**2
        return (p['N'] * (1.-m_S) / p['κ']**2) * (J**2 / (denom + 1e-12))

def solve_TAP(solver, w_start=None):
    state = {'w': w_start, 'χ': 1.}
    def H(m):
        m = float(np.clip(m, 1e-9, 1-1e-9))
        χ = state['χ']
        for _ in range(200):
            w, Σ, J = solver.saddle_point(m, χ, state['w'])
            χ_new   = solver.χ_next(m, χ, Σ, J)
            if abs(χ_new - χ) < 1e-7:
                χ = χ_new; break
            χ = 0.9 * χ + 0.1 * χ_new
            χ = min(χ, solver.p['N']*100)
        
        state.update(w=w, χ=χ)
        return solver.F(m, Σ, J, χ) - m
        
    a, b = 1e-6, 1-1e-6
    H_a, H_b = H(a), H(b)
    if np.sign(H_a) == np.sign(H_b):
        return (0., state) if H_a < 0 else (1., state)
    m_sol = brentq(H, a, b, xtol=1e-8, rtol=1e-8)
    H(m_sol)
    return m_sol, state

# ====================================================================
#                                MAIN
# ====================================================================
if __name__ == "__main__":
    check_gpu()
    BASE = dict(d=30, N=1024, k=4, σa=1., σw=1., γ=1., lr=2e-3, opt_steps=40000)
    κ_list = sorted([0.1, 0.01, 0.005, 0.001, 1e-4], reverse=True)

    master = jax.random.PRNGKey(42)
    S_idx  = jax.random.permutation(master, jnp.arange(BASE['d']))[:BASE['k']]
    console.print(f"\n[bold green]Teacher parity indices[/bold green]: {np.array(S_idx)}\n")
    save_dir = "/home/goring/mean_field_langevin/MCMC_Pinf_1N/results/0708_d30k4_diagosntic3"
    os.makedirs(save_dir, exist_ok=True)

    warm_p = BASE | dict(kappa=1., κ=1., n_samples=32768, opt_steps=4)
    warm   = DMFTSolver(warm_p, jax.random.PRNGKey(0), S_idx)
    w_warm, _, _ = warm.saddle_point(m_S=0.5, χ_SS=1.)
    w_warm.block_until_ready()
    console.print("[green]JIT warm-up done.[/green]\n")

    results, w_anneal = {}, w_warm

    with Progress(console=console) as progress:
        task = progress.add_task("[cyan]Running κ-sweep...", total=len(κ_list))
        for κ in κ_list:
            progress.update(task, description=f"[cyan]Running κ={κ:.4g}...")
            
            n_samples = int(min(65536, math.ceil(200*BASE['N']/κ**2)))
            lr = 1e-3 if κ > 1e-3 else 5e-5
            opt_steps = 60000 if κ <= 0.01 else 20000

            pκ   = BASE | dict(kappa=κ, κ=κ, n_samples=n_samples, lr=lr, opt_steps=opt_steps)
            key, master = jax.random.split(master)
            
            console.print(f"κ={pκ['kappa']:.3g} | MC samples={pκ['n_samples']} | Opt Steps={pκ['opt_steps']} | LR={pκ['lr']:.2g}")
            solver = DMFTSolver(pκ, key, S_idx)
            m_S, st = solve_TAP(solver, w_start=w_anneal)

            w_anneal = st['w']
            Σ, J     = solver._expectations(w_anneal, solver.X, solver.S)
            diag = dict(m_S=m_S, χ_SS=st['χ'], Σ=float(Σ), J=float(J), ratio=float(J**2/(Σ+1e-12)))
            results[str(κ)] = diag

            t = Table(title=f"Results for κ={κ}")
            t.add_column("Metric"); t.add_column("Value")
            for k, v in diag.items(): t.add_row(k, f"{v:.6g}")
            console.print(t)

            progress.update(task, advance=1)

    with open(os.path.join(save_dir, "results.json"), "w") as f: json.dump(results, f, indent=2)
    
    κs = sorted(float(k) for k in results)
    mS = [results[str(k)]['m_S'] for k in κs]
    plt.plot(κs, mS, "o-"); plt.gca().set_xscale("log"); plt.gca().invert_xaxis()
    plt.axhline(1., ls="--"); plt.xlabel("κ"); plt.ylabel("$m_S$")
    plt.title("Order parameter vs noise"); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mS_vs_kappa.png"))
    plt.show()