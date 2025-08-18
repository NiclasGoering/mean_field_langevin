#!/usr/bin/env python3
# --------------------------------------------------------------
#  DMFT / TAP fixed-point solver  –  κ-sweep
#  (safe χ update + adaptive MC count + JIT-compiled optimizer)
# --------------------------------------------------------------
import math, os, gc, json
import numpy as np
import jax, jax.numpy as jnp
from jax import value_and_grad, lax # Import lax for fori_loop
import numpyro.distributions as dist
import optax
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table  import Table

jax.config.update("jax_enable_x64", True)
console = Console()


# ----------------------------------------------------------------
#  helpers
# ----------------------------------------------------------------
def check_gpu():
    console.rule("[bold cyan]JAX devices[/bold cyan]")
    for i, d in enumerate(jax.devices()):
        console.print(f"  {i}: {d.platform.upper()}  ({d.device_kind})")
    if not any(d.platform == "gpu" for d in jax.devices()):
        console.print("[yellow]Warning: running on CPU.[/yellow]")


# ----------------------------------------------------------------
#  TAP / DMFT solver
# ----------------------------------------------------------------
class DMFTSolver:
    # ------------------------------------------------------------
    def __init__(self, p: dict, key: jax.random.PRNGKey, S_idx: jnp.ndarray):
        self.p   = p
        self.key = key
        self.S   = S_idx
        self.X   = self._sample_X(p["n_samples"])
        console.print(f"κ={p['kappa']:.3g} | MC samples={p['n_samples']}")

    # ------------------------------------------------------------
    def _sample_X(self, n):
        self.key, sub = jax.random.split(self.key)
        return jax.random.choice(sub, jnp.array([-1., 1.]), shape=(n, self.p['d']))

    # ------------------------------------------------------------
    @staticmethod
    @jax.jit
    def _phi(z):
        return jnp.maximum(0., z)

    # ------------------------------------------------------------
    @staticmethod
    @jax.jit
    def _expectations(w, X, S_idx):
        z   = X @ w
        phi = DMFTSolver._phi(z)
        Σ   = jnp.mean(phi ** 2)
        χ_S = jnp.prod(X[:, S_idx], axis=1)
        J_S = jnp.mean(phi * χ_S)
        return Σ, J_S

    # ------------------------------------------------------------
    def _adam(self):
        lr = self.p['lr']
        return optax.adam(optax.constant_schedule(lr))

    # ------------------------------------------------------------
    def saddle_point(self, m_S, χ_SS, w0=None):
        """Minimise −log P(w | m_S, χ_SS) using a JIT-compiled loop."""
        p   = self.p
        X,S = self.X, self.S
        opt = self._adam()

        def loss_fn(w):
            Σ, J = DMFTSolver._expectations(w, X, S)
            Σ_corr = Σ - (χ_SS / p['N']) * J**2
            α = (p['N']**p['γ'] / p['σa'] + Σ_corr / p['κ']**2)
            α = jnp.maximum(α, 1e-9)
            β = (1. - m_S) * J / p['κ']**2
            logP = dist.Normal(0., math.sqrt(p['σw']/p['d'])).log_prob(w).sum()
            logP += -0.5*jnp.log(α) + 0.5*β**2/α
            return -logP

        # --- PERFORMANCE FIX: JIT-compile the optimization step ---
        @jax.jit
        def opt_step(_, state):
            w_curr, opt_state = state
            loss, grads = value_and_grad(loss_fn)(w_curr)
            updates, new_opt_state = opt.update(grads, opt_state, w_curr)
            new_w = optax.apply_updates(w_curr, updates)
            return new_w, new_opt_state
        
        if w0 is None:
            self.key, sub = jax.random.split(self.key)
            w0 = jax.random.normal(sub, (p['d'],)) * math.sqrt(p['σw']/p['d'])

        # --- PERFORMANCE FIX: Run the entire loop on the GPU ---
        initial_state = (w0, opt.init(w0))
        w_star, _ = lax.fori_loop(0, p['opt_steps'], opt_step, initial_state)

        Σ, J = DMFTSolver._expectations(w_star, X, S)
        return w_star, float(Σ), float(J)

    # ------------------------------------------------------------
    def χ_next(self, m_S, χ_SS, Σ, J):
        # ... (This function remains the same)
        p = self.p
        Σ_corr = Σ - (χ_SS / p['N']) * J**2
        α = (p['N']**p['γ'] / p['σa'] + Σ_corr / p['κ']**2)
        α = max(α, 1e-9)
        σ_a2 = 1./α
        μ_a  = (1.-m_S)*J/p['κ']**2 * σ_a2
        a2   = σ_a2 + μ_a**2
        J_safe = max(abs(J), 1e-300)
        a2_safe= max(a2, 1e-300)
        return (p['N']/p['κ']**2) * a2_safe * J_safe**2

    # ------------------------------------------------------------
    def F(self, m_S, Σ, J):
        # ... (This function remains the same)
        p = self.p
        denom = (p['N']**p['γ']/p['σa'] + Σ/p['κ']**2)
        return (p['N']*(1.-m_S)/p['κ']**2) * (J**2/denom)

# ----------------------------------------------------------------
#  The rest of your script (solve_TAP, main block, plotting)
#  remains exactly the same.
# ----------------------------------------------------------------

def solve_TAP(solver, w_start=None):
    state = {'w': w_start, 'χ': 1.}

    def H(m):
        m = float(np.clip(m, 1e-9, 1-1e-9))
        χ = state['χ']
        # inner χ-fixed-point
        for _ in range(200):
            w, Σ, J = solver.saddle_point(m, χ, state['w'])
            χ_new   = solver.χ_next(m, χ, Σ, J)
            if abs(χ_new-χ) < 1e-7:
                χ = χ_new
                break
            χ = 0.2*χ_new + 0.8*χ 
            χ = min(χ, solver.p['N']*100)     # damping
        state.update(w=w, χ=χ)
        return solver.F(m, Σ, J) - m

    a, b = 1e-6, 1-1e-6
    if np.sign(H(a)) == np.sign(H(b)):
        return (0., state) if H(a) < 0 else (1., state)

    m_sol = brentq(H, a, b, xtol=1e-8, rtol=1e-8)
    H(m_sol)                          # final update
    return m_sol, state

if __name__ == "__main__":
    check_gpu()

    BASE = dict(d=30, N=1024, k=4, σa=1., σw=1., γ=1., lr=2e-3, opt_steps=40000)
    κ_list = sorted([0.1, 0.01, 0.0075, 0.005, 0.0025, 0.001, 1e-4], reverse=True)

    master = jax.random.PRNGKey(42)
    S_idx  = jax.random.permutation(master, jnp.arange(BASE['d']))[:BASE['k']]
    console.print(f"\n[bold green]Teacher parity indices[/bold green]: {np.array(S_idx)}\n")

    save_dir = "/home/goring/mean_field_langevin/MCMC_Pinf_1N/results/0708_d30k4_diagosntic3_old"
    os.makedirs(save_dir, exist_ok=True)

    warm_p = BASE | dict(kappa=1., κ=1., n_samples=32768, opt_steps=4)
    warm   = DMFTSolver(warm_p, jax.random.PRNGKey(0), S_idx)
    w_warm, _, _ = warm.saddle_point(m_S=0.5, χ_SS=1.)
    w_warm.block_until_ready()
    console.print("[green]JIT warm-up done.[/green]\n")

    results, w_anneal = {}, w_warm

    for κ in κ_list:
        n_samples = int(min(65536, math.ceil(200*BASE['N']/κ**2)))
        pκ   = BASE | dict(kappa=κ, κ=κ, n_samples=n_samples)
        key, master = jax.random.split(master)
        console.rule(f"[bold magenta]κ = {κ:.3g}   |   MC = {n_samples}[/bold magenta]")

        solver = DMFTSolver(pκ, key, S_idx)
        m_S, st = solve_TAP(solver, w_start=w_anneal)

        w_anneal = st['w']
        Σ, J     = solver._expectations(w_anneal, solver.X, solver.S)
        diag = dict(m_S=m_S, χ_SS=st['χ'], Σ=Σ, J=J, ratio=J**2/(Σ+1e-12))
        results[str(κ)] = diag

        t = Table(title=f"κ={κ}")
        for k, v in diag.items():
            t.add_row(k, f"{v:.6g}")
        console.print(t)

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    κs = sorted(float(k) for k in results)
    mS = [results[str(k)]['m_S'] for k in κs]
    plt.plot(κs, mS, "o-"); plt.gca().set_xscale("log"); plt.gca().invert_xaxis()
    plt.axhline(1., ls="--"); plt.xlabel("κ"); plt.ylabel("$m_S$")
    plt.title("Order parameter vs noise"); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mS_vs_kappa.png"))
    plt.show()