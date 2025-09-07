import math
import json
from typing import List, Optional

import torch
import torch.nn.functional as F

# ------------------------------------------------------------
# Minimal Winner-Take-All (WTA) feature learning for a 1-hidden-layer NN
# ------------------------------------------------------------
# Model: f(x) = sum_{k=1..K} a_k * phi(w_k^T x)
# Greedy training loop:
#  - Keep a running residual r = y - f(X)
#  - Sample M candidate weights w ~ N(0, (sigma_w^2/d) I)
#  - For each candidate, compute phi = phi(X @ w) and the score
#        s(w) = <phi, r>^2 / (kappa^2 P * (||phi||^2 + kappa^2 P / sigma_a^2)) - rho * ||w||^2
#    (this is the single-neuron energy gain after integrating out a and adding a simple w-prior)
#  - Pick the winner w* = argmax s(w)
#  - Closed-form amplitude a* = <phi*, r> / (||phi*||^2 + kappa^2 P / sigma_a^2)
#  - Update f <- f + a* phi*, r <- r - a* phi*
#  - Repeat K times (or until the best score <= 0)
# Notes:
#  * This is the simplest feature-learning mechanism: at each step the most aligned neuron "wins".
#  * Equivalent to matching pursuit in a learned (nonlinear) dictionary.
#  * No MALA/SGLD/EMA/ARD machinery; just WTA + closed-form a.
# ------------------------------------------------------------

# ---------------------- utilities ---------------------------

def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def activation(z: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "relu":
        return F.relu(z)
    if kind == "tanh":
        return torch.tanh(z)
    raise ValueError(f"Unknown activation: {kind}")


def parity_character(X_pm1: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    if S.numel() == 0:
        return torch.ones(X_pm1.shape[0], device=X_pm1.device, dtype=X_pm1.dtype)
    return X_pm1[:, S].prod(dim=1).to(X_pm1.dtype)


def parse_sets(spec: str) -> List[List[int]]:
    import re
    blocks = re.findall(r"\{([^}]*)\}", spec)
    out: List[List[int]] = []
    for s in blocks:
        toks = [t.strip() for t in s.split(",") if t.strip() != ""]
        out.append(sorted(map(int, toks)))
    if not out:
        raise ValueError("bad teacher spec")
    return out


def generate_parity(P: int, d: int, sets: List[torch.Tensor], device, dtype):
    g = torch.Generator(device=device).manual_seed(123)
    X = (torch.randint(0, 2, (P, d), generator=g, device=device, dtype=torch.int8).to(dtype) * 2.0 - 1.0)
    Ccols = [parity_character(X, S) for S in sets]
    C = torch.stack(Ccols, dim=1) if Ccols else torch.zeros(P, 0, device=device, dtype=dtype)
    y = C.sum(dim=1, keepdim=True)
    return X, y, C


# ---------------------- core WTA solver ---------------------

class WTASolver:
    def __init__(
        self,
        d: int,
        act: str = "relu",
        sigma_w: float = 1.0,
        sigma_a: float = 1.0,
        kappa: float = 1e-2,
        rho: float = 0.0,  # L2 precision on w (simple isotropic prior term 0.5*rho*||w||^2)
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.d = d
        self.act = act
        self.sigma_w = float(sigma_w)
        self.sigma_a = float(sigma_a)
        self.kappa = float(kappa)
        self.rho = float(rho)
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.dtype = dtype
        # learned atoms
        self.W: List[torch.Tensor] = []
        self.a: List[torch.Tensor] = []

    @torch.no_grad()
    def _sample_weights(self, M: int) -> torch.Tensor:
        return torch.randn(M, self.d, device=self.device, dtype=self.dtype) * (self.sigma_w / math.sqrt(self.d))

    @torch.no_grad()
    def _score_candidates(self, X: torch.Tensor, r: torch.Tensor, Wc: torch.Tensor):
        # X: (P,d), r: (P,1), Wc: (M,d)
        Z = X @ Wc.t()  # (P,M)
        Phi = activation(Z, self.act)
        P = X.shape[0]
        # inner products
        v = Phi.t() @ r  # (M,1)
        num = (v.squeeze(1) ** 2)  # (M,)
        denom = (Phi * Phi).sum(dim=0) + (self.kappa ** 2) * P / (self.sigma_a ** 2)  # (M,)
        score = num / ( (self.kappa ** 2) * P * denom + 1e-30) - self.rho * (Wc * Wc).sum(dim=1)
        return score, Phi, v, denom

    @torch.no_grad()
    def step(self, X: torch.Tensor, r: torch.Tensor, M: int):
        Wc = self._sample_weights(M)
        score, Phi, v, denom = self._score_candidates(X, r, Wc)
        j = int(torch.argmax(score).item())
        w_star = Wc[j:j+1]  # (1,d)
        phi_star = Phi[:, j:j+1]  # (P,1)
        a_star = v[j:j+1] / denom[j:j+1]  # (1,1)
        return w_star.clone(), a_star.clone(), phi_star.clone(), float(score[j].item())

    @torch.no_grad()
    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        K: int = 64,
        M: int = 4096,
        stop_if_nonpositive: bool = True,
        log_every: int = 1,
    ):
        X = X.to(self.device, self.dtype)
        y = y.to(self.device, self.dtype)
        P = X.shape[0]
        f = torch.zeros_like(y)
        r = (y - f)

        hist = {
            "iter": [],
            "train_mse": [],
            "best_score": [],
            "num_atoms": [],
        }

        for k in range(1, K + 1):
            w_star, a_star, phi_star, s = self.step(X, r, M)
            if stop_if_nonpositive and s <= 0.0:
                # no feature reduces the energy -> stop
                break
            # update state
            f = f + a_star * phi_star
            r = y - f
            self.W.append(w_star.squeeze(0).detach())
            self.a.append(a_star.squeeze(0).detach())

            if (k % log_every == 0) or (k in (1, K)):
                train_mse = float(((y - f) ** 2).mean().item())
                hist["iter"].append(k)
                hist["train_mse"].append(train_mse)
                hist["best_score"].append(float(s))
                hist["num_atoms"].append(len(self.W))
                print(json.dumps({
                    "k": k,
                    "train_mse": train_mse,
                    "best_score": float(s),
                    "num_atoms": len(self.W),
                }))

        return hist

    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if not self.W:
            return torch.zeros(X.shape[0], 1, device=self.device, dtype=self.dtype)
        W = torch.stack(self.W, dim=0)  # (K,d)
        a = torch.stack(self.a, dim=0).view(-1, 1)  # (K,1)
        Z = X.to(self.device, self.dtype) @ W.t()
        Phi = activation(Z, self.act)
        f = Phi @ a
        return f


# ---------------------- tiny demo ---------------------------
if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Teacher: parity {0,1,2,3} on d=35
    d = 35
    spec = "{0,1,2,3}"
    sets = [torch.tensor(S, device=device, dtype=torch.long) for S in parse_sets(spec)]

    # Data
    P_train = 2500
    X, y, C = generate_parity(P_train, d, sets, device, dtype)

    # Model
    solver = WTASolver(d=d, act="relu", sigma_w=1.0, sigma_a=1.0, kappa=7.5e-3, rho=0.2, device=device, dtype=dtype)

    # Train
    hist = solver.fit(X, y, K=12800, M=8192, stop_if_nonpositive=True, log_every=5)

    # Held-out evaluation
    P_eval = 10000
    g = torch.Generator(device=device).manual_seed(777)
    X_eval = (torch.randint(0, 2, (P_eval, d), generator=g, device=device, dtype=torch.int8).to(dtype) * 2.0 - 1.0)
    f_eval = solver.predict(X_eval)

    # Report simple metrics
    # Alignments with teacher parities (mode amplitudes m_S)
    C_eval = torch.stack([parity_character(X_eval, S) for S in sets], dim=1)
    m_S = (C_eval.t() @ f_eval.squeeze(1)) / float(P_eval)  # (M,)

    f2_bar = float((f_eval.squeeze(1) ** 2).mean().item())
    ones = torch.ones(len(sets), device=device, dtype=dtype)
    G = (C_eval.t() @ C_eval) / float(P_eval)
    v = (C_eval.t() @ f_eval.squeeze(1)) / float(P_eval)
    half_mse_modes = 0.5 * float(((ones - v) ** 2).sum().item())
    mTGm = float((v.view(1, -1) @ G @ v.view(-1, 1)).item())
    noise = f2_bar - 2.0 * float((ones @ v).item()) + mTGm

    print(json.dumps({
        "m_S": [float(x) for x in m_S.tolist()],
        "half_mse_modes": half_mse_modes,
        "half_noise": 0.5 * float(noise),
        "half_mse_total_ms": half_mse_modes + 0.5 * float(noise),
        "num_atoms": len(solver.W),
    }, indent=2))
