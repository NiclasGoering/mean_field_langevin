import torch, math, itertools

# ---------- hyper-parameters ---------------------------------
d, k = 10, 2                     # input dim and parity order
N     = 1024
gamma = 1.0
sigma_w, sigma_a = 1.0, 1.0
kappa           = 0.001
samples, burn   = 8000, 2000
damping         = 0.5
device          = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------- helper: chi_A(x) and J_A(w) -----------------------
X_bank = (2*torch.randint(0,2,(200_000,d),device=device)-1).float()
S      = torch.tensor(range(k), device=device)      # target subset

def chi(A, X):           # parity
    return torch.prod(X[:, A], dim=1)

def Sigma(w):            # ⟨φ²⟩ over X_bank
    z = X_bank @ w
    return (torch.relu(z)**2).mean()

def J_A(w, A):
    z = X_bank @ w
    return (torch.relu(z) * chi(A, X_bank)).mean()

# ---------- single-neuron distribution p̃_m(w) ----------------
g_w = math.sqrt(sigma_w/d)
g_a = math.sqrt(sigma_a/N**gamma)

def log_p_tilde(w, m):
    """log p̃_m(w)  up to a constant."""
    w2   = 0.5*d/sigma_w * w.dot(w)
    Σ    = Sigma(w)
    B    = (J_A(w, S) - m*J_A(w,S)) / kappa**2     # only m_S appears
    A    = N**gamma/sigma_a + Σ/kappa**2
    mu2A = 0.5 * B*B/A                             # (μ²)/(2σ²)
    return -w2 + mu2A

# ---------- new grad_log_p ----------------------------------
def grad_log_p(w, m):
    w = w.detach().clone().requires_grad_(True)
    lp = log_p_tilde(w, m)
    g, = torch.autograd.grad(lp, w)
    return g


# ---------- Langevin sampler ---------------------------------
def sample_w(m, n_samp, burn, η=5e-3):
    w = torch.randn(d, device=device)*g_w
    traj = []
    for t in range(burn+n_samp):
        g = grad_log_p(w, m)
        w = w + 0.5*η*g + math.sqrt(η)*torch.randn_like(w)
        if t>=burn: traj.append(w.clone())
    return torch.stack(traj)

# ---------- fixed-point iteration ----------------------------
m = torch.tensor(0.0, device=device)
for it in range(30):
    Ws = sample_w(m, samples, burn)
    with torch.no_grad():
        Js = torch.stack([J_A(w,S) for w in Ws])
        Σs = torch.stack([Sigma(w) for w in Ws])
        A  = N**gamma/sigma_a + Σs/kappa**2
        B  = (Js - m*Js)/kappa**2
        mu = B/A
        m_new = N * torch.mean(mu*Js)
    m = (1-damping)*m + damping*m_new
    print(f"iter {it:2d}  m_S = {m.item():.6f}")
    if abs(m_new-m) < 1e-4: break

















