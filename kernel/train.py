#!/usr/bin/env python3
"""
Train a deep ReLU FFNN on merged staircase functions over {±1}^d and
compute last-hidden kernel metrics exactly as specified by the user.

Saves:
- <save_dir>/metrics.csv                (epoch, train_mse, test_mse)
- <save_dir>/kernel/summary.csv         (Ak, deff, ΔK, ρk, ρ̄k, Sk, C, G, top-k evals)
- <save_dir>/kernel/shares.csv          (per-layer transfer shares)
- <save_dir>/kernel/epoch_XXXX/*.npy    (eigpairs, transfers)
"""

import os, csv, random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import helpers as H


# ============== data: merged staircase labels ==============

def parse_interaction_spec(spec: str) -> List[List[int]]:
    cleaned = spec.replace(" ", "")
    parts = cleaned.split("+")
    terms: List[List[int]] = []
    for p in parts:
        assert p.startswith("{") and p.endswith("}"), f"Malformed term '{p}'."
        inside = p[1:-1]
        assert inside != "", f"Empty term '{p}'."
        idxs = inside.split(",")
        term = []
        for s in idxs:
            assert s.isdigit(), f"Non-integer index '{s}'."
            j = int(s); assert j > 0, "Indices must be 1-based positive."
            term.append(j - 1)
        terms.append(term)
    return terms

def generate_inputs(n: int, d: int, device: torch.device) -> torch.Tensor:
    x = torch.randint(0, 2, (n, d), device=device, dtype=torch.int8).to(torch.float32)
    return 2.0 * x - 1.0

def compute_labels(x: torch.Tensor, interactions: List[List[int]]) -> torch.Tensor:
    y = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    for term in interactions:
        y += torch.prod(x[:, term], dim=1)
    return y.unsqueeze(1)

class StaircaseDataset(Dataset):
    def __init__(self, n: int, d: int, interactions: List[List[int]],
                 device: torch.device, noise_std: float = 0.0, seed: int = 0):
        _ = seed
        self.x = generate_inputs(n, d, device)
        self.y = compute_labels(self.x, interactions)
        if noise_std > 0:
            self.y = self.y + noise_std * torch.randn_like(self.y)
        self.n = n
    def __len__(self): return self.n
    def __getitem__(self, idx: int): return self.x[idx], self.y[idx]


# ============== model (ReLU) ==============

class MLP(nn.Module):
    def __init__(self, d_in: int, width: int, depth: int, d_out: int = 1):
        super().__init__()
        layers: List[nn.Module] = []
        if depth <= 0:
            layers.append(nn.Linear(d_in, d_out))
        else:
            layers.append(nn.Linear(d_in, width))
            layers.append(nn.ReLU())
            for _ in range(depth - 1):
                layers.append(nn.Linear(width, width))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(width, d_out))
        self.net = nn.Sequential(*layers)
        self.apply(self._init)
    @staticmethod
    def _init(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x): return self.net(x)


# ============== training utils ==============

@dataclass
class TrainConfig:
    # data
    train_size: int
    test_size: int
    dim: int
    spec: str
    noise_std: float
    shuffle: bool
    # model
    width: int
    depth: int
    # training
    epochs: int
    lr: float
    mode: str          # 'gd' or 'sgd'
    batch_size: int    # for SGD
    optimizer: str     # 'sgd' or 'adam'
    momentum: float
    # infra
    save_dir: str
    seed: int
    device: str        # 'cuda' or 'cpu'
    # metrics (last-hidden kernel on chosen set)
    compute_kernel: bool
    compute_kernel_every: int
    kernel_set: str          # 'train' or 'test'
    max_kernel_points: int   # 0 = use all; else subset to this many
    top_k: int               # top-k for last layer
    lower_top_k: int         # how many lower-layer modes p to include in T
    betas: Tuple[float, float, float]  # (β1, β2, β3)
    track_U: bool            # save U_top per epoch
    save_transfers: bool     # save T^{(ℓ)}_{p→i}

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device(name: str) -> torch.device:
    return torch.device("cuda" if (name=="cuda" and torch.cuda.is_available()) else "cpu")

def evaluate_mse(model: nn.Module, loader: DataLoader) -> float:
    model.eval(); loss_fn = nn.MSELoss()
    tot=0.0; n=0
    with torch.no_grad():
        for xb,yb in loader:
            pred = model(xb); loss = loss_fn(pred, yb)
            b = xb.shape[0]; tot += float(loss.item())*b; n += b
    return tot/max(n,1)


def main():
    # ---------- choose everything here ----------
    cfg = TrainConfig(
        train_size=15000, test_size=5000, dim=30,
        spec="{1,2}+{1,2,3,4}+{8}", noise_std=0.0, shuffle=True,
        width=512, depth=4,
        epochs=8000, lr=1e-3, mode="gd", batch_size=512,
        optimizer="sgd", momentum=0.9,
        save_dir="results/test_2610_2_shuffle2", seed=123, device="cuda",
        compute_kernel=True, compute_kernel_every=5, kernel_set="train",
        max_kernel_points=2048, top_k=15, lower_top_k=15,
        betas=(0.2, 0.2, 0.2), track_U=True, save_transfers=True
    )
    # -------------------------------------------

    device = get_device(cfg.device)
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    # data
    interactions = parse_interaction_spec(cfg.spec)
    need_dim = max([i for t in interactions for i in t]) + 1
    assert cfg.dim >= need_dim, f"dim={cfg.dim} < required {need_dim}"
    train_ds = StaircaseDataset(cfg.train_size, cfg.dim, interactions, device, cfg.noise_std, cfg.seed)
    test_ds  = StaircaseDataset(cfg.test_size,  cfg.dim, interactions, device, 0.0, cfg.seed+1)

    if cfg.shuffle:
        idx = torch.randperm(train_ds.y.shape[0], device=train_ds.y.device)
        train_ds.y = train_ds.y[idx]

    if cfg.mode == "gd":
        train_loader = DataLoader(train_ds, batch_size=cfg.train_size, shuffle=False)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=4096, shuffle=False)

    # model & opt
    model = MLP(cfg.dim, cfg.width, cfg.depth, 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr) if cfg.optimizer=="adam" \
          else torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)

    # logs
    mse_csv = os.path.join(cfg.save_dir, "metrics.csv")
    with open(mse_csv, "w", newline="") as f: csv.writer(f).writerow(["epoch","train_mse","test_mse"])

    # kernel cache across epochs
    kernel_dir = os.path.join(cfg.save_dir, "kernel")
    kernel_cache = H.make_kernel_cache(kernel_dir, k=cfg.top_k)

    # epoch 0 eval
    tr_mse0 = evaluate_mse(model, DataLoader(train_ds, batch_size=4096, shuffle=False))
    te_mse0 = evaluate_mse(model, test_loader)
    with open(mse_csv, "a", newline="") as f: csv.writer(f).writerow([0, tr_mse0, te_mse0])
    print(f"[Epoch 0] train_mse={tr_mse0:.6f}  test_mse={te_mse0:.6f}")

    # metrics at epoch 0
    if cfg.compute_kernel:
        Xset, yset = (train_ds.x, train_ds.y) if cfg.kernel_set=="train" else (test_ds.x, test_ds.y)
        if cfg.max_kernel_points>0 and Xset.shape[0]>cfg.max_kernel_points:
            idx = torch.randperm(Xset.shape[0], device=Xset.device)[:cfg.max_kernel_points]
            Xk, yk = Xset[idx], yset[idx]
        else:
            Xk, yk = Xset, yset
        H.compute_and_log_all_metrics(
            out_dir=kernel_dir, model=model, X=Xk, y=yk,
            top_k=cfg.top_k, lower_top_k=cfg.lower_top_k,
            epoch=0, betas=cfg.betas,
            track_U=cfg.track_U, save_transfers=cfg.save_transfers,
            kernel_cache=kernel_cache
        )

    # train loop
    for epoch in range(1, cfg.epochs+1):
        model.train()
        for xb,yb in train_loader:
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = nn.MSELoss()(pred, yb)
            loss.backward(); opt.step()

        tr_mse = evaluate_mse(model, DataLoader(train_ds, batch_size=4096, shuffle=False))
        te_mse = evaluate_mse(model, test_loader)
        with open(mse_csv, "a", newline="") as f: csv.writer(f).writerow([epoch, tr_mse, te_mse])

        should_metrics = cfg.compute_kernel and (epoch % cfg.compute_kernel_every == 0 or epoch == cfg.epochs)
        if should_metrics:
            Xset, yset = (train_ds.x, train_ds.y) if cfg.kernel_set=="train" else (test_ds.x, test_ds.y)
            if cfg.max_kernel_points>0 and Xset.shape[0]>cfg.max_kernel_points:
                idx = torch.randperm(Xset.shape[0], device=Xset.device)[:cfg.max_kernel_points]
                Xk, yk = Xset[idx], yset[idx]
            else:
                Xk, yk = Xset, yset
            H.compute_and_log_all_metrics(
                out_dir=kernel_dir, model=model, X=Xk, y=yk,
                top_k=cfg.top_k, lower_top_k=cfg.lower_top_k,
                epoch=epoch, betas=cfg.betas,
                track_U=cfg.track_U, save_transfers=cfg.save_transfers,
                kernel_cache=kernel_cache
            )

        if epoch % max(1, cfg.epochs//20) == 0 or epoch == cfg.epochs:
            print(f"[Epoch {epoch}] train_mse={tr_mse:.6f}  test_mse={te_mse:.6f}")

    print(f"Saved to: {cfg.save_dir}")


if __name__ == "__main__":
    main()