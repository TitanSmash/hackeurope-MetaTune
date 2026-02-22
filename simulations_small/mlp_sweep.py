#!/usr/bin/env python3
"""
Hyperparam sweep for a simple MLP regressor on synthetic function curves.

Assumes your generator created:
  synthetic_functions.npz with X_train, y_train, X_test, y_test
Shapes:
  X_*: (N, L, 1)
  y_*: (N, L, 1)

We flatten sequences into point samples:
  (N*L, 1) -> (N*L, 1)
so the MLP learns y = f(x) across mixed function families.

Outputs:
  results.jsonl  (one JSON per run)
"""

from __future__ import annotations
import argparse
import json
import math
import random
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# -----------------------
# Model
# -----------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden_dim), nn.GELU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = hidden_dim
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------
# Utils
# -----------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def flatten_npz(npz_path: str) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    data = np.load(npz_path, allow_pickle=True)
    Xtr = data["X_train"].astype(np.float32)  # (N, L, 1)
    ytr = data["y_train"].astype(np.float32)
    Xte = data["X_test"].astype(np.float32)
    yte = data["y_test"].astype(np.float32)

    # Flatten: (N, L, 1) -> (N*L, 1)
    Xtr_f = Xtr.reshape(-1, 1)
    ytr_f = ytr.reshape(-1, 1)
    Xte_f = Xte.reshape(-1, 1)
    yte_f = yte.reshape(-1, 1)

    return (Xtr_f, ytr_f), (Xte_f, yte_f)

@torch.no_grad()
def eval_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        diff = pred - y
        mse_sum += float((diff ** 2).sum().item())
        mae_sum += float(diff.abs().sum().item())
        n += int(y.numel())
    mse = mse_sum / max(1, n)
    mae = mae_sum / max(1, n)
    rmse = math.sqrt(mse)
    return {"mse": mse, "rmse": rmse, "mae": mae}


# -----------------------
# Training one run
# -----------------------

def train_one_run(npz_path: str, h: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expected hyperparams:
      seed, epochs, batch_size, lr, weight_decay,
      hidden_dim, n_layers, dropout,
      grad_clip
    """
    set_seed(int(h.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    (Xtr, ytr), (Xte, yte) = flatten_npz(npz_path)

    # Torch datasets
    ds_tr = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    ds_te = TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))

    bs = int(h.get("batch_size", 1024))
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=0, drop_last=False)
    dl_te = DataLoader(ds_te, batch_size=bs, shuffle=False, num_workers=0, drop_last=False)

    model = MLP(
        in_dim=1,
        hidden_dim=int(h.get("hidden_dim", 128)),
        n_layers=int(h.get("n_layers", 3)),
        dropout=float(h.get("dropout", 0.0)),
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(h.get("lr", 1e-3)),
        weight_decay=float(h.get("weight_decay", 1e-5)),
    )

    loss_fn = nn.MSELoss()
    grad_clip = float(h.get("grad_clip", 1.0))
    epochs = int(h.get("epochs", 20))

    best = {"rmse": float("inf"), "epoch": -1, "metrics": None}

    for ep in range(1, epochs + 1):
        model.train()
        for x, y in dl_tr:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        m = eval_metrics(model, dl_te, device)
        if m["rmse"] < best["rmse"]:
            best = {"rmse": m["rmse"], "epoch": ep, "metrics": m}

    return {
        "hparams": dict(h),
        "best_epoch": best["epoch"],
        **{f"best_{k}": v for k, v in best["metrics"].items()},
    }


# -----------------------
# Sweep strategies
# -----------------------

def sample_loguniform(rng: random.Random, lo: float, hi: float) -> float:
    return 10 ** rng.uniform(math.log10(lo), math.log10(hi))

def random_sweep(n_runs: int, base_seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(base_seed)
    runs = []
    for i in range(n_runs):
        h = {
            "seed": base_seed + i + 1,
            "epochs": rng.choice([10, 15, 20, 30]),
            "batch_size": rng.choice([256, 512, 1024, 2048]),
            "lr": sample_loguniform(rng, 1e-4, 5e-3),
            "weight_decay": sample_loguniform(rng, 1e-7, 1e-3),
            "hidden_dim": rng.choice([32, 64, 128, 256]),
            "n_layers": rng.choice([1, 2, 3, 4, 5]),
            "dropout": rng.choice([0.0, 0.05, 0.1, 0.2]),
            "grad_clip": rng.choice([0.0, 0.5, 1.0, 2.0]),
        }
        runs.append(h)
    return runs


# -----------------------
# CLI
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True, help="synthetic_functions.npz")
    ap.add_argument("--runs", type=int, default=25)
    ap.add_argument("--base_seed", type=int, default=123)
    ap.add_argument("--out", type=str, default="results.jsonl")
    args = ap.parse_args()

    runs = random_sweep(args.runs, args.base_seed)

    results = []
    with open(args.out, "w", encoding="utf-8") as f:
        for i, h in enumerate(runs, start=1):
            res = train_one_run(args.npz, h)
            results.append(res)
            f.write(json.dumps(res) + "\n")
            f.flush()
            print(f"[{i}/{len(runs)}] best_rmse={res['best_rmse']:.6f} (epoch {res['best_epoch']})")

    # Show top 5
    results.sort(key=lambda r: r["best_rmse"])
    print("\nTop 5 runs by RMSE:")
    for r in results[:5]:
        hp = r["hparams"]
        print(
            f"rmse={r['best_rmse']:.6f}  lr={hp['lr']:.2e}  bs={hp['batch_size']}  "
            f"hid={hp['hidden_dim']}  layers={hp['n_layers']}  drop={hp['dropout']}"
        )

    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()