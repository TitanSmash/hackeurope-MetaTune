#!/usr/bin/env python3
"""
Train a model to estimate generating function parameters from noisy curves.

Inputs:
- NPZ from your generator: contains X_train, y_train, X_test, y_test
- JSONL metadata: contains family + params per sample (train/test)

Model:
- Takes (x,y) sequences -> predicts:
  (1) family logits
  (2) param vector (fixed size across families)

Loss:
- family CE
- params MSE masked to the true family's parameter slots

Includes:
- train_one_run(hparams): train + return metrics dict
- sweep_random / sweep_grid: hyperparam sweep -> results.jsonl

Requirements:
  pip install torch numpy
"""

from __future__ import annotations
import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------
# Metadata parsing + param packing
# ---------------------------

# Must match your generator families/params.
# IMPORTANT: Keep a stable ordering.
FAMILY_PARAM_SPECS: Dict[str, List[str]] = {
    "sine":     ["a", "b", "c", "d"],
    "cosine":   ["a", "b", "c", "d"],
    "exp":      ["a", "b", "c"],
    "log":      ["a", "b", "c"],
    "poly2":    ["a", "b", "c"],
    "poly3":    ["a", "b", "c", "d"],
    "rational": ["a", "b", "c", "d"],
    "abs":      ["a", "b"],
    "step":     ["a", "b", "t"],
    "sawtooth": ["a", "b", "c"],
}

FAMILIES = list(FAMILY_PARAM_SPECS.keys())
FAMILY_TO_ID = {f: i for i, f in enumerate(FAMILIES)}

# Build a global parameter index space so we can regress one fixed-size vector.
# Each family uses a subset of slots.
ALL_PARAM_NAMES: List[str] = []
for fam in FAMILIES:
    for p in FAMILY_PARAM_SPECS[fam]:
        name = f"{fam}.{p}"
        if name not in ALL_PARAM_NAMES:
            ALL_PARAM_NAMES.append(name)
PARAM_TO_IDX = {name: i for i, name in enumerate(ALL_PARAM_NAMES)}
P = len(ALL_PARAM_NAMES)  # total param dims


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def pack_params(family: str, params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      param_vec: (P,) float32
      mask_vec:  (P,) float32 (1 for active params of this family)
    """
    vec = np.zeros((P,), dtype=np.float32)
    mask = np.zeros((P,), dtype=np.float32)
    for p in FAMILY_PARAM_SPECS[family]:
        key = f"{family}.{p}"
        idx = PARAM_TO_IDX[key]
        vec[idx] = float(params[p])
        mask[idx] = 1.0
    return vec, mask


# ---------------------------
# Dataset
# ---------------------------

class CurveParamDataset(Dataset):
    def __init__(self, npz_path: str, meta_jsonl_path: str, split: str):
        data = np.load(npz_path, allow_pickle=True)
        if split == "train":
            X = data["X_train"]
            y = data["y_train"]
        elif split == "test":
            X = data["X_test"]
            y = data["y_test"]
        else:
            raise ValueError("split must be train or test")

        meta = load_jsonl(meta_jsonl_path)
        assert len(meta) == X.shape[0], f"Meta len {len(meta)} != X {X.shape[0]}"

        # combine x and y as 2-channel sequence: (L, 2)
        # Your generator stores (N, L, 1) for each.
        xy = np.concatenate([X, y], axis=-1).astype(np.float32)  # (N, L, 2)

        fam_ids = np.zeros((X.shape[0],), dtype=np.int64)
        param_vecs = np.zeros((X.shape[0], P), dtype=np.float32)
        param_masks = np.zeros((X.shape[0], P), dtype=np.float32)

        for i, m in enumerate(meta):
            fam = m["family"]
            fam_ids[i] = FAMILY_TO_ID[fam]
            vec, mask = pack_params(fam, m["params"])
            param_vecs[i] = vec
            param_masks[i] = mask

        self.xy = xy
        self.fam_ids = fam_ids
        self.param_vecs = param_vecs
        self.param_masks = param_masks

    def __len__(self) -> int:
        return self.xy.shape[0]

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.xy[idx]),          # (L, 2)
            torch.tensor(self.fam_ids[idx]),         # ()
            torch.from_numpy(self.param_vecs[idx]),  # (P,)
            torch.from_numpy(self.param_masks[idx]), # (P,)
        )


# ---------------------------
# Model: 1D CNN encoder + heads
# ---------------------------

class Conv1DEncoder(nn.Module):
    def __init__(self, in_ch: int, hidden: int, n_layers: int, dropout: float):
        super().__init__()
        layers = []
        ch = in_ch
        for i in range(n_layers):
            layers += [
                nn.Conv1d(ch, hidden, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            ch = hidden
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) -> (B, C, L)
        x = x.transpose(1, 2)
        h = self.net(x)          # (B, hidden, L)
        h = self.pool(h).squeeze(-1)  # (B, hidden)
        return h


class ParamEstimator(nn.Module):
    def __init__(self, in_ch: int, hidden: int, n_layers: int, dropout: float, n_families: int, p_dim: int):
        super().__init__()
        self.enc = Conv1DEncoder(in_ch, hidden, n_layers, dropout)
        self.family_head = nn.Linear(hidden, n_families)
        self.param_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, p_dim),
        )

    def forward(self, xy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(xy)
        fam_logits = self.family_head(h)
        params = self.param_head(h)
        return fam_logits, params


# ---------------------------
# Train/eval utilities
# ---------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    mse_sum = 0.0
    mse_count = 0.0
    correct = 0
    total = 0
    ce_sum = 0.0

    for xy, fam_id, p_true, p_mask in loader:
        xy = xy.to(device)
        fam_id = fam_id.to(device)
        p_true = p_true.to(device)
        p_mask = p_mask.to(device)

        fam_logits, p_pred = model(xy)

        ce_sum += float(ce(fam_logits, fam_id).item())
        pred_cls = fam_logits.argmax(dim=1)
        correct += int((pred_cls == fam_id).sum().item())
        total += int(fam_id.numel())

        # masked MSE over active params only
        diff2 = (p_pred - p_true) ** 2
        diff2 = diff2 * p_mask
        mse_sum += float(diff2.sum().item())
        mse_count += float(p_mask.sum().item())

    return {
        "family_acc": correct / max(1, total),
        "ce": ce_sum / max(1, total),
        "masked_mse": mse_sum / max(1e-9, mse_count),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------
# One run (given hyperparams)
# ---------------------------

def train_one_run(
    npz_path: str,
    meta_train_path: str,
    meta_test_path: str,
    hparams: Dict[str, Any],
) -> Dict[str, Any]:
    """
    hparams expected (with defaults):
      seed, epochs, batch_size, lr, weight_decay,
      hidden, n_layers, dropout,
      alpha_ce (family loss weight), alpha_mse (param loss weight),
      grad_clip
    """
    seed = int(hparams.get("seed", 42))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_train = CurveParamDataset(npz_path, meta_train_path, split="train")
    ds_test  = CurveParamDataset(npz_path, meta_test_path,  split="test")

    batch_size = int(hparams.get("batch_size", 128))
    num_workers = int(hparams.get("num_workers", 0))

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    dl_test  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    model = ParamEstimator(
        in_ch=2,
        hidden=int(hparams.get("hidden", 128)),
        n_layers=int(hparams.get("n_layers", 4)),
        dropout=float(hparams.get("dropout", 0.1)),
        n_families=len(FAMILIES),
        p_dim=P,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(hparams.get("lr", 3e-4)),
        weight_decay=float(hparams.get("weight_decay", 1e-4)),
    )

    ce_loss = nn.CrossEntropyLoss()
    alpha_ce = float(hparams.get("alpha_ce", 1.0))
    alpha_mse = float(hparams.get("alpha_mse", 1.0))
    grad_clip = float(hparams.get("grad_clip", 1.0))
    epochs = int(hparams.get("epochs", 20))

    best = {"score": -1e9, "epoch": -1, "metrics": None}

    for ep in range(1, epochs + 1):
        model.train()
        for xy, fam_id, p_true, p_mask in dl_train:
            xy = xy.to(device)
            fam_id = fam_id.to(device)
            p_true = p_true.to(device)
            p_mask = p_mask.to(device)

            fam_logits, p_pred = model(xy)

            loss_ce = ce_loss(fam_logits, fam_id)

            diff2 = (p_pred - p_true) ** 2
            # average over active params
            mse = (diff2 * p_mask).sum() / (p_mask.sum() + 1e-9)

            loss = alpha_ce * loss_ce + alpha_mse * mse

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        metrics = evaluate(model, dl_test, device=device)

        # One scalar to rank sweeps; tune if you care more about params vs family.
        score = metrics["family_acc"] - 0.25 * metrics["masked_mse"]

        if score > best["score"]:
            best = {"score": score, "epoch": ep, "metrics": metrics}

    out = {
        "hparams": dict(hparams),
        "best_epoch": best["epoch"],
        "best_score": best["score"],
        **{f"best_{k}": v for k, v in best["metrics"].items()},
    }
    return out


# ---------------------------
# Sweep helpers
# ---------------------------

def sweep_grid(space: Dict[str, List[Any]]):
    keys = list(space.keys())
    def rec(i: int, cur: Dict[str, Any]):
        if i == len(keys):
            yield dict(cur)
            return
        k = keys[i]
        for v in space[k]:
            cur[k] = v
            yield from rec(i + 1, cur)
    yield from rec(0, {})


def sweep_random(space: Dict[str, Tuple[Any, Any]], n: int, seed: int = 0):
    rng = random.Random(seed)
    for _ in range(n):
        hp = {}
        for k, (lo, hi) in space.items():
            # float range vs int range
            if isinstance(lo, int) and isinstance(hi, int):
                hp[k] = rng.randint(lo, hi)
            else:
                # log-uniform for lr/weight_decay if requested by key name
                if k in {"lr", "weight_decay"}:
                    lo_f, hi_f = float(lo), float(hi)
                    hp[k] = 10 ** rng.uniform(math.log10(lo_f), math.log10(hi_f))
                else:
                    hp[k] = rng.uniform(float(lo), float(hi))
        yield hp


# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True, help="e.g. synthetic_functions.npz")
    ap.add_argument("--meta_train", type=str, required=True, help="e.g. synthetic_functions_meta_train.jsonl")
    ap.add_argument("--meta_test", type=str, required=True, help="e.g. synthetic_functions_meta_test.jsonl")
    ap.add_argument("--mode", type=str, default="random", choices=["random", "grid"])
    ap.add_argument("--runs", type=int, default=20, help="number of sweep runs (random) or total combos (grid)")
    ap.add_argument("--out", type=str, default="sweep_results.jsonl")
    ap.add_argument("--base_seed", type=int, default=123)
    args = ap.parse_args()

    results = []

    if args.mode == "random":
        # Random search space (edit freely)
        space = {
            "lr": (1e-4, 3e-3),
            "batch_size": (64, 256),
            "hidden": (64, 256),
            "n_layers": (2, 6),
            "dropout": (0.0, 0.3),
            "weight_decay": (1e-6, 1e-3),
            "epochs": (10, 30),
            "alpha_mse": (0.5, 2.0),
            "alpha_ce": (0.5, 2.0),
            "grad_clip": (0.5, 2.0),
        }
        runs = list(sweep_random(space, n=args.runs, seed=args.base_seed))

    else:
        # Small grid (edit freely)
        grid = {
            "lr": [1e-4, 3e-4, 1e-3],
            "batch_size": [64, 128],
            "hidden": [96, 128, 192],
            "n_layers": [3, 4],
            "dropout": [0.0, 0.1],
            "weight_decay": [1e-5, 1e-4],
            "epochs": [20],
            "alpha_ce": [1.0],
            "alpha_mse": [1.0],
            "grad_clip": [1.0],
        }
        runs = list(sweep_grid(grid))[: args.runs]

    # Run sweep
    with open(args.out, "w", encoding="utf-8") as f:
        for i, hp in enumerate(runs, start=1):
            hp = dict(hp)
            hp["seed"] = args.base_seed + i
            res = train_one_run(args.npz, args.meta_train, args.meta_test, hp)
            results.append(res)
            f.write(json.dumps(res) + "\n")
            f.flush()
            print(f"[{i}/{len(runs)}] score={res['best_score']:.4f} acc={res['best_family_acc']:.3f} mse={res['best_masked_mse']:.5f}")

    # Print top 5
    results.sort(key=lambda r: r["best_score"], reverse=True)
    print("\nTop 5 runs:")
    for r in results[:5]:
        hp = r["hparams"]
        print(
            f"score={r['best_score']:.4f} acc={r['best_family_acc']:.3f} mse={r['best_masked_mse']:.5f} "
            f"lr={hp.get('lr'):.2e} bs={hp.get('batch_size')} hid={hp.get('hidden')} layers={hp.get('n_layers')} drop={hp.get('dropout'):.2f}"
        )


if __name__ == "__main__":
    main()