

#!/usr/bin/env python3
"""
Synthetic function dataset generator (train/test) with noise.

Generates samples by:
- choosing a function family (sin, exp, poly, log, rational, etc.)
- sampling parameters for that family
- sampling x-values (grid or random)
- computing y = f(x; params)
- adding noise (Gaussian + optional outliers + optional heteroscedasticity)
- optionally normalizing x and/or y per-sample

Outputs:
- X: shape (N, L, 1)   (sequence of x values)
- y: shape (N, L, 1)   (sequence of noisy y values)
- meta: list of dicts describing the function + parameters (saved as JSON lines)
"""

from __future__ import annotations

import json
import math
import argparse
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Tuple, Any, List

import numpy as np


# ----------------------------
# Function families
# ----------------------------

def f_sine(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    # a*sin(b*x + c) + d
    return a * np.sin(b * x + c) + d

def f_cosine(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    return a * np.cos(b * x + c) + d

def f_exp(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    # a*exp(b*x) + c
    return a * np.exp(b * x) + c

def f_log(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    # a*log(b*x) + c  (requires b*x > 0)
    return a * np.log(b * x) + c

def f_poly2(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    # ax^2 + bx + c
    return a * x**2 + b * x + c

def f_poly3(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    return a * x**3 + b * x**2 + c * x + d

def f_rational(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    # (a*x + b) / (c*x + d)  (avoid denom near 0)
    return (a * x + b) / (c * x + d)

def f_abs(x: np.ndarray, a: float, b: float) -> np.ndarray:
    # a*|x| + b
    return a * np.abs(x) + b

def f_step(x: np.ndarray, a: float, b: float, t: float) -> np.ndarray:
    # a if x < t else b
    return np.where(x < t, a, b)

def f_sawtooth(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    # a * sawtooth(b*x + c) in [-a, a]
    # sawtooth via fractional part
    z = (b * x + c) / (2 * np.pi)
    frac = z - np.floor(z)
    return a * (2 * frac - 1)

FAMILIES: Dict[str, Tuple[Callable[..., np.ndarray], Dict[str, Tuple[float, float]]]] = {
    # name: (function, parameter ranges)
    "sine":     (f_sine,     {"a": (0.5, 2.0), "b": (0.5, 3.0), "c": (0.0, 2*np.pi), "d": (-1.0, 1.0)}),
    "cosine":   (f_cosine,   {"a": (0.5, 2.0), "b": (0.5, 3.0), "c": (0.0, 2*np.pi), "d": (-1.0, 1.0)}),
    "exp":      (f_exp,      {"a": (0.3, 2.0), "b": (-1.2, 1.2), "c": (-1.0, 1.0)}),
    "poly2":    (f_poly2,    {"a": (-1.0, 1.0), "b": (-2.0, 2.0), "c": (-1.0, 1.0)}),
    "poly3":    (f_poly3,    {"a": (-0.5, 0.5), "b": (-1.0, 1.0), "c": (-2.0, 2.0), "d": (-1.0, 1.0)}),
    "rational": (f_rational, {"a": (-2.0, 2.0), "b": (-2.0, 2.0), "c": (-1.5, 1.5), "d": (0.5, 2.0)}),
    "abs":      (f_abs,      {"a": (0.2, 3.0), "b": (-1.0, 1.0)}),
    "step":     (f_step,     {"a": (-1.0, 1.0), "b": (-1.0, 1.0), "t": (-0.5, 0.5)}),
    "sawtooth": (f_sawtooth, {"a": (0.3, 2.0), "b": (0.5, 3.0), "c": (0.0, 2*np.pi)}),
    # log is trickier because domain constraints; enabled below via special handling
}


# ----------------------------
# Config / metadata
# ----------------------------

@dataclass
class SampleMeta:
    family: str
    params: Dict[str, float]
    x_min: float
    x_max: float
    noise_sigma: float
    outlier_prob: float
    outlier_scale: float
    hetero: bool


# ----------------------------
# Sampling helpers
# ----------------------------

def uniform_params(rng: np.random.Generator, ranges: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    return {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in ranges.items()}

def sample_x(
    rng: np.random.Generator,
    length: int,
    x_min: float,
    x_max: float,
    mode: str = "grid",
) -> np.ndarray:
    if mode == "grid":
        return np.linspace(x_min, x_max, length, dtype=np.float32)
    if mode == "random":
        x = rng.uniform(x_min, x_max, size=length).astype(np.float32)
        return np.sort(x)
    raise ValueError(f"Unknown x sampling mode: {mode}")

def add_noise(
    rng: np.random.Generator,
    y: np.ndarray,
    sigma: float,
    outlier_prob: float = 0.0,
    outlier_scale: float = 5.0,
    hetero: bool = False,
) -> np.ndarray:
    # heteroscedastic noise: sigma grows with |y|
    if hetero:
        local_sigma = sigma * (0.25 + 0.75 * (np.abs(y) / (np.max(np.abs(y)) + 1e-6)))
    else:
        local_sigma = sigma

    eps = rng.normal(0.0, local_sigma, size=y.shape).astype(np.float32)
    y_noisy = y + eps

    if outlier_prob > 0:
        mask = rng.random(size=y.shape) < outlier_prob
        outliers = rng.normal(0.0, sigma * outlier_scale, size=y.shape).astype(np.float32)
        y_noisy = np.where(mask, y_noisy + outliers, y_noisy)

    return y_noisy

def safe_eval_family(
    family: str,
    func: Callable[..., np.ndarray],
    x: np.ndarray,
    params: Dict[str, float],
) -> np.ndarray:
    # extra safety for rational/log or explosive exp
    if family == "rational":
        denom = params["c"] * x + params["d"]
        # avoid near-zero denom by shifting d if needed
        if np.any(np.abs(denom) < 0.15):
            params = dict(params)
            params["d"] = float(params["d"] + np.sign(params["d"]) * 0.5)
    if family == "exp":
        # clip x to avoid overflow for exp(bx)
        x = np.clip(x, -5.0, 5.0)
    return func(x, **params).astype(np.float32)

def gen_log_sample(
    rng: np.random.Generator,
    x: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    # Ensure domain: b*x > 0. Pick b>0 and x>0 or b<0 and x<0.
    sign = 1.0 if rng.random() < 0.5 else -1.0
    # make x strictly away from 0
    x_safe = np.where(sign > 0, np.clip(x, 0.05, None), np.clip(x, None, -0.05))
    a = float(rng.uniform(0.3, 2.0))
    b = float(sign * rng.uniform(0.5, 2.0))
    c = float(rng.uniform(-1.0, 1.0))
    y = f_log(x_safe, a=a, b=b, c=c).astype(np.float32)
    return y, {"a": a, "b": b, "c": c}


# ----------------------------
# Dataset generation
# ----------------------------

def generate_dataset(
    n_samples: int,
    length: int,
    rng: np.random.Generator,
    families: List[str],
    x_range: Tuple[float, float],
    x_mode: str,
    noise_sigma_range: Tuple[float, float],
    outlier_prob: float,
    outlier_scale: float,
    hetero_prob: float,
    normalize_xy: bool,
) -> Tuple[np.ndarray, np.ndarray, List[SampleMeta]]:
    X = np.zeros((n_samples, length, 1), dtype=np.float32)
    Y = np.zeros((n_samples, length, 1), dtype=np.float32)
    metas: List[SampleMeta] = []

    x_min, x_max = x_range

    for i in range(n_samples):
        family = rng.choice(families)
        x = sample_x(rng, length, x_min, x_max, mode=x_mode)

        sigma = float(rng.uniform(noise_sigma_range[0], noise_sigma_range[1]))
        hetero = bool(rng.random() < hetero_prob)

        if family == "log":
            y_clean, params = gen_log_sample(rng, x)
        else:
            func, pr = FAMILIES[family]
            params = uniform_params(rng, pr)
            y_clean = safe_eval_family(family, func, x, params)

        y_noisy = add_noise(
            rng=rng,
            y=y_clean,
            sigma=sigma,
            outlier_prob=outlier_prob,
            outlier_scale=outlier_scale,
            hetero=hetero,
        )

        # Optional per-sample normalization (often good for sequence models)
        x_out = x.astype(np.float32)
        y_out = y_noisy.astype(np.float32)

        if normalize_xy:
            x_out = (x_out - x_out.mean()) / (x_out.std() + 1e-6)
            y_out = (y_out - y_out.mean()) / (y_out.std() + 1e-6)

        X[i, :, 0] = x_out
        Y[i, :, 0] = y_out

        metas.append(
            SampleMeta(
                family=str(family),
                params={k: float(v) for k, v in params.items()},
                x_min=float(x_min),
                x_max=float(x_max),
                noise_sigma=float(sigma),
                outlier_prob=float(outlier_prob),
                outlier_scale=float(outlier_scale),
                hetero=bool(hetero),
            )
        )

    return X, Y, metas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=400)
    ap.add_argument("--length", type=int, default=128)
    ap.add_argument("--x_min", type=float, default=-2.0)
    ap.add_argument("--x_max", type=float, default=2.0)
    ap.add_argument("--x_mode", type=str, default="grid", choices=["grid", "random"])
    ap.add_argument("--noise_min", type=float, default=0.02)
    ap.add_argument("--noise_max", type=float, default=0.15)
    ap.add_argument("--outlier_prob", type=float, default=0.02)
    ap.add_argument("--outlier_scale", type=float, default=6.0)
    ap.add_argument("--hetero_prob", type=float, default=0.3)
    ap.add_argument("--normalize_xy", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="synthetic_functions")
    ap.add_argument(
        "--families",
        type=str,
        default="sine,cosine,exp,log,poly2,poly3,rational,abs,step,sawtooth",
        help="comma-separated list",
    )
    ap.add_argument("--write_csv", action="store_true", help="also write flattened CSVs (big)")

    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    families = [s.strip() for s in args.families.split(",") if s.strip()]
    # allow log even though it's handled separately
    for f in families:
        if f != "log" and f not in FAMILIES:
            raise ValueError(f"Unknown family: {f}")

    x_range = (args.x_min, args.x_max)
    noise_range = (args.noise_min, args.noise_max)

    Xtr, Ytr, Mtr = generate_dataset(
        n_samples=args.n_train,
        length=args.length,
        rng=rng,
        families=families,
        x_range=x_range,
        x_mode=args.x_mode,
        noise_sigma_range=noise_range,
        outlier_prob=args.outlier_prob,
        outlier_scale=args.outlier_scale,
        hetero_prob=args.hetero_prob,
        normalize_xy=args.normalize_xy,
    )

    # new RNG stream for test for reproducibility separation
    rng_test = np.random.default_rng(args.seed + 1)
    Xte, Yte, Mte = generate_dataset(
        n_samples=args.n_test,
        length=args.length,
        rng=rng_test,
        families=families,
        x_range=x_range,
        x_mode=args.x_mode,
        noise_sigma_range=noise_range,
        outlier_prob=args.outlier_prob,
        outlier_scale=args.outlier_scale,
        hetero_prob=args.hetero_prob,
        normalize_xy=args.normalize_xy,
    )

    np.savez_compressed(
        f"{args.out}.npz",
        X_train=Xtr,
        y_train=Ytr,
        X_test=Xte,
        y_test=Yte,
        families=np.array(families, dtype=object),
    )

    # Save metadata as jsonl for easy inspection
    with open(f"{args.out}_meta_train.jsonl", "w", encoding="utf-8") as f:
        for m in Mtr:
            f.write(json.dumps(asdict(m)) + "\n")

    with open(f"{args.out}_meta_test.jsonl", "w", encoding="utf-8") as f:
        for m in Mte:
            f.write(json.dumps(asdict(m)) + "\n")

    if args.write_csv:
        # Flatten each sample to one row: x0..xL-1, y0..yL-1, family
        def to_csv(path: str, X: np.ndarray, Y: np.ndarray, M: List[SampleMeta]) -> None:
            header = [f"x{i}" for i in range(X.shape[1])] + [f"y{i}" for i in range(Y.shape[1])] + ["family"]
            with open(path, "w", encoding="utf-8") as f:
                f.write(",".join(header) + "\n")
                for i in range(X.shape[0]):
                    row = list(X[i, :, 0].astype(float)) + list(Y[i, :, 0].astype(float)) + [M[i].family]
                    f.write(",".join(map(str, row)) + "\n")

        to_csv(f"{args.out}_train.csv", Xtr, Ytr, Mtr)
        to_csv(f"{args.out}_test.csv", Xte, Yte, Mte)

    print(f"Saved: {args.out}.npz")
    print(f"Saved: {args.out}_meta_train.jsonl")
    print(f"Saved: {args.out}_meta_test.jsonl")
    if args.write_csv:
        print(f"Saved: {args.out}_train.csv and {args.out}_test.csv")


if __name__ == "__main__":
    main()