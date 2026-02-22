from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pandas as pd
except ImportError:  # pragma: no cover - depends on local env.
    pd = None


DEFAULT_HP_SPACE: Dict[str, Dict[str, Any]] = {
    "lora_r": {"type": "categorical", "values": [4, 8, 16, 32, 64]},
    "learning_rate": {"type": "float", "low": 5e-5, "high": 3e-3, "log": True},
    "lora_alpha": {"type": "categorical", "values": [8, 16, 32, 64, 128]},
    "lora_dropout": {"type": "float", "low": 0.0, "high": 0.5, "log": False},
    "batch_size": {"type": "categorical", "values": [4, 8, 16, 32]},
}


def _dominates(left_row: Sequence[float], right_row: Sequence[float]) -> bool:
    return all(l <= r for l, r in zip(left_row, right_row)) and any(
        l < r for l, r in zip(left_row, right_row)
    )


def pareto_front_df(df, metric_cols: Sequence[str] = ("val_loss", "gpu_hours")):
    if pd is None:
        raise RuntimeError("pandas is required for pareto_front_df.")
    if df.empty:
        return df.copy()

    filtered = df.dropna(subset=list(metric_cols)).copy()
    if filtered.empty:
        return filtered

    dominated_indices = set()
    rows = list(filtered[metric_cols].itertuples(index=True, name=None))
    for idx_i, *metrics_i in rows:
        if idx_i in dominated_indices:
            continue
        for idx_j, *metrics_j in rows:
            if idx_i == idx_j:
                continue
            if _dominates(metrics_j, metrics_i):
                dominated_indices.add(idx_i)
                break
    return filtered.drop(index=list(dominated_indices))


def _sample_value(spec: Dict[str, Any], rng: random.Random) -> Any:
    kind = spec.get("type")
    if kind == "categorical":
        return rng.choice(spec["values"])
    if kind == "float":
        low = float(spec["low"])
        high = float(spec["high"])
        if spec.get("log"):
            return math.exp(rng.uniform(math.log(low), math.log(high)))
        return rng.uniform(low, high)
    if kind == "int":
        low = int(spec["low"])
        high = int(spec["high"])
        return rng.randint(low, high)
    raise ValueError(f"Unsupported HP spec type: {kind}")


def _stratified_float_values(spec: Dict[str, Any], n: int, rng: random.Random) -> List[float]:
    buckets = list(range(n))
    rng.shuffle(buckets)
    values = []
    low = float(spec["low"])
    high = float(spec["high"])
    for idx in buckets:
        u = (idx + rng.random()) / n
        if spec.get("log"):
            value = math.exp(math.log(low) + u * (math.log(high) - math.log(low)))
        else:
            value = low + u * (high - low)
        values.append(value)
    return values


def generate_coarse_grid(
    hp_space: Dict[str, Dict[str, Any]],
    n_trials: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    names = list(hp_space.keys())
    stratified_values: Dict[str, List[Any]] = {}
    for name in names:
        spec = hp_space[name]
        kind = spec.get("type")
        if kind == "float":
            stratified_values[name] = _stratified_float_values(spec, n_trials, rng)
        elif kind in {"categorical", "int"}:
            values = []
            choices = spec["values"] if kind == "categorical" else list(
                range(int(spec["low"]), int(spec["high"]) + 1)
            )
            offset = rng.randint(0, max(1, len(choices) - 1))
            for idx in range(n_trials):
                values.append(choices[(idx + offset) % len(choices)])
            rng.shuffle(values)
            stratified_values[name] = values
        else:
            raise ValueError(f"Unsupported HP type in coarse grid: {kind}")

    configs: List[Dict[str, Any]] = []
    for i in range(n_trials):
        config = {name: stratified_values[name][i] for name in names}
        configs.append(config)
    return configs


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class MetaBayesianOptimizer:
    def __init__(self, history_df):
        """
        Initializes the Knowledge Base.
        history_df must contain:
        - Metafeatures: ['ttr_score', 'hapax_ratio', 'zipf_alpha']
        - Hyperparameters: ['lora_r', 'learning_rate', 'lora_alpha', 'lora_dropout', 'batch_size']
        - Target Metric: ['val_loss']
        """
        self.history_df = history_df if history_df is not None else self._empty_df()
        self.metafeature_cols = ["ttr_score", "hapax_ratio", "zipf_alpha"]
        self.hp_cols = ["lora_r", "learning_rate", "lora_alpha", "lora_dropout", "batch_size"]

    @staticmethod
    def _empty_df():
        if pd is None:
            raise RuntimeError("pandas is required for MetaBayesianOptimizer.")
        return pd.DataFrame()

    def update_history(self, history_df) -> None:
        self.history_df = history_df

    def _prepare_history(self):
        if pd is None:
            raise RuntimeError("pandas is required for MetaBayesianOptimizer.")
        if self.history_df is None or self.history_df.empty:
            return self._empty_df()
        frame = self.history_df.copy()
        for col in ("val_loss", "gpu_hours"):
            if col in frame.columns:
                frame[col] = frame[col].apply(_safe_float)
        return frame.dropna(subset=["val_loss", "gpu_hours"], how="any")

    def _nearest_history(self, target_metafeatures: Dict[str, float], top_k: int = 200):
        frame = self._prepare_history()
        if frame.empty:
            return frame
        for col in self.metafeature_cols:
            if col not in frame.columns:
                frame[col] = 0.0
        for col in self.metafeature_cols:
            frame[col] = frame[col].apply(_safe_float)
        metric = []
        for _, row in frame.iterrows():
            sq_sum = 0.0
            for col in self.metafeature_cols:
                target_value = _safe_float(target_metafeatures.get(col, 0.0))
                value = _safe_float(row.get(col, 0.0))
                sq_sum += (value - target_value) ** 2
            metric.append(math.sqrt(sq_sum))
        frame["_distance"] = metric
        frame = frame.sort_values("_distance", ascending=True)
        return frame.head(top_k)

    def _sample_from_good_set(
        self,
        good_set,
        hp_space: Dict[str, Dict[str, Any]],
        rng: random.Random,
    ) -> Dict[str, Any]:
        config: Dict[str, Any] = {}
        for hp_name, spec in hp_space.items():
            if good_set.empty or hp_name not in good_set.columns:
                config[hp_name] = _sample_value(spec, rng)
                continue
            kind = spec.get("type")
            column_values = [value for value in good_set[hp_name].tolist() if value is not None]
            if not column_values:
                config[hp_name] = _sample_value(spec, rng)
                continue
            if kind == "categorical":
                pool = list(spec["values"])
                counts = {value: 1 for value in pool}
                for value in column_values:
                    if value in counts:
                        counts[value] += 1
                total = sum(counts.values())
                threshold = rng.uniform(0.0, total)
                cumulative = 0.0
                picked = pool[0]
                for value in pool:
                    cumulative += counts[value]
                    if cumulative >= threshold:
                        picked = value
                        break
                config[hp_name] = picked
            elif kind == "float":
                low = float(spec["low"])
                high = float(spec["high"])
                sampled = _safe_float(rng.choice(column_values))
                if spec.get("log"):
                    sampled_log = math.log(max(low, sampled))
                    sigma = 0.2 * (math.log(high) - math.log(low))
                    candidate = math.exp(rng.gauss(sampled_log, sigma))
                else:
                    sigma = 0.15 * (high - low)
                    candidate = rng.gauss(sampled, sigma)
                config[hp_name] = min(high, max(low, candidate))
            else:
                config[hp_name] = _sample_value(spec, rng)
        return config

    def suggest(
        self,
        target_metafeatures: Dict[str, float],
        n_suggestions: int,
        hp_space: Optional[Dict[str, Dict[str, Any]]] = None,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        if n_suggestions <= 0:
            return []
        hp_space = hp_space or DEFAULT_HP_SPACE
        rng = random.Random(seed)

        history = self._nearest_history(target_metafeatures=target_metafeatures)
        if history.empty:
            return [generate_coarse_grid(hp_space, 1, seed=rng.randint(1, 10_000))[0] for _ in range(n_suggestions)]

        pareto = pareto_front_df(history, metric_cols=("val_loss", "gpu_hours"))
        if pareto.empty:
            pareto = history.nsmallest(min(len(history), 20), columns=["val_loss"])
        good_set = pareto
        results: List[Dict[str, Any]] = []
        seen = set()
        max_attempts = n_suggestions * 30
        attempts = 0
        while len(results) < n_suggestions and attempts < max_attempts:
            attempts += 1
            candidate = self._sample_from_good_set(good_set, hp_space=hp_space, rng=rng)
            signature = tuple((key, candidate[key]) for key in sorted(candidate.keys()))
            if signature in seen:
                continue
            seen.add(signature)
            results.append(candidate)

        while len(results) < n_suggestions:
            candidate = {k: _sample_value(v, rng) for k, v in hp_space.items()}
            results.append(candidate)
        return results
