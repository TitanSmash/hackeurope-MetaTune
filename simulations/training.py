from __future__ import annotations

import json
import math
import os
import random
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from simulations.io_utils import append_jsonl, ensure_dir, iter_jsonl, utc_now_iso


@dataclass
class TrialConfig:
    run_id: str
    dataset_id: str
    dataset_slug: str
    stage: str
    seed: int
    token_budget: int
    seq_len: int
    hp: Dict[str, Any]
    paths: Dict[str, str]
    device: str
    slurm_meta: Dict[str, Any]
    retry_count: int = 0


@dataclass
class TrialResult:
    status: str
    val_loss: float
    gpu_hours: float
    elapsed_sec: float
    peak_mem_mb: float
    train_curve_path: str
    error_class: str = ""
    error_message: str = ""
    best_val_loss: float = 0.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _estimate_peak_mem_mb(hp: Dict[str, Any], seq_len: int) -> float:
    lora_r = _safe_float(hp.get("lora_r"), 16)
    lora_alpha = _safe_float(hp.get("lora_alpha"), 32)
    batch_size = _safe_float(hp.get("batch_size"), 8)
    grad_accum = _safe_float(hp.get("grad_accum"), 2)
    dropout = _safe_float(hp.get("dropout"), 0.1)
    base = 4500.0
    mem = (
        base
        + 300.0 * batch_size
        + 120.0 * grad_accum
        + 15.0 * lora_r
        + 2.0 * lora_alpha
        + 1.8 * seq_len
        + 500.0 * dropout
    )
    return float(mem)


def _compute_simulated_metrics(
    hp: Dict[str, Any],
    dataset_meta: Dict[str, Any],
    token_budget: int,
    seq_len: int,
    seed: int,
    gpu_count: int = 1,
) -> Dict[str, float]:
    rng = random.Random(seed)
    ttr = _safe_float(dataset_meta.get("ttr_score"), 0.1)
    hapax = _safe_float(dataset_meta.get("hapax_ratio"), 0.2)
    zipf_alpha = _safe_float(dataset_meta.get("zipf_alpha"), 1.1)

    lora_r = _safe_float(hp.get("lora_r"), 16)
    learning_rate = _safe_float(hp.get("learning_rate"), 2e-4)
    dropout = _safe_float(hp.get("dropout"), 0.05)
    lora_dropout = _safe_float(hp.get("lora_dropout"), dropout)
    batch_size = _safe_float(hp.get("batch_size"), 8)
    grad_accum = _safe_float(hp.get("grad_accum"), 2)
    warmup_ratio = _safe_float(hp.get("warmup_ratio"), 0.03)
    weight_decay = _safe_float(hp.get("weight_decay"), 0.02)

    # Synthetic but deterministic quality landscape with dataset-conditioned difficulty.
    target_lr = 2e-4
    lr_penalty = abs(math.log(max(1e-8, learning_rate)) - math.log(target_lr))
    rank_penalty = abs(math.log2(max(1.0, lora_r)) - math.log2(16.0)) * 0.06
    dropout_penalty = abs(dropout - 0.05) * 0.8 + max(0.0, lora_dropout - 0.12) * 0.7
    regularization_penalty = abs(weight_decay - 0.02) * 1.6
    warmup_penalty = abs(warmup_ratio - 0.04) * 0.8
    dataset_difficulty = 1.2 + 0.8 * ttr + 0.6 * hapax + 0.3 * max(0.0, zipf_alpha - 1.0)
    stochastic = rng.gauss(0.0, 0.025)

    final_val_loss = (
        1.0
        + dataset_difficulty
        + 0.18 * lr_penalty
        + rank_penalty
        + dropout_penalty
        + regularization_penalty
        + warmup_penalty
        + stochastic
    )
    final_val_loss = max(0.8, final_val_loss)
    best_val_loss = final_val_loss - abs(rng.gauss(0.0, 0.02))

    # Throughput model.
    throughput_tps = (
        2500.0
        * (batch_size * max(1.0, grad_accum)) / 8.0
        * (1024.0 / max(256.0, float(seq_len))) ** 0.75
        * (16.0 / max(4.0, lora_r)) ** 0.08
    )
    throughput_tps = max(120.0, throughput_tps)
    elapsed_sec = float(token_budget) / throughput_tps
    elapsed_sec *= rng.uniform(0.92, 1.08)
    elapsed_sec = max(1.0, elapsed_sec)
    gpu_hours = elapsed_sec * max(1, int(gpu_count)) / 3600.0

    return {
        "final_val_loss": final_val_loss,
        "best_val_loss": max(0.7, best_val_loss),
        "elapsed_sec": elapsed_sec,
        "gpu_hours": gpu_hours,
    }


def _write_simulated_curve(
    *,
    curve_path: Path,
    token_budget: int,
    eval_points: int,
    metrics: Dict[str, float],
    hp: Dict[str, Any],
    seed: int,
) -> None:
    rng = random.Random(seed + 17)
    ensure_dir(curve_path.parent)
    if curve_path.exists():
        curve_path.unlink()

    start_loss = metrics["final_val_loss"] + 0.8 + rng.uniform(0.0, 0.25)
    best = metrics["best_val_loss"]
    for point_idx in range(1, eval_points + 1):
        progress = point_idx / float(eval_points)
        val_loss = metrics["final_val_loss"] + (start_loss - metrics["final_val_loss"]) * math.exp(-4.0 * progress)
        val_loss += rng.gauss(0.0, 0.015)
        val_loss = max(best, val_loss)
        train_loss = max(0.5, val_loss - rng.uniform(0.03, 0.2))
        lr_peak = _safe_float(hp.get("learning_rate"), 2e-4)
        warmup_ratio = _safe_float(hp.get("warmup_ratio"), 0.03)
        if progress < warmup_ratio:
            lr = lr_peak * (progress / max(warmup_ratio, 1e-4))
        else:
            lr = lr_peak * max(0.1, 1.0 - 0.9 * (progress - warmup_ratio) / max(1e-4, 1.0 - warmup_ratio))
        row = {
            "step": point_idx,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
            "tokens_seen": int(token_budget * progress),
            "elapsed_sec": metrics["elapsed_sec"] * progress,
        }
        append_jsonl(curve_path, row)


def _run_external_training(
    trial: TrialConfig,
    global_config: Dict[str, Any],
    dataset_metadata: Dict[str, Any],
) -> TrialResult:
    train_cfg = global_config.get("training", {})
    command_template = train_cfg.get("external_command")
    if not command_template:
        raise ValueError(
            "training.mode=external requires training.external_command in config."
        )
    curves_root = Path(global_config.get("paths", {}).get("history_root", "outputs/history")) / "curves"
    metrics_root = Path(global_config.get("paths", {}).get("history_root", "outputs/history")) / "metrics"
    ensure_dir(curves_root)
    ensure_dir(metrics_root)
    curve_path = curves_root / f"{trial.run_id}.jsonl"
    metrics_path = metrics_root / f"{trial.run_id}.json"
    start = time.time()

    cmd = command_template.format(
        run_id=trial.run_id,
        dataset_id=trial.dataset_id,
        train_path=trial.paths["train_path"],
        val_path=trial.paths["val_path"],
        token_budget=trial.token_budget,
        seq_len=trial.seq_len,
        hp_json=json.dumps(trial.hp),
        metrics_path=str(metrics_path),
        curve_path=str(curve_path),
    )
    completed = subprocess.run(cmd, shell=True, check=False, capture_output=True, text=True)
    elapsed_sec = time.time() - start

    if completed.returncode != 0:
        return TrialResult(
            status="failed",
            val_loss=float("nan"),
            gpu_hours=0.0,
            elapsed_sec=elapsed_sec,
            peak_mem_mb=0.0,
            train_curve_path=str(curve_path),
            error_class="ExternalCommandFailed",
            error_message=(completed.stderr or completed.stdout or "").strip()[:2000],
        )

    if not metrics_path.exists():
        return TrialResult(
            status="failed",
            val_loss=float("nan"),
            gpu_hours=0.0,
            elapsed_sec=elapsed_sec,
            peak_mem_mb=0.0,
            train_curve_path=str(curve_path),
            error_class="MissingExternalMetrics",
            error_message=f"Expected metrics file at {metrics_path}",
        )

    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    val_loss = _safe_float(metrics_payload.get("val_loss"), float("nan"))
    best_val_loss = _safe_float(metrics_payload.get("best_val_loss"), val_loss)
    gpu_hours = _safe_float(metrics_payload.get("gpu_hours"), elapsed_sec / 3600.0)
    peak_mem_mb = _safe_float(metrics_payload.get("peak_mem_mb"), 0.0)
    return TrialResult(
        status="ok",
        val_loss=val_loss,
        gpu_hours=gpu_hours,
        elapsed_sec=elapsed_sec,
        peak_mem_mb=peak_mem_mb,
        train_curve_path=str(curve_path),
        best_val_loss=best_val_loss,
    )


def run_trial_once(
    trial: TrialConfig,
    global_config: Dict[str, Any],
    dataset_metadata: Dict[str, Any],
) -> TrialResult:
    train_cfg = global_config.get("training", {})
    mode = str(train_cfg.get("mode", "simulate")).lower()
    eval_points = int(train_cfg.get("eval_points", 20))
    gpu_count = int(train_cfg.get("gpu_count", 1))
    if "gpu_count" in trial.slurm_meta:
        gpu_count = int(trial.slurm_meta["gpu_count"])

    curves_root = Path(global_config.get("paths", {}).get("history_root", "outputs/history")) / "curves"
    curve_path = curves_root / f"{trial.run_id}.jsonl"

    if mode == "external":
        return _run_external_training(
            trial=trial,
            global_config=global_config,
            dataset_metadata=dataset_metadata,
        )

    try:
        metrics = _compute_simulated_metrics(
            hp=trial.hp,
            dataset_meta=dataset_metadata.get("metafeatures", {}),
            token_budget=trial.token_budget,
            seq_len=trial.seq_len,
            seed=trial.seed,
            gpu_count=gpu_count,
        )
        _write_simulated_curve(
            curve_path=curve_path,
            token_budget=trial.token_budget,
            eval_points=eval_points,
            metrics=metrics,
            hp=trial.hp,
            seed=trial.seed,
        )
        peak_mem = _estimate_peak_mem_mb(hp=trial.hp, seq_len=trial.seq_len)
        return TrialResult(
            status="ok",
            val_loss=metrics["final_val_loss"],
            best_val_loss=metrics["best_val_loss"],
            gpu_hours=metrics["gpu_hours"],
            elapsed_sec=metrics["elapsed_sec"],
            peak_mem_mb=peak_mem,
            train_curve_path=str(curve_path),
        )
    except Exception as exc:  # pragma: no cover - defensive.
        return TrialResult(
            status="failed",
            val_loss=float("nan"),
            gpu_hours=0.0,
            elapsed_sec=0.0,
            peak_mem_mb=0.0,
            train_curve_path=str(curve_path),
            error_class=type(exc).__name__,
            error_message=str(exc),
        )


def run_trial_with_retry(
    trial: TrialConfig,
    global_config: Dict[str, Any],
    dataset_metadata: Dict[str, Any],
) -> TrialResult:
    max_retries = int(global_config.get("retry", {}).get("max_retries", 1))
    attempt = 0
    last_result: Optional[TrialResult] = None
    while attempt <= max_retries:
        trial.retry_count = attempt
        result = run_trial_once(
            trial=trial,
            global_config=global_config,
            dataset_metadata=dataset_metadata,
        )
        if result.status == "ok":
            return result
        last_result = result
        attempt += 1
    assert last_result is not None
    return last_result


def build_run_record(
    trial: TrialConfig,
    result: TrialResult,
    dataset_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    metafeatures = dataset_metadata.get("metafeatures", {})
    record: Dict[str, Any] = {
        "run_id": trial.run_id,
        "dataset_id": trial.dataset_id,
        "dataset_slug": trial.dataset_slug,
        "stage": trial.stage,
        "seed": int(trial.seed),
        "status": result.status,
        "val_loss": result.val_loss,
        "best_val_loss": result.best_val_loss,
        "gpu_hours": result.gpu_hours,
        "elapsed_sec": result.elapsed_sec,
        "peak_mem_mb": result.peak_mem_mb,
        "train_curve_path": result.train_curve_path,
        "retry_count": int(trial.retry_count),
        "timestamp": utc_now_iso(),
        "hp_json": json.dumps(trial.hp, sort_keys=True),
        "metafeatures_json": json.dumps(metafeatures, sort_keys=True),
        "error_class": result.error_class,
        "error_message": result.error_message,
    }
    for key in ("ttr_score", "hapax_ratio", "zipf_alpha", "mean_seq_len", "std_seq_len", "token_entropy", "char_per_token"):
        if key in metafeatures:
            record[key] = metafeatures[key]
    for hp_key, hp_value in trial.hp.items():
        record[hp_key] = hp_value
    return record
