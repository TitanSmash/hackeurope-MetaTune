from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any

import requests


ALLOWED_MODELS = {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
STEP_RE = re.compile(
    r"step\s+(?P<step>\d+):\s*train loss\s+(?P<train>[0-9eE+\-.]+),\s*val loss\s+(?P<val>[0-9eE+\-.]+)",
    flags=re.IGNORECASE,
)


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _safe_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _safe_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _normalize_rel_path(value: str) -> str:
    rel = value.strip()
    if rel.startswith("/"):
        raise ValueError("DATASET_REL_PATH must be relative")
    if rel in {"", ".", ".."}:
        raise ValueError("DATASET_REL_PATH must not be empty or traversal")
    parts = PurePosixPath(rel).parts
    if ".." in parts:
        raise ValueError("DATASET_REL_PATH must not contain '..'")
    return str(PurePosixPath(rel))


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value)
    return cleaned.strip("_").lower() or "dataset"


def _parse_hp_json(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"HP_JSON is invalid JSON: {e.msg}") from e
    if not isinstance(payload, dict):
        raise ValueError("HP_JSON must decode to an object")

    required = ["lora_r", "lora_alpha", "lora_dropout", "learning_rate", "batch_size"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"HP_JSON missing required keys: {', '.join(missing)}")
    return payload


def _build_train_command(
    *,
    model: str,
    dataset_alias: str,
    hp: dict[str, Any],
    token_budget: int,
    seq_len: int,
    out_dir: Path,
) -> tuple[list[str], int]:
    lora_r = int(hp["lora_r"])
    lora_alpha = int(hp["lora_alpha"])
    lora_dropout = float(hp["lora_dropout"])
    learning_rate = float(hp["learning_rate"])
    batch_size = int(hp["batch_size"])
    grad_accum = max(1, _safe_int(os.getenv("GRADIENT_ACCUMULATION_STEPS"), 1))
    eval_iters = max(1, _safe_int(os.getenv("EVAL_ITERS"), 20))
    compile_flag = os.getenv("COMPILE", "False")
    device = os.getenv("DEVICE", "cuda")

    tokens_per_iter = max(1, batch_size * seq_len * grad_accum)
    max_iters = max(2, token_budget // tokens_per_iter)
    eval_interval = max(1, min(max_iters, max_iters // 10 or 1))

    cmd = [
        "python",
        "train.py",
        "config/lora_shakespeare.py",
        f"--dataset={dataset_alias}",
        f"--init_from={model}",
        "--wandb_log=False",
        "--always_save_checkpoint=False",
        "--decay_lr=False",
        f"--learning_rate={learning_rate}",
        f"--lora_rank={lora_r}",
        f"--lora_alpha={lora_alpha}",
        f"--lora_dropout={lora_dropout}",
        f"--batch_size={batch_size}",
        f"--block_size={seq_len}",
        f"--gradient_accumulation_steps={grad_accum}",
        f"--max_iters={max_iters}",
        f"--lr_decay_iters={max_iters}",
        f"--eval_interval={eval_interval}",
        f"--eval_iters={eval_iters}",
        f"--out_dir={out_dir}",
        f"--device={device}",
        f"--compile={compile_flag}",
    ]
    return cmd, max_iters


def _parse_curve(log_text: str, curve_path: Path, token_budget: int, max_iters: int) -> dict[str, float]:
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    final_val = float("nan")
    rows = 0
    with curve_path.open("w", encoding="utf-8") as handle:
        for match in STEP_RE.finditer(log_text):
            step = int(match.group("step"))
            train_loss = float(match.group("train"))
            val_loss = float(match.group("val"))
            best_val = min(best_val, val_loss)
            final_val = val_loss
            progress = min(1.0, step / max(1, max_iters))
            row = {
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "tokens_seen": int(token_budget * progress),
            }
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            rows += 1
    return {
        "best_val_loss": best_val if rows > 0 else float("nan"),
        "final_val_loss": final_val if rows > 0 else float("nan"),
        "curve_rows": float(rows),
    }


def _send_callback(callback_url: str, payload: dict[str, Any]) -> None:
    timeout = _safe_float(os.getenv("CALLBACK_TIMEOUT_SECONDS"), 10.0)
    try:
        response = requests.post(callback_url, json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"[runner] callback failed: {e}", file=sys.stderr)


def main() -> int:
    run_id = _required_env("RUN_ID")
    model = _required_env("MODEL")
    if model not in ALLOWED_MODELS:
        raise ValueError(f"Unsupported MODEL '{model}'")
    dataset_rel_path = _normalize_rel_path(_required_env("DATASET_REL_PATH"))
    hp = _parse_hp_json(_required_env("HP_JSON"))
    token_budget = max(1, _safe_int(_required_env("TOKEN_BUDGET"), 1))
    seq_len = max(1, _safe_int(_required_env("SEQ_LEN"), 1))
    callback_url = _required_env("SCHEDULER_CALLBACK_URL")
    datasets_root = Path(os.getenv("DATASETS_ROOT", "/data/datasets"))
    output_root = Path(os.getenv("OUTPUT_ROOT", "/data/outputs"))
    nanogpt_repo = Path(os.getenv("NANOGPT_REPO_DIR", "simulations/nanoGPT-LoRA-master"))

    dataset_dir = datasets_root / dataset_rel_path
    train_bin = dataset_dir / "train.bin"
    val_bin = dataset_dir / "val.bin"
    if not train_bin.exists() or not val_bin.exists():
        raise FileNotFoundError(
            f"Dataset is missing train.bin or val.bin under {dataset_dir}"
        )
    if not nanogpt_repo.exists():
        raise FileNotFoundError(f"nanoGPT repo not found at {nanogpt_repo}")

    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_path = output_dir / "curve.jsonl"
    train_log_path = output_dir / "train.log"
    metrics_path = output_dir / "metrics.json"
    run_meta_path = output_dir / "run_meta.json"

    dataset_alias = f"metatune_{_slugify(dataset_rel_path)}_{run_id[-8:]}"
    data_dir = nanogpt_repo / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    alias_dir = data_dir / dataset_alias
    if alias_dir.exists() or alias_dir.is_symlink():
        if alias_dir.is_dir() and not alias_dir.is_symlink():
            raise RuntimeError(
                f"Refusing to replace existing directory at {alias_dir}; "
                "expected symlink path."
            )
        alias_dir.unlink()
    alias_dir.symlink_to(dataset_dir.resolve(), target_is_directory=True)

    command, max_iters = _build_train_command(
        model=model,
        dataset_alias=dataset_alias,
        hp=hp,
        token_budget=token_budget,
        seq_len=seq_len,
        out_dir=output_dir,
    )

    started = time.time()
    completed = subprocess.run(
        command,
        cwd=nanogpt_repo,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.time() - started
    combined_log = (completed.stdout or "") + "\n" + (completed.stderr or "")
    train_log_path.write_text(combined_log, encoding="utf-8")

    parsed = _parse_curve(
        log_text=combined_log,
        curve_path=curve_path,
        token_budget=token_budget,
        max_iters=max_iters,
    )
    best_val_loss = float(parsed["best_val_loss"])
    final_val_loss = float(parsed["final_val_loss"])
    has_curve = not math.isnan(best_val_loss) and not math.isnan(final_val_loss)
    status = "SUCCEEDED" if completed.returncode == 0 and has_curve else "FAILED"

    metrics = {
        "val_loss": final_val_loss,
        "best_val_loss": best_val_loss,
        "gpu_hours": elapsed / 3600.0,
        "elapsed_sec": elapsed,
        "return_code": completed.returncode,
        "max_iters": max_iters,
        "curve_rows": int(parsed["curve_rows"]),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=True, indent=2), encoding="utf-8")
    run_meta = {
        "run_id": run_id,
        "model": model,
        "dataset_rel_path": dataset_rel_path,
        "dataset_dir": str(dataset_dir),
        "dataset_alias": dataset_alias,
        "nanogpt_repo": str(nanogpt_repo),
        "command": command,
        "status": status,
    }
    run_meta_path.write_text(json.dumps(run_meta, ensure_ascii=True, indent=2), encoding="utf-8")

    artifacts = {
        "output_dir": str(output_dir),
        "metrics_path": str(metrics_path),
        "curve_path": str(curve_path),
        "train_log_path": str(train_log_path),
        "run_meta_path": str(run_meta_path),
    }
    callback_payload = {
        "status": status,
        "metrics": metrics,
        "artifacts": artifacts,
        "error_message": ""
        if status == "SUCCEEDED"
        else "Training command failed or no validation curve parsed",
    }
    _send_callback(callback_url, callback_payload)

    return 0 if status == "SUCCEEDED" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[runner] fatal error: {exc}", file=sys.stderr)
        callback_url = os.getenv("SCHEDULER_CALLBACK_URL")
        run_id = os.getenv("RUN_ID", "unknown")
        output_root = Path(os.getenv("OUTPUT_ROOT", "/data/outputs"))
        output_dir = output_root / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        failure_payload = {
            "status": "FAILED",
            "metrics": {},
            "artifacts": {"output_dir": str(output_dir)},
            "error_message": str(exc),
        }
        if callback_url:
            _send_callback(callback_url, failure_payload)
        raise

