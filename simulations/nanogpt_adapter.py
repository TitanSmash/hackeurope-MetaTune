from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


STEP_RE = re.compile(
    r"step\s+(?P<step>\d+):\s*train loss\s+(?P<train>[0-9eE+\-.]+),\s*val loss\s+(?P<val>[0-9eE+\-.]+)",
    flags=re.IGNORECASE,
)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _iter_jsonl_text(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("text")
            if isinstance(text, str) and text.strip():
                yield text


def _prepare_nanogpt_dataset(
    *,
    repo_dir: Path,
    train_jsonl: Path,
    val_jsonl: Path,
    dataset_id: str,
) -> str:
    import tiktoken

    dataset_hash = hashlib.sha1(
        f"{dataset_id}|{train_jsonl.resolve()}|{val_jsonl.resolve()}".encode("utf-8")
    ).hexdigest()[:12]
    safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", dataset_id).strip("_").lower() or "dataset"
    dataset_name = f"metatune_{safe_name}_{dataset_hash}"
    dataset_dir = repo_dir / "data" / dataset_name
    train_bin = dataset_dir / "train.bin"
    val_bin = dataset_dir / "val.bin"
    stats_json = dataset_dir / "stats.json"

    if train_bin.exists() and val_bin.exists() and stats_json.exists():
        return dataset_name

    dataset_dir.mkdir(parents=True, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    def encode_file(src_jsonl: Path, dst_bin: Path) -> int:
        tokens_written = 0
        with dst_bin.open("wb") as handle:
            for text in _iter_jsonl_text(src_jsonl):
                ids = enc.encode_ordinary(text)
                ids.append(eot)
                arr = np.asarray(ids, dtype=np.uint16)
                arr.tofile(handle)
                tokens_written += int(arr.size)
        return tokens_written

    train_tokens = encode_file(train_jsonl, train_bin)
    val_tokens = encode_file(val_jsonl, val_bin)
    stats = {
        "tokenizer": "gpt2",
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
    }
    stats_json.write_text(json.dumps(stats, ensure_ascii=True, indent=2), encoding="utf-8")
    return dataset_name


def _build_train_command(
    *,
    repo_dir: Path,
    config_file: str,
    dataset_name: str,
    out_dir: Path,
    hp: Dict[str, Any],
    token_budget: int,
    seq_len: int,
    device: str,
    compile_flag: bool,
    grad_accum: int,
    eval_iters: int,
    log_interval: int,
    extra_args: str,
) -> List[str]:
    batch_size = _safe_int(hp.get("batch_size"), 8)
    learning_rate = _safe_float(hp.get("learning_rate"), 2e-4)
    lora_rank = _safe_int(hp.get("lora_r"), 8)
    lora_alpha = _safe_int(hp.get("lora_alpha"), 32)
    lora_dropout = _safe_float(hp.get("lora_dropout"), 0.0)

    grad_accum = max(1, grad_accum)
    tokens_per_iter = max(1, batch_size * seq_len * grad_accum)
    max_iters = max(2, token_budget // tokens_per_iter)
    eval_interval = max(1, min(max_iters, max(1, max_iters // 10)))

    cmd = [
        "python",
        "train.py",
        config_file,
        f"--dataset={dataset_name}",
        "--init_from=gpt2",
        f"--out_dir={out_dir}",
        f"--device={device}",
        f"--compile={str(compile_flag)}",
        "--wandb_log=False",
        "--always_save_checkpoint=False",
        f"--batch_size={batch_size}",
        f"--learning_rate={learning_rate}",
        f"--lora_rank={lora_rank}",
        f"--lora_alpha={lora_alpha}",
        f"--lora_dropout={lora_dropout}",
        f"--block_size={seq_len}",
        f"--gradient_accumulation_steps={grad_accum}",
        f"--max_iters={max_iters}",
        f"--lr_decay_iters={max_iters}",
        "--decay_lr=False",
        f"--eval_interval={eval_interval}",
        f"--eval_iters={max(1, eval_iters)}",
        f"--log_interval={max(1, log_interval)}",
    ]
    if extra_args.strip():
        cmd.extend(shlex.split(extra_args))
    return cmd


def _extract_curve(log_text: str, curve_path: Path, token_budget: int, max_iters: int) -> float:
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    with curve_path.open("w", encoding="utf-8") as handle:
        for match in STEP_RE.finditer(log_text):
            step = int(match.group("step"))
            train_loss = float(match.group("train"))
            val_loss = float(match.group("val"))
            if val_loss < best_val:
                best_val = val_loss
            progress = min(1.0, step / max(1, max_iters))
            row = {
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "tokens_seen": int(token_budget * progress),
            }
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return best_val


def _query_peak_mem_mb() -> float:
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return 0.0
    best = 0.0
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            best = max(best, float(line))
        except ValueError:
            continue
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="nanoGPT-LoRA adapter for simulation pipeline")
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--config-file", default="config/lora_shakespeare.py")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--val-path", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--token-budget", required=True, type=int)
    parser.add_argument("--seq-len", required=True, type=int)
    parser.add_argument("--hp-json", required=True)
    parser.add_argument("--metrics-path", required=True)
    parser.add_argument("--curve-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile", default="False")
    parser.add_argument("--gradient-accumulation-steps", default=1, type=int)
    parser.add_argument("--eval-iters", default=20, type=int)
    parser.add_argument("--log-interval", default=1, type=int)
    parser.add_argument("--extra-args", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_dir = Path(args.repo_dir).resolve()
    train_jsonl = Path(args.train_path).resolve()
    val_jsonl = Path(args.val_path).resolve()
    metrics_path = Path(args.metrics_path).resolve()
    curve_path = Path(args.curve_path).resolve()
    out_dir = metrics_path.parent / "nanogpt_runs" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    hp = json.loads(args.hp_json)
    dataset_name = _prepare_nanogpt_dataset(
        repo_dir=repo_dir,
        train_jsonl=train_jsonl,
        val_jsonl=val_jsonl,
        dataset_id=args.dataset_id,
    )
    compile_flag = str(args.compile).lower() in {"1", "true", "yes", "on"}
    command = _build_train_command(
        repo_dir=repo_dir,
        config_file=args.config_file,
        dataset_name=dataset_name,
        out_dir=out_dir,
        hp=hp,
        token_budget=int(args.token_budget),
        seq_len=int(args.seq_len),
        device=args.device,
        compile_flag=compile_flag,
        grad_accum=int(args.gradient_accumulation_steps),
        eval_iters=int(args.eval_iters),
        log_interval=int(args.log_interval),
        extra_args=str(args.extra_args),
    )
    max_iters = 2
    for arg in command:
        if arg.startswith("--max_iters="):
            max_iters = _safe_int(arg.split("=", 1)[1], 2)
            break

    started = time.time()
    completed = subprocess.run(
        command,
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.time() - started
    combined_log = (completed.stdout or "") + "\n" + (completed.stderr or "")
    (out_dir / "train.log").write_text(combined_log, encoding="utf-8")

    best_val_loss = _extract_curve(
        log_text=combined_log,
        curve_path=curve_path,
        token_budget=int(args.token_budget),
        max_iters=max_iters,
    )

    metrics = {
        "val_loss": best_val_loss,
        "best_val_loss": best_val_loss,
        "gpu_hours": elapsed / 3600.0,
        "peak_mem_mb": _query_peak_mem_mb(),
        "elapsed_sec": elapsed,
        "return_code": completed.returncode,
        "dataset_name": dataset_name,
        "out_dir": str(out_dir),
        "command": " ".join(shlex.quote(part) for part in command),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=True, indent=2), encoding="utf-8")

    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    if not np.isfinite(best_val_loss):
        raise SystemExit("No validation loss parsed from nanoGPT logs. Check out_dir/train.log.")


if __name__ == "__main__":
    main()

