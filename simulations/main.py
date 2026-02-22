from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from simulations.ds_downloader import download_datasets, normalize_dataset_entry
from simulations.ds_loader import prepare_one_dataset
from simulations.io_utils import append_jsonl, count_jsonl_rows, ensure_dir, iter_jsonl, load_yaml, read_json, read_jsonl, slugify_dataset_id, write_jsonl
from simulations.nanogpt_dataset_prep import prepare_many_datasets_for_nanogpt
from simulations.optimizer import DEFAULT_HP_SPACE, generate_coarse_grid
from simulations.slurm import SlurmConfig, write_sbatch_script, write_submit_all_script
from simulations.training import TrialConfig, build_run_record, run_trial_with_retry


def _load_config(path: Path | str) -> Dict[str, Any]:
    config = load_yaml(path)
    config["_config_path"] = str(Path(path).resolve())
    return config


def _resolve_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    paths = config.get("paths", {})
    output_root_env = os.environ.get("SIM_OUTPUT_ROOT")
    if output_root_env:
        output_root = Path(output_root_env)
        default_roots = {
            "raw_root": output_root / "raw_datasets",
            "processed_root": output_root / "datasets",
            "manifests_root": output_root / "manifests",
            "history_root": output_root / "history",
            "slurm_root": output_root / "slurm",
            "hf_cache_dir": output_root / "hf_cache",
        }
    else:
        default_roots = {
            "raw_root": Path("outputs/raw_datasets"),
            "processed_root": Path("outputs/datasets"),
            "manifests_root": Path("outputs/manifests"),
            "history_root": Path("outputs/history"),
            "slurm_root": Path("outputs/slurm"),
            "hf_cache_dir": Path("outputs/hf_cache"),
        }

    env_overrides = {
        "raw_root": os.environ.get("SIM_RAW_ROOT"),
        "processed_root": os.environ.get("SIM_PROCESSED_ROOT"),
        "manifests_root": os.environ.get("SIM_MANIFESTS_ROOT"),
        "history_root": os.environ.get("SIM_HISTORY_ROOT"),
        "slurm_root": os.environ.get("SIM_SLURM_ROOT"),
        "hf_cache_dir": os.environ.get("SIM_HF_CACHE_DIR"),
    }
    resolved = {
        key: Path(env_overrides[key]) if env_overrides[key] else Path(paths.get(key, default_roots[key]))
        for key in ("raw_root", "processed_root", "manifests_root", "history_root", "slurm_root", "hf_cache_dir")
    }
    for path in resolved.values():
        ensure_dir(path)
    return resolved


def _dataset_entries(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    datasets = config.get("datasets", [])
    if not datasets:
        raise ValueError("Config requires a non-empty `datasets` list.")
    return [normalize_dataset_entry(entry) for entry in datasets]


def _history_paths(paths: Dict[str, Path]) -> Tuple[Path, Path, Path]:
    history_jsonl = paths["history_root"] / "runs.jsonl"
    history_parquet = paths["history_root"] / "runs.parquet"
    history_csv = paths["history_root"] / "runs.csv"
    return history_jsonl, history_parquet, history_csv


def _safe_float(value: Any, default: float = float("inf")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sync_history_outputs(history_jsonl: Path, history_parquet: Path, history_csv: Path) -> None:
    if not history_jsonl.exists():
        return
    records = read_jsonl(history_jsonl)
    if not records:
        return

    try:
        import pandas as pd
        df = pd.DataFrame(records)
        try:
            df.to_parquet(history_parquet, index=False)
        except Exception:
            pass
        df.to_csv(history_csv, index=False)
    except ImportError:
        keys: List[str] = []
        seen = set()
        for row in records:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        ensure_dir(history_csv.parent)
        with history_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for row in records:
                writer.writerow(row)


def _resolve_hp_space(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    hp_space = config.get("search", {}).get("hp_space")
    if not hp_space:
        return DEFAULT_HP_SPACE
    resolved: Dict[str, Dict[str, Any]] = {}
    for key, value in hp_space.items():
        if isinstance(value, list):
            resolved[key] = {"type": "categorical", "values": value}
        elif isinstance(value, dict):
            resolved[key] = dict(value)
        else:
            raise ValueError(f"Invalid HP space spec for {key}: {value!r}")
    return resolved


def _trial_signature(dataset_id: str, hp: Dict[str, Any], seed: int) -> str:
    hp_json = json.dumps(hp, sort_keys=True, separators=(",", ":"))
    return f"{dataset_id}|{seed}|{hp_json}"


def _collect_seen_signatures(paths: Dict[str, Path]) -> set[str]:
    seen: set[str] = set()
    history_jsonl, _, _ = _history_paths(paths)
    for row in iter_jsonl(history_jsonl):
        dataset_id = row.get("dataset_id")
        hp = {}
        hp_json = row.get("hp_json")
        if isinstance(hp_json, str):
            try:
                hp = json.loads(hp_json)
            except json.JSONDecodeError:
                hp = {}
        if not dataset_id or not hp:
            continue
        seen.add(_trial_signature(dataset_id, hp=hp, seed=int(row.get("seed", 1))))

    for manifest_path in paths["manifests_root"].glob("*.jsonl"):
        for row in iter_jsonl(manifest_path):
            dataset_id = row.get("dataset_id")
            hp = row.get("hp")
            seed = int(row.get("seed", 1))
            if dataset_id and isinstance(hp, dict):
                seen.add(_trial_signature(dataset_id, hp=hp, seed=seed))
    return seen


def _build_manifest_rows(
    *,
    dataset_meta: Dict[str, Any],
    hp_configs: Sequence[Dict[str, Any]],
    stage: str,
    seeds: Sequence[int],
    token_budget: int,
    seq_len: int,
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    dataset_id = dataset_meta["dataset_id"]
    dataset_slug = slugify_dataset_id(dataset_meta["dataset_key"])
    for hp_idx, hp in enumerate(hp_configs):
        for seed in seeds:
            hp_signature = abs(hash(json.dumps(hp, sort_keys=True))) % 10_000_000
            run_id = f"{stage}_{dataset_slug}_{hp_idx:04d}_s{seed}_{hp_signature:07d}"
            rows.append(
                {
                    "run_id": run_id,
                    "dataset_id": dataset_id,
                    "dataset_slug": dataset_slug,
                    "stage": stage,
                    "seed": int(seed),
                    "token_budget": int(token_budget),
                    "seq_len": int(seq_len),
                    "hp": hp,
                    "paths": {
                        "train_path": dataset_meta["train_path"],
                        "val_path": dataset_meta["val_path"],
                    },
                    "device": device,
                    "slurm_meta": {"gpu_count": 1},
                    "retry_count": 0,
                }
            )
    return rows


def command_prefetch(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    paths = _resolve_paths(config)
    dataset_entries = _dataset_entries(config)
    split_policy = config.get("split_policy", "native_then_95_5")
    format_policy = config.get("format_policy", "instruction_chat")
    metafeature_sample_tokens = int(config.get("metafeature_sample_tokens", 2_000_000))

    download_results = download_datasets(
        dataset_entries=dataset_entries,
        raw_root=paths["raw_root"],
        hf_cache_dir=paths["hf_cache_dir"],
        force=args.force,
    )

    metadata_entries: List[Dict[str, Any]] = []
    for dataset_entry, download_meta in zip(dataset_entries, download_results):
        max_train_rows = dataset_entry.get("max_train_rows")
        max_val_rows = dataset_entry.get("max_val_rows")
        max_train_bytes = dataset_entry.get("max_train_bytes")
        max_val_bytes = dataset_entry.get("max_val_bytes")
        max_total_bytes = dataset_entry.get("max_total_bytes")
        if max_total_bytes is not None and (max_train_bytes is None and max_val_bytes is None):
            total_bytes = int(max_total_bytes)
            max_train_bytes = int(total_bytes * 0.95)
            max_val_bytes = total_bytes - int(max_train_bytes)
        prepared = prepare_one_dataset(
            download_metadata=download_meta,
            processed_root=paths["processed_root"],
            format_policy=format_policy,
            split_policy=split_policy,
            val_ratio=0.05,
            seed=int(config.get("seed", 42)),
            max_train_rows=max_train_rows,
            max_val_rows=max_val_rows,
            max_train_bytes=max_train_bytes,
            max_val_bytes=max_val_bytes,
            metafeature_sample_tokens=metafeature_sample_tokens,
        )
        metadata_entries.append(prepared)
        print(f"[prefetch] prepared {prepared['dataset_id']} -> {prepared['processed_dir']}")

    (paths["processed_root"] / "index.json").write_text(
        json.dumps(metadata_entries, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    print(f"[prefetch] completed {len(metadata_entries)} datasets")


def command_prepare_nanogpt_data(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    paths = _resolve_paths(config)
    metadata_entries = _metadata_entries_or_raise(paths)

    nanogpt_cfg = config.get("training", {}).get("nanogpt", {})
    repo_dir = Path(
        args.repo_dir
        if args.repo_dir
        else nanogpt_cfg.get("repo_dir", "simulations/nanoGPT-LoRA-master")
    )
    if not repo_dir.exists():
        raise FileNotFoundError(f"nanoGPT repo not found: {repo_dir}")
    dataset_prefix = str(nanogpt_cfg.get("dataset_prefix", "metatune"))

    results = prepare_many_datasets_for_nanogpt(
        repo_dir=repo_dir,
        prepared_metadata_rows=metadata_entries,
        dataset_prefix=dataset_prefix,
        force=bool(args.force),
    )

    csv_path = Path(args.output) if args.output else paths["history_root"] / "nanogpt_datasets.csv"
    ensure_dir(csv_path.parent)
    fieldnames = [
        "dataset_id",
        "dataset_key",
        "dataset_name",
        "dataset_dir",
        "train_bin",
        "val_bin",
        "train_tokens",
        "val_tokens",
        "prepared",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    mapping_path = paths["history_root"] / "nanogpt_dataset_map.json"
    mapping_payload = {row["dataset_id"]: row["dataset_name"] for row in results}
    mapping_path.write_text(json.dumps(mapping_payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[prepare-nanogpt-data] wrote dataset csv: {csv_path}")
    print(f"[prepare-nanogpt-data] wrote mapping json: {mapping_path}")


def _metadata_entries_or_raise(paths: Dict[str, Path]) -> List[Dict[str, Any]]:
    index_path = paths["processed_root"] / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Dataset metadata index not found at {index_path}. Run `prefetch` first."
        )
    payload = read_json(index_path, default=[])
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict) and "datasets" in payload:
        entries = payload["datasets"]
    else:
        entries = []
    if not entries:
        raise RuntimeError("No prepared datasets found. Run `prefetch` first.")
    return entries


def _write_stage_manifests(
    *,
    stage: str,
    rows_by_dataset: Dict[str, List[Dict[str, Any]]],
    manifests_root: Path,
) -> Dict[str, Path]:
    ensure_dir(manifests_root)
    written: Dict[str, Path] = {}
    global_rows: List[Dict[str, Any]] = []

    for dataset_slug, rows in rows_by_dataset.items():
        manifest_path = manifests_root / f"{dataset_slug}_{stage}.jsonl"
        write_jsonl(manifest_path, rows)
        written[dataset_slug] = manifest_path
        global_rows.extend(rows)

    global_manifest = manifests_root / f"{stage}_all.jsonl"
    write_jsonl(global_manifest, global_rows)
    written["_all"] = global_manifest
    return written


def command_build_stage1(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    paths = _resolve_paths(config)
    metadata_entries = _metadata_entries_or_raise(paths)
    hp_space = _resolve_hp_space(config)
    stage1_trials = int(config.get("search", {}).get("stage1_trials", 96))
    token_budget = int(config.get("training", {}).get("token_budget_per_trial", 25_000_000))
    seq_len = int(config.get("training", {}).get("seq_len", 1024))
    seed_stage1 = config.get("search", {}).get("seed_stage1", [1])
    if isinstance(seed_stage1, int):
        seed_stage1 = [seed_stage1]
    base_seed = int(config.get("seed", 42))
    seen = _collect_seen_signatures(paths)

    rows_by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for ds_idx, dataset_meta in enumerate(metadata_entries):
        dataset_slug = slugify_dataset_id(dataset_meta["dataset_key"])
        coarse = generate_coarse_grid(
            hp_space=hp_space,
            n_trials=stage1_trials,
            seed=base_seed + ds_idx * 997,
        )
        filtered: List[Dict[str, Any]] = []
        for hp in coarse:
            duplicate = False
            for seed in seed_stage1:
                sig = _trial_signature(dataset_meta["dataset_id"], hp=hp, seed=int(seed))
                if sig in seen:
                    duplicate = True
                    break
            if not duplicate:
                filtered.append(hp)
            if len(filtered) >= stage1_trials:
                break
        rows = _build_manifest_rows(
            dataset_meta=dataset_meta,
            hp_configs=filtered,
            stage="stage1",
            seeds=[int(seed) for seed in seed_stage1],
            token_budget=token_budget,
            seq_len=seq_len,
            device="cuda",
        )
        rows_by_dataset[dataset_slug] = rows
        print(f"[stage1] {dataset_meta['dataset_id']}: {len(rows)} trials")

    written = _write_stage_manifests(
        stage="stage1",
        rows_by_dataset=rows_by_dataset,
        manifests_root=paths["manifests_root"],
    )
    print(f"[stage1] global manifest: {written['_all']}")


def _sample_unique_configs(
    *,
    hp_space: Dict[str, Dict[str, Any]],
    n_trials: int,
    seed: int,
    dataset_id: str,
    seeds: Sequence[int],
    seen_signatures: set[str],
) -> List[Dict[str, Any]]:
    chosen: List[Dict[str, Any]] = []
    attempts = 0
    max_attempts = max(5, n_trials * 20)
    while len(chosen) < n_trials and attempts < max_attempts:
        attempts += 1
        # Generate one stratified sample at a time to keep retry logic simple.
        candidate = generate_coarse_grid(hp_space=hp_space, n_trials=1, seed=seed + attempts)[0]
        duplicate = False
        for trial_seed in seeds:
            sig = _trial_signature(dataset_id, hp=candidate, seed=int(trial_seed))
            if sig in seen_signatures:
                duplicate = True
                break
        if duplicate:
            continue
        chosen.append(candidate)
        for trial_seed in seeds:
            seen_signatures.add(_trial_signature(dataset_id, hp=candidate, seed=int(trial_seed)))
    return chosen


def command_build_stage2(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    paths = _resolve_paths(config)
    metadata_entries = _metadata_entries_or_raise(paths)
    hp_space = _resolve_hp_space(config)
    stage2_trials = int(config.get("search", {}).get("stage2_trials", 48))
    seed_stage = config.get("search", {}).get("seed_stage2", config.get("search", {}).get("seed_stage1", [1]))
    if isinstance(seed_stage, int):
        seed_stage = [seed_stage]
    token_budget = int(config.get("training", {}).get("token_budget_per_trial", 25_000_000))
    seq_len = int(config.get("training", {}).get("seq_len", 1024))
    seen = _collect_seen_signatures(paths)
    base_seed = int(config.get("seed", 42)) + 11_000

    rows_by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for ds_idx, dataset_meta in enumerate(metadata_entries):
        dataset_slug = slugify_dataset_id(dataset_meta["dataset_key"])
        chosen = _sample_unique_configs(
            hp_space=hp_space,
            n_trials=stage2_trials,
            seed=base_seed + ds_idx * 7919,
            dataset_id=dataset_meta["dataset_id"],
            seeds=[int(seed) for seed in seed_stage],
            seen_signatures=seen,
        )

        rows = _build_manifest_rows(
            dataset_meta=dataset_meta,
            hp_configs=chosen,
            stage="stage2",
            seeds=[int(seed) for seed in seed_stage],
            token_budget=token_budget,
            seq_len=seq_len,
            device="cuda",
        )
        rows_by_dataset[dataset_slug] = rows
        print(f"[stage2] {dataset_meta['dataset_id']}: {len(rows)} trials")

    written = _write_stage_manifests(
        stage="stage2",
        rows_by_dataset=rows_by_dataset,
        manifests_root=paths["manifests_root"],
    )
    print(f"[stage2] global manifest: {written['_all']}")


def command_build_reruns(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    paths = _resolve_paths(config)
    metadata_entries = _metadata_entries_or_raise(paths)
    history_jsonl, _, _ = _history_paths(paths)
    history_rows = read_jsonl(history_jsonl)
    if not history_rows:
        raise RuntimeError("No history found. Run stage trials first before reruns.")

    token_budget = int(config.get("training", {}).get("token_budget_per_trial", 25_000_000))
    seq_len = int(config.get("training", {}).get("seq_len", 1024))
    seed_reruns = config.get("search", {}).get("seed_reruns", [2, 3])
    if isinstance(seed_reruns, int):
        seed_reruns = [seed_reruns]
    top_k = int(config.get("search", {}).get("top_k_reruns", 12))
    seen = _collect_seen_signatures(paths)

    rows_by_dataset: Dict[str, List[Dict[str, Any]]] = {}
    for dataset_meta in metadata_entries:
        dataset_id = dataset_meta["dataset_id"]
        dataset_slug = slugify_dataset_id(dataset_meta["dataset_key"])
        ranked_rows: List[Tuple[float, float, Dict[str, Any]]] = []
        for row in history_rows:
            if row.get("dataset_id") != dataset_id:
                continue
            if row.get("status") != "ok":
                continue
            if int(row.get("seed", -1)) != 1:
                continue
            hp_json = row.get("hp_json")
            if not isinstance(hp_json, str):
                continue
            try:
                hp = json.loads(hp_json)
            except json.JSONDecodeError:
                continue
            if not isinstance(hp, dict) or not hp:
                continue
            ranked_rows.append((_safe_float(row.get("val_loss")), _safe_float(row.get("gpu_hours")), hp))

        if not ranked_rows:
            rows_by_dataset[dataset_slug] = []
            continue
        ranked_rows.sort(key=lambda item: (item[0], item[1]))
        hp_rows = [hp for _, _, hp in ranked_rows[:top_k]]

        selected_hps: List[Dict[str, Any]] = []
        for hp in hp_rows:
            duplicate = False
            for seed in seed_reruns:
                sig = _trial_signature(dataset_id, hp=hp, seed=int(seed))
                if sig in seen:
                    duplicate = True
                    break
            if not duplicate:
                selected_hps.append(hp)
        rows = _build_manifest_rows(
            dataset_meta=dataset_meta,
            hp_configs=selected_hps,
            stage="rerun",
            seeds=[int(seed) for seed in seed_reruns],
            token_budget=token_budget,
            seq_len=seq_len,
            device="cuda",
        )
        rows_by_dataset[dataset_slug] = rows
        print(f"[rerun] {dataset_id}: {len(rows)} trials")

    written = _write_stage_manifests(
        stage="rerun",
        rows_by_dataset=rows_by_dataset,
        manifests_root=paths["manifests_root"],
    )
    print(f"[rerun] global manifest: {written['_all']}")


def _slurm_config_from_yaml(config: Dict[str, Any]) -> SlurmConfig:
    slurm_cfg = config.get("slurm", {})
    account = str(slurm_cfg.get("account", "")).strip()
    if not account or account.upper() == "CHANGE_ME":
        raise ValueError("`slurm.account` is required in config.")
    return SlurmConfig(
        account=account,
        gpus=str(slurm_cfg.get("gpus", "rtx_4090:1")),
        time=str(slurm_cfg.get("time", "04:00:00")),
        array_concurrency=int(slurm_cfg.get("array_concurrency", 16)),
        cpus_per_task=int(slurm_cfg.get("cpus_per_task", 4)),
        mem=str(slurm_cfg.get("mem", "24G")),
        partition=slurm_cfg.get("partition"),
        constraint=slurm_cfg.get("constraint"),
    )


def command_make_slurm(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    paths = _resolve_paths(config)
    slurm_cfg = _slurm_config_from_yaml(config)
    stage_names = ["stage1", "stage2", "rerun"] if args.stage == "all" else [args.stage]
    script_paths: List[Path] = []

    for stage in stage_names:
        global_manifest = paths["manifests_root"] / f"{stage}_all.jsonl"
        if global_manifest.exists():
            script_path = paths["slurm_root"] / f"{stage}_all.sbatch"
            script_paths.append(
                write_sbatch_script(
                    output_path=script_path,
                    job_name=f"metatune_{stage}_all",
                    manifest_path=global_manifest,
                    config_path=Path(config["_config_path"]),
                    slurm=slurm_cfg,
                )
            )
            print(f"[slurm] wrote {script_path}")

        if args.per_dataset:
            for manifest_path in sorted(paths["manifests_root"].glob(f"*_{stage}.jsonl")):
                if manifest_path.name.endswith("_all.jsonl"):
                    continue
                dataset_slug = manifest_path.stem.replace(f"_{stage}", "")
                script_path = paths["slurm_root"] / f"{dataset_slug}_{stage}.sbatch"
                script_paths.append(
                    write_sbatch_script(
                        output_path=script_path,
                        job_name=f"metatune_{stage}_{dataset_slug}",
                        manifest_path=manifest_path,
                        config_path=Path(config["_config_path"]),
                        slurm=slurm_cfg,
                    )
                )
                print(f"[slurm] wrote {script_path}")

    if script_paths:
        submit_all = write_submit_all_script(
            script_paths=script_paths,
            destination=paths["slurm_root"] / "submit_all.sh",
        )
        print(f"[slurm] wrote {submit_all}")
    else:
        raise RuntimeError("No manifests found for requested stage(s).")


def _get_manifest_row(manifest_path: Path, index: int) -> Dict[str, Any]:
    for row_idx, row in enumerate(iter_jsonl(manifest_path)):
        if row_idx == index:
            return row
    raise IndexError(f"Manifest index out of range: {index} for {manifest_path}")


def command_run_trial(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    paths = _resolve_paths(config)
    manifest_path = Path(args.manifest)
    row = _get_manifest_row(manifest_path, args.index)
    dataset_slug = row["dataset_slug"]
    metadata_path = paths["processed_root"] / dataset_slug / "metadata.json"
    dataset_meta = read_json(metadata_path, default=None)
    if dataset_meta is None:
        raise FileNotFoundError(f"Missing dataset metadata at {metadata_path}")

    trial = TrialConfig(
        run_id=row["run_id"],
        dataset_id=row["dataset_id"],
        dataset_slug=row["dataset_slug"],
        stage=row["stage"],
        seed=int(row.get("seed", 1)),
        token_budget=int(row["token_budget"]),
        seq_len=int(row["seq_len"]),
        hp=dict(row["hp"]),
        paths=dict(row["paths"]),
        device=str(row.get("device", "cuda")),
        slurm_meta=dict(row.get("slurm_meta", {})),
        retry_count=int(row.get("retry_count", 0)),
    )
    result = run_trial_with_retry(trial=trial, global_config=config, dataset_metadata=dataset_meta)
    record = build_run_record(trial=trial, result=result, dataset_metadata=dataset_meta)
    history_jsonl, history_parquet, history_csv = _history_paths(paths)
    append_jsonl(history_jsonl, record)
    _sync_history_outputs(history_jsonl, history_parquet, history_csv)
    print(
        f"[run-trial] {trial.run_id} status={record['status']} val_loss={record['val_loss']:.4f} gpu_hours={record['gpu_hours']:.4f}"
    )


def _query_gpu_memory_ratio() -> Optional[float]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return None
    ratios = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        used, total = float(parts[0]), float(parts[1])
        if total > 0:
            ratios.append(used / total)
    if not ratios:
        return None
    return max(ratios)


def command_local_run(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    manifest_path = Path(args.manifest)
    total_rows = count_jsonl_rows(manifest_path)
    if total_rows <= 0:
        raise RuntimeError(f"Empty manifest: {manifest_path}")

    local_cfg = config.get("local", {})
    memory_guard = float(args.memory_guard if args.memory_guard is not None else local_cfg.get("memory_guard", 0.90))
    max_parallel = int(args.max_parallel if args.max_parallel is not None else local_cfg.get("max_parallel", 32))
    poll_sec = float(local_cfg.get("poll_sec", 5.0))

    pending = list(range(total_rows))
    running: List[Tuple[int, subprocess.Popen[str]]] = []
    command_base = [
        sys.executable,
        "-m",
        "simulations.main",
        "run-trial",
        "--config",
        str(Path(config["_config_path"])),
        "--manifest",
        str(manifest_path),
    ]

    while pending or running:
        still_running: List[Tuple[int, subprocess.Popen[str]]] = []
        for trial_idx, proc in running:
            code = proc.poll()
            if code is None:
                still_running.append((trial_idx, proc))
            else:
                print(f"[local] trial index={trial_idx} exit_code={code}")
        running = still_running

        while pending and len(running) < max_parallel:
            ratio = _query_gpu_memory_ratio()
            if ratio is not None and ratio >= memory_guard:
                break
            idx = pending.pop(0)
            cmd = command_base + ["--index", str(idx)]
            proc = subprocess.Popen(cmd)
            running.append((idx, proc))
            print(f"[local] launched trial index={idx} running={len(running)}")
            time.sleep(0.4)

        time.sleep(poll_sec)


def command_export_csv(args: argparse.Namespace) -> None:
    if args.history_jsonl:
        history_jsonl = Path(args.history_jsonl)
        history_parquet = Path(args.history_parquet) if args.history_parquet else history_jsonl.with_suffix(".parquet")
        history_csv = Path(args.output) if args.output else history_jsonl.with_suffix(".csv")
    else:
        try:
            config = _load_config(args.config)
            paths = _resolve_paths(config)
            history_jsonl, history_parquet, history_csv = _history_paths(paths)
            if args.output:
                history_csv = Path(args.output)
        except Exception:
            history_root = Path("outputs/history")
            history_jsonl = history_root / "runs.jsonl"
            history_parquet = history_root / "runs.parquet"
            history_csv = Path(args.output) if args.output else history_root / "runs.csv"
    if not history_jsonl.exists():
        raise FileNotFoundError(f"No history file found at {history_jsonl}. Run trials first.")
    _sync_history_outputs(history_jsonl, history_parquet, history_csv)
    print(f"[export-csv] wrote {history_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoRA meta-tuning data generation pipeline")
    parser.add_argument(
        "--config",
        default="simulations/configs/lora_metatune_v1.yaml",
        help="Path to experiment YAML config",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    prefetch = subparsers.add_parser("prefetch", help="Download and prepare datasets.")
    prefetch.add_argument("--force", action="store_true", help="Re-download raw datasets.")
    prefetch.set_defaults(func=command_prefetch)

    prep_nanogpt = subparsers.add_parser(
        "prepare-nanogpt-data",
        help="Convert prepared JSONL datasets into nanoGPT train.bin/val.bin and export CSV.",
    )
    prep_nanogpt.add_argument("--repo-dir", default=None, help="Path to nanoGPT-LoRA repo")
    prep_nanogpt.add_argument("--output", default=None, help="CSV output path")
    prep_nanogpt.add_argument("--force", action="store_true", help="Rebuild existing train.bin/val.bin")
    prep_nanogpt.set_defaults(func=command_prepare_nanogpt_data)

    stage1 = subparsers.add_parser("build-stage1", help="Create stage1 coarse-grid manifests.")
    stage1.set_defaults(func=command_build_stage1)

    stage2 = subparsers.add_parser("build-stage2", help="Create stage2 exploration manifests.")
    stage2.set_defaults(func=command_build_stage2)

    reruns = subparsers.add_parser("build-reruns", help="Create rerun manifests for seeds 2/3.")
    reruns.set_defaults(func=command_build_reruns)

    slurm = subparsers.add_parser("make-slurm", help="Generate sbatch scripts from manifests.")
    slurm.add_argument("--stage", default="all", choices=["stage1", "stage2", "rerun", "all"])
    slurm.add_argument("--per-dataset", action="store_true", help="Also generate dataset-specific sbatch files.")
    slurm.set_defaults(func=command_make_slurm)

    run_trial = subparsers.add_parser("run-trial", help="Run one trial from a manifest index.")
    run_trial.add_argument("--manifest", required=True, help="Path to jsonl manifest")
    run_trial.add_argument("--index", required=True, type=int, help="Row index from manifest")
    run_trial.set_defaults(func=command_run_trial)

    local_run = subparsers.add_parser("local-run", help="Run a manifest locally with GPU memory guard.")
    local_run.add_argument("--manifest", required=True, help="Path to jsonl manifest")
    local_run.add_argument("--memory-guard", type=float, default=None, help="Launch guard ratio (e.g. 0.90)")
    local_run.add_argument("--max-parallel", type=int, default=None, help="Max local parallel jobs")
    local_run.set_defaults(func=command_local_run)

    export_csv = subparsers.add_parser("export-csv", help="Export run history to CSV.")
    export_csv.add_argument("--output", default=None, help="Optional custom output CSV path")
    export_csv.add_argument("--history-jsonl", default=None, help="Optional path to history JSONL source")
    export_csv.add_argument("--history-parquet", default=None, help="Optional path to history parquet output")
    export_csv.set_defaults(func=command_export_csv)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
