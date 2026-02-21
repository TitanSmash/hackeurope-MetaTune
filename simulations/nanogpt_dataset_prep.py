from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from simulations.io_utils import ensure_dir, iter_jsonl


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def _dataset_name(dataset_key: str, train_path: Path, val_path: Path, prefix: str = "metatune") -> str:
    signature = f"{dataset_key}|{train_path.resolve()}|{val_path.resolve()}"
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{_slugify(dataset_key)}_{digest}"


def _source_signature(train_jsonl: Path, val_jsonl: Path) -> str:
    train_stat = train_jsonl.stat()
    val_stat = val_jsonl.stat()
    payload = (
        f"{train_jsonl.resolve()}|{train_stat.st_size}|{int(train_stat.st_mtime_ns)}|"
        f"{val_jsonl.resolve()}|{val_stat.st_size}|{int(val_stat.st_mtime_ns)}"
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _iter_texts(jsonl_path: Path) -> Iterable[str]:
    for row in iter_jsonl(jsonl_path):
        text = row.get("text")
        if isinstance(text, str) and text.strip():
            yield text


def _encode_to_bin(src_jsonl: Path, dst_bin: Path, encoding_name: str = "gpt2") -> int:
    import tiktoken

    enc = tiktoken.get_encoding(encoding_name)
    eot = enc.eot_token
    tokens_written = 0
    with dst_bin.open("wb") as handle:
        for text in _iter_texts(src_jsonl):
            ids = enc.encode_ordinary(text)
            ids.append(eot)
            arr = np.asarray(ids, dtype=np.uint16)
            arr.tofile(handle)
            tokens_written += int(arr.size)
    return tokens_written


def prepare_one_dataset_for_nanogpt(
    *,
    repo_dir: Path,
    dataset_id: str,
    dataset_key: str,
    train_jsonl: Path,
    val_jsonl: Path,
    dataset_prefix: str = "metatune",
    force: bool = False,
) -> Dict[str, object]:
    data_root = ensure_dir(repo_dir / "data")
    dataset_name = _dataset_name(dataset_key, train_jsonl, val_jsonl, prefix=dataset_prefix)
    dataset_dir = ensure_dir(data_root / dataset_name)
    train_bin = dataset_dir / "train.bin"
    val_bin = dataset_dir / "val.bin"
    stats_json = dataset_dir / "stats.json"
    current_source_sig = _source_signature(train_jsonl, val_jsonl)

    if train_bin.exists() and val_bin.exists() and stats_json.exists() and not force:
        stats = json.loads(stats_json.read_text(encoding="utf-8"))
        if stats.get("source_signature") != current_source_sig:
            force = True
        else:
            return {
                "dataset_id": dataset_id,
                "dataset_key": dataset_key,
                "dataset_name": dataset_name,
                "dataset_dir": str(dataset_dir),
                "train_bin": str(train_bin),
                "val_bin": str(val_bin),
                "train_tokens": int(stats.get("train_tokens", 0)),
                "val_tokens": int(stats.get("val_tokens", 0)),
                "prepared": False,
            }

    if force:
        for target in (train_bin, val_bin, stats_json):
            if target.exists():
                target.unlink()

    train_tokens = _encode_to_bin(train_jsonl, train_bin, encoding_name="gpt2")
    val_tokens = _encode_to_bin(val_jsonl, val_bin, encoding_name="gpt2")
    stats = {
        "tokenizer": "gpt2",
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "source_signature": current_source_sig,
    }
    stats_json.write_text(json.dumps(stats, ensure_ascii=True, indent=2), encoding="utf-8")
    return {
        "dataset_id": dataset_id,
        "dataset_key": dataset_key,
        "dataset_name": dataset_name,
        "dataset_dir": str(dataset_dir),
        "train_bin": str(train_bin),
        "val_bin": str(val_bin),
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "prepared": True,
    }


def prepare_many_datasets_for_nanogpt(
    *,
    repo_dir: Path,
    prepared_metadata_rows: List[Dict[str, object]],
    dataset_prefix: str = "metatune",
    force: bool = False,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for row in prepared_metadata_rows:
        result = prepare_one_dataset_for_nanogpt(
            repo_dir=repo_dir,
            dataset_id=str(row["dataset_id"]),
            dataset_key=str(row["dataset_key"]),
            train_jsonl=Path(str(row["train_path"])),
            val_jsonl=Path(str(row["val_path"])),
            dataset_prefix=dataset_prefix,
            force=force,
        )
        results.append(result)
    return results
