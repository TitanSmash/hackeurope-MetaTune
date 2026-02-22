from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from simulations.io_utils import (
    append_jsonl,
    ensure_dir,
    slugify_dataset_id,
    write_json,
)
from simulations.metafeatures import extract_metafeatures_from_jsonl


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(_safe_text(item) for item in value if _safe_text(item))
    if isinstance(value, dict):
        if "text" in value:
            return _safe_text(value["text"])
        if "content" in value:
            return _safe_text(value["content"])
        if "value" in value:
            return _safe_text(value["value"])
        return "\n".join(f"{k}: {_safe_text(v)}" for k, v in value.items())
    return str(value)


def _format_chat_pairs(instruction: str, output: str, user_tag: str = "<|user|>", assistant_tag: str = "<|assistant|>") -> str:
    instruction = instruction.strip()
    output = output.strip()
    if not instruction:
        instruction = "Continue the text."
    if not output:
        return ""
    return f"{user_tag}\n{instruction}\n{assistant_tag}\n{output}"


def _format_messages(messages: Iterable[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for message in messages:
        role = _safe_text(message.get("role") or message.get("from") or "user").lower()
        content = _safe_text(message.get("content") or message.get("value") or message.get("text"))
        if not content:
            continue
        if role in {"assistant", "gpt", "bot", "model"}:
            prefix = "<|assistant|>"
        elif role in {"system"}:
            prefix = "<|system|>"
        else:
            prefix = "<|user|>"
        lines.append(prefix)
        lines.append(content)
    return "\n".join(lines).strip()


def normalize_record_to_text(record: Dict[str, Any], format_policy: str = "instruction_chat") -> str:
    if "messages" in record and isinstance(record["messages"], list):
        result = _format_messages(record["messages"])
        if result:
            return result
    if "conversations" in record and isinstance(record["conversations"], list):
        result = _format_messages(record["conversations"])
        if result:
            return result

    instruction = _safe_text(record.get("instruction"))
    input_text = _safe_text(record.get("input"))
    output = _safe_text(record.get("output") or record.get("response") or record.get("completion") or record.get("answer"))
    prompt = _safe_text(record.get("prompt") or record.get("question"))
    text = _safe_text(record.get("text"))

    if input_text:
        instruction = (instruction + "\n\n" + input_text).strip()

    if instruction and output:
        return _format_chat_pairs(instruction, output)
    if prompt and output:
        return _format_chat_pairs(prompt, output)
    if text:
        if format_policy == "instruction_chat":
            return f"<|assistant|>\n{text}"
        return text
    return ""


def _hash_to_val_bucket(text: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()
    prefix = digest[:8]
    as_int = int(prefix, 16)
    return as_int / 0xFFFFFFFF


def _iter_split(dataset, split_name: str) -> Iterator[Dict[str, Any]]:
    split = dataset[split_name]
    for row in split:
        if isinstance(row, dict):
            yield row


def _pick_train_split(split_names: List[str]) -> str:
    for candidate in ("train", "training"):
        if candidate in split_names:
            return candidate
    if not split_names:
        raise ValueError("No split names found in dataset.")
    return split_names[0]


def _pick_val_split(split_names: List[str]) -> Optional[str]:
    for candidate in ("validation", "val", "dev", "test"):
        if candidate in split_names:
            return candidate
    return None


def _text_bytes(text: str) -> int:
    # +1 accounts for newline in JSONL row writing.
    return len(text.encode("utf-8")) + 1


def _write_rows(
    rows: Iterable[Dict[str, Any]],
    destination: Path,
    format_policy: str,
    max_rows: Optional[int] = None,
    max_bytes: Optional[int] = None,
) -> Tuple[int, int]:
    if destination.exists():
        destination.unlink()
    count = 0
    bytes_written = 0
    for row in rows:
        text = normalize_record_to_text(row, format_policy=format_policy)
        if not text:
            continue
        row_bytes = _text_bytes(text)
        if max_bytes is not None and (bytes_written + row_bytes) > max_bytes:
            break
        append_jsonl(destination, {"text": text})
        count += 1
        bytes_written += row_bytes
        if max_rows is not None and count >= max_rows:
            break
    return count, bytes_written


def _split_train_rows(
    rows: Iterable[Dict[str, Any]],
    seed: int,
    val_ratio: float,
    train_destination: Path,
    val_destination: Path,
    format_policy: str,
    max_train_rows: Optional[int] = None,
    max_val_rows: Optional[int] = None,
    max_train_bytes: Optional[int] = None,
    max_val_bytes: Optional[int] = None,
) -> Tuple[int, int, int, int]:
    if train_destination.exists():
        train_destination.unlink()
    if val_destination.exists():
        val_destination.unlink()

    train_count = 0
    val_count = 0
    train_bytes = 0
    val_bytes = 0
    for row in rows:
        text = normalize_record_to_text(row, format_policy=format_policy)
        if not text:
            continue
        row_bytes = _text_bytes(text)
        bucket = _hash_to_val_bucket(text, seed=seed)
        if bucket < val_ratio:
            can_add_row = max_val_rows is None or val_count < max_val_rows
            can_add_bytes = max_val_bytes is None or (val_bytes + row_bytes) <= max_val_bytes
            if can_add_row and can_add_bytes:
                append_jsonl(val_destination, {"text": text})
                val_count += 1
                val_bytes += row_bytes
        else:
            can_add_row = max_train_rows is None or train_count < max_train_rows
            can_add_bytes = max_train_bytes is None or (train_bytes + row_bytes) <= max_train_bytes
            if can_add_row and can_add_bytes:
                append_jsonl(train_destination, {"text": text})
                train_count += 1
                train_bytes += row_bytes
        if max_train_rows is not None and max_val_rows is not None:
            if train_count >= max_train_rows and val_count >= max_val_rows:
                break
        if max_train_bytes is not None and max_val_bytes is not None:
            if train_bytes >= max_train_bytes and val_bytes >= max_val_bytes:
                break
    return train_count, val_count, train_bytes, val_bytes


def prepare_one_dataset(
    download_metadata: Dict[str, Any],
    processed_root: Path | str,
    format_policy: str = "instruction_chat",
    split_policy: str = "native_then_95_5",
    val_ratio: float = 0.05,
    seed: int = 42,
    max_train_rows: Optional[int] = None,
    max_val_rows: Optional[int] = None,
    max_train_bytes: Optional[int] = None,
    max_val_bytes: Optional[int] = None,
    metafeature_sample_tokens: int = 2_000_000,
) -> Dict[str, Any]:
    try:
        from datasets import load_from_disk
    except ImportError as exc:  # pragma: no cover - depends on local env.
        raise RuntimeError(
            "Missing `datasets` package. Install with `pip install datasets`."
        ) from exc

    dataset_id = download_metadata["dataset_id"]
    dataset_key = download_metadata["dataset_key"]
    raw_path = Path(download_metadata["raw_path"])
    processed_dir = ensure_dir(Path(processed_root) / slugify_dataset_id(dataset_key))
    train_path = processed_dir / "train.jsonl"
    val_path = processed_dir / "val.jsonl"

    dataset = load_from_disk(str(raw_path))
    split_names = list(dataset.keys())
    train_split_name = _pick_train_split(split_names)
    val_split_name = _pick_val_split(split_names) if split_policy == "native_then_95_5" else None

    if val_split_name:
        train_count, train_bytes = _write_rows(
            _iter_split(dataset, train_split_name),
            train_path,
            format_policy=format_policy,
            max_rows=max_train_rows,
            max_bytes=max_train_bytes,
        )
        val_count, val_bytes = _write_rows(
            _iter_split(dataset, val_split_name),
            val_path,
            format_policy=format_policy,
            max_rows=max_val_rows,
            max_bytes=max_val_bytes,
        )
        split_source = f"native:{train_split_name}/{val_split_name}"
    else:
        train_count, val_count, train_bytes, val_bytes = _split_train_rows(
            _iter_split(dataset, train_split_name),
            seed=seed,
            val_ratio=val_ratio,
            train_destination=train_path,
            val_destination=val_path,
            format_policy=format_policy,
            max_train_rows=max_train_rows,
            max_val_rows=max_val_rows,
            max_train_bytes=max_train_bytes,
            max_val_bytes=max_val_bytes,
        )
        split_source = f"hashed:{train_split_name}->95_5"

    metafeatures = extract_metafeatures_from_jsonl(
        train_path,
        max_tokens=metafeature_sample_tokens,
    )
    metadata = {
        "dataset_id": dataset_id,
        "dataset_key": dataset_key,
        "processed_dir": str(processed_dir),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "format_policy": format_policy,
        "split_policy": split_policy,
        "split_source": split_source,
        "train_rows": train_count,
        "val_rows": val_count,
        "train_bytes": train_bytes,
        "val_bytes": val_bytes,
        "metafeatures": metafeatures,
    }
    write_json(processed_dir / "metadata.json", metadata)
    return metadata


def prepare_datasets(
    download_results: Iterable[Dict[str, Any]],
    processed_root: Path | str,
    format_policy: str = "instruction_chat",
    split_policy: str = "native_then_95_5",
    val_ratio: float = 0.05,
    seed: int = 42,
    max_train_rows: Optional[int] = None,
    max_val_rows: Optional[int] = None,
    metafeature_sample_tokens: int = 2_000_000,
) -> List[Dict[str, Any]]:
    results = []
    for entry in download_results:
        result = prepare_one_dataset(
            download_metadata=entry,
            processed_root=processed_root,
            format_policy=format_policy,
            split_policy=split_policy,
            val_ratio=val_ratio,
            seed=seed,
            max_train_rows=max_train_rows,
            max_val_rows=max_val_rows,
            metafeature_sample_tokens=metafeature_sample_tokens,
        )
        results.append(result)
    return results
