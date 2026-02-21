from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only without PyYAML installed.
    yaml = None


def ensure_dir(path: Path | str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def slugify_dataset_id(dataset_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", dataset_id).strip("_").lower()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: Path | str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install with `pip install pyyaml`.")
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML object at root in {path}.")
    return data


def write_json(path: Path | str, payload: Dict[str, Any]) -> None:
    destination = Path(path)
    ensure_dir(destination.parent)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)


def read_json(path: Path | str, default: Any | None = None) -> Any:
    target = Path(path)
    if not target.exists():
        return default
    with target.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def append_jsonl(path: Path | str, row: Dict[str, Any]) -> None:
    destination = Path(path)
    ensure_dir(destination.parent)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def iter_jsonl(path: Path | str) -> Iterator[Dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return
    # `utf-8-sig` tolerates BOM-prefixed files often produced by shell tools.
    with source.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def read_jsonl(path: Path | str) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def count_jsonl_rows(path: Path | str) -> int:
    return sum(1 for _ in iter_jsonl(path))


def write_jsonl(path: Path | str, rows: Iterable[Dict[str, Any]]) -> None:
    destination = Path(path)
    ensure_dir(destination.parent)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
