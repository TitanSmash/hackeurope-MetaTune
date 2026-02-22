from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any
from uuid import uuid4


K8S_NAME_MAX = 63
_K8S_INVALID_CHARS = re.compile(r"[^a-z0-9-]+")
_K8S_DASHES = re.compile(r"-{2,}")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"run_{ts}_{uuid4().hex[:8]}"


def normalize_dataset_rel_path(value: str) -> str:
    rel = value.strip()
    if rel.startswith("/"):
        raise ValueError("dataset_rel_path must be relative")
    if rel in {"", ".", ".."}:
        raise ValueError("dataset_rel_path must not be empty or traversal")
    parts = PurePosixPath(rel).parts
    if ".." in parts:
        raise ValueError("dataset_rel_path must not contain '..'")
    return str(PurePosixPath(rel))


def sanitize_k8s_name(value: str, prefix: str | None = None) -> str:
    raw = value.lower().replace("_", "-")
    if prefix:
        raw = f"{prefix}-{raw}"
    raw = _K8S_INVALID_CHARS.sub("-", raw)
    raw = _K8S_DASHES.sub("-", raw).strip("-")
    if not raw:
        raw = "run"
    if len(raw) > K8S_NAME_MAX:
        raw = raw[:K8S_NAME_MAX].rstrip("-")
    if not raw:
        raw = "run"
    return raw


def safe_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def safe_json_loads(value: str | None, default: Any) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default

