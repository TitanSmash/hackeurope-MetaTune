from __future__ import annotations

import json
import os
import posixpath
from pathlib import Path, PurePosixPath
from typing import Any

HP_COLS = ["lora_r", "learning_rate", "lora_dropout"]
RUNS_CSV_PATH = Path(__file__).resolve().with_name("train_runs.csv")
DISPATCHER_INTERVAL_SECONDS = float(os.getenv("TRAIN_DISPATCHER_INTERVAL_SECONDS", "5"))

DEFAULT_NAMESPACE = os.getenv("TRAINING_NAMESPACE", "hack-europe-team-i")
DEFAULT_CACHE_MOUNT_PATH = os.getenv("TRAINING_CACHE_MOUNT_PATH", "/mnt/cache")
DEFAULT_DATASET_ROOT_REL = os.getenv("TRAINING_DATASET_ROOT_REL", "datasets")
DEFAULT_MODEL_CACHE_REL = os.getenv("TRAINING_MODEL_CACHE_REL", "models")
DEFAULT_GPU_RESOURCE = os.getenv("TRAINING_GPU_RESOURCE", "nvidia.com/gpu")
DEFAULT_GPU_COUNT = int(os.getenv("TRAINING_GPU_COUNT", "1"))
DEFAULT_SERVICE_ACCOUNT = os.getenv("TRAINING_SERVICE_ACCOUNT", "green-scheduler-sa")
DEFAULT_TRAIN_IMAGE = os.getenv(
    "TRAINING_IMAGE",
    (
        "image-registry.openshift-image-registry.svc:5000/"
        "hack-europe-team-i/train@sha256:02ffe0bfec264b4c434b27dda3ba36ca691a7bbe7443c22a66a8605c9191561c"
    ),
)


def _default_profiles() -> dict[str, dict[str, Any]]:
    default_dc_id = os.getenv("TRAINING_DEFAULT_DC_ID", "default")
    default_pvc = os.getenv("TRAINING_DEFAULT_PVC_NAME", "train-cache-pvc")
    return {
        default_dc_id: {
            "namespace": DEFAULT_NAMESPACE,
            "pvc_name": default_pvc,
            "gpu_count": DEFAULT_GPU_COUNT,
            "gpu_resource_name": DEFAULT_GPU_RESOURCE,
            "cache_mount_path": DEFAULT_CACHE_MOUNT_PATH,
            "dataset_root_rel": DEFAULT_DATASET_ROOT_REL,
            "model_cache_rel": DEFAULT_MODEL_CACHE_REL,
            "service_account_name": DEFAULT_SERVICE_ACCOUNT,
            "image": DEFAULT_TRAIN_IMAGE,
            "node_selector": None,
            "tolerations": None,
        }
    }


def _load_profiles() -> dict[str, dict[str, Any]]:
    raw_profiles = os.getenv("TRAINING_DC_PROFILES_JSON")
    if not raw_profiles:
        return _default_profiles()

    parsed = json.loads(raw_profiles)
    if not isinstance(parsed, dict) or not parsed:
        raise ValueError("TRAINING_DC_PROFILES_JSON must be a non-empty JSON object")

    return parsed


DC_PROFILES = _load_profiles()


def get_dc_profile(dc_id: str) -> dict[str, Any]:
    if dc_id not in DC_PROFILES:
        raise KeyError(f"Unknown dc_id '{dc_id}'")

    profile = dict(DC_PROFILES[dc_id])
    profile.setdefault("namespace", DEFAULT_NAMESPACE)
    profile.setdefault("gpu_count", DEFAULT_GPU_COUNT)
    profile.setdefault("gpu_resource_name", DEFAULT_GPU_RESOURCE)
    profile.setdefault("cache_mount_path", DEFAULT_CACHE_MOUNT_PATH)
    profile.setdefault("dataset_root_rel", DEFAULT_DATASET_ROOT_REL)
    profile.setdefault("model_cache_rel", DEFAULT_MODEL_CACHE_REL)
    profile.setdefault("service_account_name", DEFAULT_SERVICE_ACCOUNT)
    profile.setdefault("image", DEFAULT_TRAIN_IMAGE)
    profile.setdefault("node_selector", None)
    profile.setdefault("tolerations", None)

    if not profile.get("namespace"):
        raise ValueError(f"dc_id '{dc_id}' missing namespace")
    if not profile.get("pvc_name"):
        raise ValueError(f"dc_id '{dc_id}' missing pvc_name")

    return profile


def resolve_dataset_abs_path(profile: dict[str, Any], dataset_rel_path: str | None) -> str:
    rel = (dataset_rel_path or "default").strip()
    if rel.startswith("/"):
        raise ValueError("dataset_rel_path must be relative")
    if rel in {"", ".", ".."}:
        raise ValueError("dataset_rel_path must not be empty or traversal")

    parts = PurePosixPath(rel).parts
    if ".." in parts:
        raise ValueError("dataset_rel_path must not contain '..'")

    return posixpath.join(
        profile["cache_mount_path"],
        profile["dataset_root_rel"],
        rel,
    )


def build_default_job_name(run_id: str) -> str:
    sanitized = run_id.lower().replace("_", "-")
    return f"metatune-train-{sanitized[:40]}"
