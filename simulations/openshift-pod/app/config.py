from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_TRAINER_COMMAND = ["python", "simulations/openshift-pod/trainer/runner.py"]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_json(name: str) -> Any | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{name} must be valid JSON: {e.msg}") from e


def _env_json_str_dict(name: str) -> dict[str, str] | None:
    value = _env_json(name)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    casted: dict[str, str] = {}
    for key, item in value.items():
        casted[str(key)] = str(item)
    return casted


def _env_json_list_of_dict(name: str) -> list[dict[str, Any]] | None:
    value = _env_json(name)
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    casted: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"{name} entries must be JSON objects")
        casted.append(dict(item))
    return casted


def _env_json_str_list(name: str, default: list[str] | None) -> list[str] | None:
    value = _env_json(name)
    if value is None:
        return default
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return [str(item) for item in value]


@dataclass(frozen=True)
class AppConfig:
    namespace: str
    data_root: Path
    datasets_root: Path
    output_root: Path
    db_path: Path
    dispatch_interval_seconds: float
    max_concurrent_jobs: int
    shared_pvc_name: str
    trainer_image: str
    trainer_image_pull_policy: str
    trainer_service_account_name: str
    gpu_resource_name: str
    gpu_count: int
    scheduler_service_name: str
    scheduler_service_port: int
    callback_url_template: str
    trainer_command: list[str] | None
    trainer_args: list[str] | None
    node_selector: dict[str, str] | None
    tolerations: list[dict[str, Any]] | None
    job_name_prefix: str
    enable_dispatcher: bool

    @classmethod
    def from_env(cls) -> "AppConfig":
        namespace = os.getenv("NAMESPACE", "hack-europe-team-i")
        data_root = Path(os.getenv("DATA_ROOT", "/data"))
        datasets_root = Path(os.getenv("DATASETS_ROOT", str(data_root / "datasets")))
        output_root = Path(os.getenv("OUTPUT_ROOT", str(data_root / "outputs")))
        db_path = Path(os.getenv("DB_PATH", str(data_root / "db" / "scheduler.sqlite")))
        dispatch_interval_seconds = float(os.getenv("DISPATCH_INTERVAL_SECONDS", "5"))
        max_concurrent_jobs = int(os.getenv("MAX_CONCURRENT_JOBS", "1"))
        # For this implementation we enforce one-at-a-time execution.
        max_concurrent_jobs = 1 if max_concurrent_jobs > 1 else max(1, max_concurrent_jobs)
        shared_pvc_name = os.getenv("SHARED_PVC_NAME", "train-cache-pvc")
        trainer_image = os.getenv("TRAINER_IMAGE", "REPLACE_TRAINER_IMAGE")
        trainer_image_pull_policy = os.getenv("TRAINER_IMAGE_PULL_POLICY", "IfNotPresent")
        trainer_service_account_name = os.getenv(
            "TRAINER_SERVICE_ACCOUNT_NAME", "metatune-scheduler-sa"
        )
        gpu_resource_name = os.getenv("GPU_RESOURCE_NAME", "nvidia.com/gpu")
        gpu_count = max(1, int(os.getenv("GPU_COUNT", "1")))
        scheduler_service_name = os.getenv("SCHEDULER_SERVICE_NAME", "metatune-scheduler")
        scheduler_service_port = int(os.getenv("SCHEDULER_SERVICE_PORT", "8000"))
        default_callback = (
            f"http://{scheduler_service_name}:{scheduler_service_port}"
            "/internal/runs/{run_id}/complete"
        )
        callback_url_template = os.getenv("CALLBACK_URL_TEMPLATE", default_callback)
        trainer_command = _env_json_str_list("TRAINER_COMMAND_JSON", DEFAULT_TRAINER_COMMAND)
        trainer_args = _env_json_str_list("TRAINER_ARGS_JSON", None)
        node_selector = _env_json_str_dict("NODE_SELECTOR_JSON")
        tolerations = _env_json_list_of_dict("TOLERATIONS_JSON")
        job_name_prefix = os.getenv("JOB_NAME_PREFIX", "metatune-train")
        enable_dispatcher = _env_bool("ENABLE_DISPATCHER", True)

        return cls(
            namespace=namespace,
            data_root=data_root,
            datasets_root=datasets_root,
            output_root=output_root,
            db_path=db_path,
            dispatch_interval_seconds=dispatch_interval_seconds,
            max_concurrent_jobs=max_concurrent_jobs,
            shared_pvc_name=shared_pvc_name,
            trainer_image=trainer_image,
            trainer_image_pull_policy=trainer_image_pull_policy,
            trainer_service_account_name=trainer_service_account_name,
            gpu_resource_name=gpu_resource_name,
            gpu_count=gpu_count,
            scheduler_service_name=scheduler_service_name,
            scheduler_service_port=scheduler_service_port,
            callback_url_template=callback_url_template,
            trainer_command=trainer_command,
            trainer_args=trainer_args,
            node_selector=node_selector,
            tolerations=tolerations,
            job_name_prefix=job_name_prefix,
            enable_dispatcher=enable_dispatcher,
        )
