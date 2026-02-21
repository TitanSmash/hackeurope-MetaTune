from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Callable

from train_config import (
    DISPATCHER_INTERVAL_SECONDS,
    HP_COLS,
    build_default_job_name,
    get_dc_profile,
    resolve_dataset_abs_path,
)
from train_store import TrainRunStore, utc_now_iso

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from training.scheduler import create_training_job, get_training_job_status  # noqa: E402


def parse_hp_json(hp_json: str) -> dict[str, float]:
    try:
        parsed = json.loads(hp_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"hp_json is invalid JSON: {e.msg}") from e

    if not isinstance(parsed, dict):
        raise ValueError("hp_json must decode to an object")

    missing = [key for key in HP_COLS if key not in parsed]
    if missing:
        raise ValueError(f"hp_json missing required keys: {', '.join(missing)}")

    normalized: dict[str, float] = {}
    for key in HP_COLS:
        try:
            normalized[key] = float(parsed[key])
        except (TypeError, ValueError) as e:
            raise ValueError(f"hp_json field '{key}' must be numeric") from e

    if normalized["lora_r"] <= 0:
        raise ValueError("lora_r must be > 0")
    if normalized["learning_rate"] <= 0:
        raise ValueError("learning_rate must be > 0")
    if normalized["lora_dropout"] < 0 or normalized["lora_dropout"] > 1:
        raise ValueError("lora_dropout must be within [0, 1]")

    return normalized


def resolve_dc_id_with_hook(
    requested_dc_id: str | None, context: dict[str, Any] | None = None
) -> str | None:
    if requested_dc_id:
        return requested_dc_id

    try:
        import milp  # local backend module
    except ImportError:
        return None

    resolver = getattr(milp, "resolve_training_dc", None)
    if not callable(resolver):
        return None

    try:
        resolved = resolver(context=context or {})
    except TypeError:
        resolved = resolver()

    if not resolved:
        return None
    return str(resolved)


class TrainingOrchestrator:
    def __init__(
        self,
        store: TrainRunStore,
        poll_interval_seconds: float = DISPATCHER_INTERVAL_SECONDS,
        submit_job_fn: Callable[..., str] = create_training_job,
        get_job_status_fn: Callable[..., dict[str, Any]] = get_training_job_status,
    ) -> None:
        self.store = store
        self.poll_interval_seconds = poll_interval_seconds
        self.submit_job_fn = submit_job_fn
        self.get_job_status_fn = get_job_status_fn
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started = False
        self.enabled = os.getenv("ENABLE_TRAIN_ORCHESTRATOR", "1") == "1"

    def start(self) -> None:
        if self._started or not self.enabled:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, name="train-orchestrator", daemon=True
        )
        self._thread.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._started = False

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.tick()
            except Exception as e:
                print(f"[train-orchestrator] tick failed: {e}")
            self._stop_event.wait(self.poll_interval_seconds)

    def tick(self) -> None:
        active = self.store.get_active_run()
        if active:
            self._reconcile_active_run(active)
            return

        queued = self.store.get_next_queued_run()
        if queued:
            self._dispatch_run(queued)

    def _reconcile_active_run(self, run: dict[str, str]) -> None:
        namespace = run.get("namespace", "")
        k8s_job_name = run.get("k8s_job_name", "")
        if not namespace or not k8s_job_name:
            self.store.update_run(
                run["run_id"],
                status="FAILED",
                finished_at=utc_now_iso(),
                failure_reason="Active run missing namespace or k8s_job_name",
            )
            return

        status = self.get_job_status_fn(namespace=namespace, job_name=k8s_job_name)
        state = status.get("state")
        reason = status.get("reason") or ""

        if state == "RUNNING":
            if run.get("status") == "SUBMITTING":
                self.store.update_run(run["run_id"], status="RUNNING")
            return

        if state == "SUBMITTING":
            return

        if state == "SUCCEEDED":
            self.store.update_run(
                run["run_id"],
                status="SUCCEEDED",
                finished_at=status.get("completion_time") or utc_now_iso(),
                failure_reason="",
            )
            return

        if state in {"FAILED", "NOT_FOUND"}:
            self.store.update_run(
                run["run_id"],
                status="FAILED",
                finished_at=status.get("completion_time") or utc_now_iso(),
                failure_reason=reason or state,
            )
            return

    def _dispatch_run(self, run: dict[str, str]) -> None:
        run_id = run["run_id"]
        try:
            hp_values = parse_hp_json(run["hp_json"])

            dc_id = resolve_dc_id_with_hook(
                run.get("dc_id") or None,
                context={"run_id": run_id, "dataset_rel_path": run.get("dataset_rel_path")},
            )
            if not dc_id:
                raise ValueError("dc_id was not provided and MILP resolver returned nothing")

            profile = get_dc_profile(dc_id)
            dataset_abs_path = resolve_dataset_abs_path(
                profile, run.get("dataset_rel_path") or None
            )
            effective_job_name = run.get("job_name") or build_default_job_name(run_id)

            self.store.update_run(
                run_id,
                status="SUBMITTING",
                started_at=utc_now_iso(),
                dc_id=dc_id,
                namespace=profile["namespace"],
                job_name=effective_job_name,
                failure_reason="",
            )

            k8s_job_name = self.submit_job_fn(
                run_id=run_id,
                dc_id=dc_id,
                namespace=profile["namespace"],
                hp_values=hp_values,
                dataset_abs_path=dataset_abs_path,
                pvc_name=profile["pvc_name"],
                cache_mount_path=profile["cache_mount_path"],
                service_account_name=profile["service_account_name"],
                gpu_count=int(profile["gpu_count"]),
                gpu_resource_name=profile["gpu_resource_name"],
                job_name=effective_job_name,
                image=profile["image"],
                command=profile.get("command"),
                args=profile.get("args"),
                node_selector=profile.get("node_selector"),
                tolerations=profile.get("tolerations"),
                auto_ensure_permissions=True,
            )

            self.store.update_run(
                run_id,
                status="RUNNING",
                k8s_job_name=k8s_job_name,
            )
        except Exception as e:
            self.store.update_run(
                run_id,
                status="FAILED",
                finished_at=utc_now_iso(),
                failure_reason=str(e),
            )
