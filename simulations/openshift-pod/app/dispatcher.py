from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

from app.config import AppConfig
from app.store import RunStore
from app.utils import safe_json_dumps, safe_json_loads, sanitize_k8s_name, utc_now_iso

try:
    from app.k8s_jobs import get_training_job_status, submit_training_job
except ModuleNotFoundError:  # pragma: no cover - only for local envs missing k8s deps.
    def submit_training_job(*, run: dict[str, Any], app_config: AppConfig) -> str:
        raise RuntimeError("kubernetes package is required to submit training jobs")

    def get_training_job_status(*, namespace: str, job_name: str) -> dict[str, Any]:
        return {
            "state": "FAILED",
            "reason": "kubernetes package is required to query job status",
            "pod_name": None,
            "completion_time": utc_now_iso(),
        }


TERMINAL_STATES = {"SUCCEEDED", "FAILED"}


class Dispatcher:
    def __init__(
        self,
        *,
        app_config: AppConfig,
        store: RunStore,
        submit_job_fn: Callable[..., str] = submit_training_job,
        get_job_status_fn: Callable[..., dict[str, Any]] = get_training_job_status,
    ) -> None:
        self.app_config = app_config
        self.store = store
        self.submit_job_fn = submit_job_fn
        self.get_job_status_fn = get_job_status_fn
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="openshift-dispatcher",
        )
        self._thread.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self._started = False

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.tick()
            except Exception as e:
                print(f"[dispatcher] tick failed: {e}")
            self._stop_event.wait(self.app_config.dispatch_interval_seconds)

    def tick(self) -> None:
        active = self.store.get_active_run()
        if active:
            self._reconcile_active_run(active)
            return

        queued = self.store.get_next_queued_run()
        if queued:
            self._dispatch_run(queued)

    def _dispatch_run(self, run: dict[str, Any]) -> None:
        run_id = str(run["run_id"])
        output_dir = str(Path(self.app_config.output_root) / run_id)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        desired_job_name = str(
            run.get("job_name") or f"{self.app_config.job_name_prefix}-{run_id}"
        )
        job_name = sanitize_k8s_name(desired_job_name)
        self.store.update_run(
            run_id,
            status="SUBMITTING",
            started_at=utc_now_iso(),
            namespace=self.app_config.namespace,
            job_name=job_name,
            output_dir=output_dir,
            failure_reason="",
        )
        latest = self.store.get_run(run_id) or run
        try:
            k8s_job_name = self.submit_job_fn(
                run=latest,
                app_config=self.app_config,
            )
        except Exception as e:
            self.store.update_run(
                run_id,
                status="FAILED",
                finished_at=utc_now_iso(),
                failure_reason=str(e),
            )
            return

        self.store.update_run(
            run_id,
            status="RUNNING",
            k8s_job_name=k8s_job_name,
        )

    def _reconcile_active_run(self, run: dict[str, Any]) -> None:
        run_id = str(run["run_id"])
        refreshed = self.store.get_run(run_id) or run
        if (
            refreshed.get("callback_received_at")
            and refreshed.get("status") in TERMINAL_STATES
        ):
            return

        namespace = str(refreshed.get("namespace") or self.app_config.namespace)
        k8s_job_name = str(refreshed.get("k8s_job_name") or "")
        if not k8s_job_name:
            self.store.update_run(
                run_id,
                status="FAILED",
                finished_at=utc_now_iso(),
                failure_reason="Active run missing k8s_job_name",
            )
            return

        status = self.get_job_status_fn(namespace=namespace, job_name=k8s_job_name)
        state = status.get("state")
        reason = str(status.get("reason") or "")

        if state == "RUNNING":
            if refreshed.get("status") == "SUBMITTING":
                self.store.update_run(run_id, status="RUNNING")
            return

        if state == "SUBMITTING":
            return

        if state == "SUCCEEDED":
            newest = self.store.get_run(run_id) or refreshed
            if newest.get("callback_received_at") and newest.get("status") in TERMINAL_STATES:
                return
            self._finalize_succeeded_without_callback(newest)
            return

        if state in {"FAILED", "NOT_FOUND"}:
            self.store.update_run(
                run_id,
                status="FAILED",
                finished_at=status.get("completion_time") or utc_now_iso(),
                failure_reason=reason or str(state),
            )

    def _finalize_succeeded_without_callback(self, run: dict[str, Any]) -> None:
        run_id = str(run["run_id"])
        output_dir = Path(str(run.get("output_dir") or (Path(self.app_config.output_root) / run_id)))
        metrics_path = output_dir / "metrics.json"
        if not metrics_path.exists():
            # Wait for callback if metrics are not yet persisted.
            return

        metrics_raw = metrics_path.read_text(encoding="utf-8")
        metrics = safe_json_loads(metrics_raw, default={})
        if not isinstance(metrics, dict):
            metrics = {}
        artifacts = {
            "metrics_path": str(metrics_path),
            "curve_path": str(output_dir / "curve.jsonl"),
            "train_log_path": str(output_dir / "train.log"),
            "run_meta_path": str(output_dir / "run_meta.json"),
            "output_dir": str(output_dir),
        }

        self.store.update_run(
            run_id,
            status="SUCCEEDED",
            finished_at=utc_now_iso(),
            metrics_json=safe_json_dumps(metrics),
            artifacts_json=safe_json_dumps(artifacts),
            failure_reason="",
        )
