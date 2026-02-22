from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query

from app.config import AppConfig
from app.dispatcher import Dispatcher
from app.schemas import CallbackPayload, CreateRunRequest
from app.store import RunStore
from app.utils import safe_json_dumps, safe_json_loads, utc_now_iso


def _serialize_run(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload = dict(row)
    payload["hp"] = safe_json_loads(str(payload.get("hp_json", "")), default={})
    payload["metrics"] = safe_json_loads(str(payload.get("metrics_json", "")), default={})
    payload["artifacts"] = safe_json_loads(
        str(payload.get("artifacts_json", "")),
        default={},
    )
    return payload


def create_app(
    *,
    app_config: AppConfig | None = None,
    store: RunStore | None = None,
    dispatcher: Dispatcher | None = None,
) -> FastAPI:
    config = app_config or AppConfig.from_env()
    run_store = store or RunStore(config.db_path)
    run_dispatcher = dispatcher or Dispatcher(
        app_config=config,
        store=run_store,
    )

    app = FastAPI(title="MetaTune OpenShift Scheduler")
    app.state.config = config
    app.state.store = run_store
    app.state.dispatcher = run_dispatcher

    @app.on_event("startup")
    def _startup() -> None:
        if config.enable_dispatcher:
            run_dispatcher.start()

    @app.on_event("shutdown")
    def _shutdown() -> None:
        run_dispatcher.stop()
        run_store.close()

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "namespace": config.namespace,
            "db_path": str(config.db_path),
            "dispatcher_enabled": config.enable_dispatcher,
        }

    @app.post("/api/v1/runs")
    def create_run(payload: CreateRunRequest) -> dict[str, Any]:
        dataset_dir = Path(config.datasets_root) / payload.dataset_rel_path
        train_bin = dataset_dir / "train.bin"
        val_bin = dataset_dir / "val.bin"
        if not dataset_dir.exists() or not train_bin.exists() or not val_bin.exists():
            raise HTTPException(
                status_code=400,
                detail=(
                    "dataset_rel_path must point to a folder under DATASETS_ROOT "
                    "containing train.bin and val.bin"
                ),
            )
        hp_json = safe_json_dumps(payload.hp.model_dump())
        row = run_store.enqueue_run(
            namespace=config.namespace,
            model=payload.model,
            dataset_rel_path=payload.dataset_rel_path,
            hp_json=hp_json,
            token_budget=payload.token_budget,
            seq_len=payload.seq_len,
            seed=payload.seed,
            job_name=payload.job_name or "",
            output_dir="",
        )
        final_output_dir = str(Path(config.output_root) / str(row["run_id"]))
        row = run_store.update_run(str(row["run_id"]), output_dir=final_output_dir) or row
        return _serialize_run(row) or {}

    @app.get("/api/v1/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        row = run_store.get_run(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"run_id '{run_id}' not found")
        return _serialize_run(row) or {}

    @app.get("/api/v1/runs")
    def list_runs(
        status: str | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=500),
    ) -> dict[str, Any]:
        rows = run_store.list_runs(status=status, limit=limit)
        return {"runs": [_serialize_run(row) for row in rows]}

    @app.post("/internal/runs/{run_id}/complete")
    def complete_run(run_id: str, payload: CallbackPayload) -> dict[str, Any]:
        row = run_store.get_run(run_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"run_id '{run_id}' not found")
        now = utc_now_iso()
        updates = {
            "status": payload.status,
            "finished_at": now,
            "callback_received_at": now,
            "metrics_json": safe_json_dumps(payload.metrics),
            "artifacts_json": safe_json_dumps(payload.artifacts),
            "failure_reason": payload.error_message
            if payload.status == "FAILED"
            else "",
        }
        updated = run_store.update_run(run_id, **updates)
        return _serialize_run(updated) or {}

    return app


app = create_app()
