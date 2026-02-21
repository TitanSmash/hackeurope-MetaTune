from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from cost_estimation import get_carbon_forecast
from milp import schedule_job_milp, calculate_naive_assignment
from train_config import get_dc_profile
from train_orchestrator import (
    TrainingOrchestrator,
    parse_hp_json,
    resolve_dc_id_with_hook,
)
from train_store import TrainRunStore

app = FastAPI()
run_store = TrainRunStore()
train_orchestrator = TrainingOrchestrator(run_store)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Maps internal DataFrame names (after .replace("_", " ")) to display metadata
DC_META: dict[str, dict] = {
    "Crusoe Abilene TX": {
        "id": "crusoe-tx",
        "label": "Abilene, TX",
        "location": "US Central",
        "lat": 32.4487,
        "lng": -99.7331,
    },
    "AWS US East VA": {
        "id": "aws-va",
        "label": "Virginia, VA",
        "location": "US East",
        "lat": 39.0438,
        "lng": -77.4874,
    },
    "Google Hamina FI": {
        "id": "google-fi",
        "label": "Hamina, FI",
        "location": "EU North",
        "lat": 60.5693,
        "lng": 27.1938,
    },
}


class CreateTrainingRunRequest(BaseModel):
    hp_json: str = Field(..., min_length=2)
    dc_id: str | None = None
    dataset_rel_path: str | None = None
    job_name: str | None = None


@app.on_event("startup")
def _startup_orchestrator() -> None:
    train_orchestrator.start()


@app.on_event("shutdown")
def _shutdown_orchestrator() -> None:
    train_orchestrator.stop()


@app.get("/api/schedule")
def get_schedule(
    job_hours: int = Query(4, ge=1, le=20),
    job_power_kw: float = Query(15.0, ge=0.1),
    deadline_hours: int = Query(22, ge=1, le=23),
):
    _pivot, raw_df = get_carbon_forecast()

    data_centers = []
    for dc_name, meta in DC_META.items():
        dc_df = raw_df[raw_df["Data_Center"] == dc_name].sort_values("Time_UTC")
        forecast = [
            {
                "hour": row["Time_UTC"].strftime("%H:00"),
                "carbon_gco2": row["Effective_Carbon_gCO2"],
            }
            for _, row in dc_df.iterrows()
        ]
        avg = float(dc_df["Effective_Carbon_gCO2"].mean()) if len(dc_df) > 0 else 0.0
        data_centers.append(
            {
                **meta,
                "name": dc_name,
                "forecast": forecast,
                "avg_carbon": round(avg, 1),
            }
        )

    optimal = None
    try:
        opt_dc, opt_start, opt_cost = schedule_job_milp(
            job_hours, deadline_hours, job_power_kw, raw_df
        )
        naive_cost = calculate_naive_assignment(job_hours, job_power_kw, raw_df)

        if opt_dc and opt_start and opt_cost is not None:
            meta = DC_META.get(opt_dc, {})
            savings_pct = (
                ((naive_cost - opt_cost) / naive_cost * 100)
                if naive_cost and naive_cost > 0
                else 0.0
            )
            optimal = {
                "data_center_id": meta.get("id", ""),
                "data_center_name": opt_dc,
                "start_time": opt_start.strftime("%H:00 GMT"),
                "total_co2_kg": round(opt_cost, 4),
                "naive_co2_kg": round(naive_cost, 4) if naive_cost else None,
                "savings_pct": round(savings_pct, 1),
            }
    except Exception as e:
        print(f"MILP solver failed: {e}")

    return {"data_centers": data_centers, "optimal": optimal}


@app.post("/api/train/runs")
def create_training_run(payload: CreateTrainingRunRequest):
    try:
        effective_hp = parse_hp_json(payload.hp_json)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    resolved_dc_id = resolve_dc_id_with_hook(
        payload.dc_id,
        context={"dataset_rel_path": payload.dataset_rel_path},
    )
    if not resolved_dc_id:
        raise HTTPException(
            status_code=400,
            detail="dc_id missing and MILP resolver hook did not return a datacentre",
        )

    try:
        profile = get_dc_profile(resolved_dc_id)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    row = run_store.create_run(
        hp_json=payload.hp_json,
        dc_id=resolved_dc_id,
        namespace=profile["namespace"],
        dataset_rel_path=payload.dataset_rel_path,
        job_name=payload.job_name,
    )

    return {
        "run_id": row["run_id"],
        "status": row["status"],
        "effective_hp": effective_hp,
        "dc_id": row["dc_id"],
        "queue_position": int(row["queue_position"]),
    }


@app.get("/api/train/runs/{run_id}")
def get_training_run(run_id: str):
    run = run_store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run_id '{run_id}' not found")
    return run


@app.get("/api/train/runs")
def list_training_runs(
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
):
    return {"runs": run_store.list_runs(status=status, limit=limit)}
