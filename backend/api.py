from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from cost_estimation import get_carbon_forecast
from milp import schedule_job_milp, calculate_naive_assignment

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
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
