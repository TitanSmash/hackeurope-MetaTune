import pandas as pd
import pulp
import json
from cost_estimation import get_carbon_forecast


def solve_preemptible_batch_scheduling(
    jobs_csv="jobs.csv", dcs_csv="datacenters.csv", time_limit_seconds=60
):
    print("🌍 Fetching 7-Day Weather Forecast & Loading Jobs...")

    # 1. Fetch Dynamic 7-Day Carbon Forecast
    try:
        pivot_df, raw_df = get_carbon_forecast(csv_path=dcs_csv, days_ahead=7)
    except Exception as e:
        print(f"❌ Error fetching weather forecast: {e}")
        return

    # 2. Load Jobs and Data Center Capacities
    try:
        jobs_df = pd.read_csv(jobs_csv)
        dcs_df = pd.read_csv(dcs_csv)
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find CSV files. {e}")
        return

    # Extract Job parameters
    jobs = jobs_df["job id"].tolist()
    est_times = jobs_df.set_index("job id")["estimated time (in hours)"].to_dict()
    ram_reqs = jobs_df.set_index("job id")["required ram"].to_dict()
    deadlines = jobs_df.set_index("job id")["deadline (in hours)"].to_dict()

    # Extract Data Center parameters
    dcs = dcs_df["data center name"].tolist()
    ram_caps = dcs_df.set_index("data center name")["ram capacity"].to_dict()

    # 3. Time Horizons & Cost Mapping
    times_list = sorted(raw_df["Time_UTC"].unique())
    time_to_idx = {t: i for i, t in enumerate(times_list)}
    max_forecast_hour = len(times_list) - 1
    times = list(range(max_forecast_hour + 1))

    C = {}
    for _, row in raw_df.iterrows():
        d = row["Data_Center"]
        t_idx = time_to_idx[row["Time_UTC"]]
        C[(d, t_idx)] = float(row["Effective_Carbon_gCO2"])

    # 4. Initialize MILP Problem
    prob = pulp.LpProblem("Preemptible_Green_Scheduling", pulp.LpMinimize)

    # Decision Variable: 1 if Job j is running at DC d at time t
    x = pulp.LpVariable.dicts(
        "x_jdt", [(j, d, t) for j in jobs for d in dcs for t in times], cat="Binary"
    )

    # 5. Objective Function: Power Draw (RAM) * Carbon Intensity * Run State
    prob += pulp.lpSum(
        [
            ram_reqs[j] * C[(d, t)] * x[(j, d, t)]
            for j in jobs
            for d in dcs
            for t in times
        ]
    )

    # 6. Constraints
    for j in jobs:
        job_deadline = min(deadlines[j], max_forecast_hour)

        # A. Total Execution Time constraint
        prob += (
            pulp.lpSum([x[(j, d, t)] for d in dcs for t in range(job_deadline + 1)])
            == est_times[j]
        )

        # B. Deadline Constraint
        for d in dcs:
            for t in range(job_deadline + 1, max_forecast_hour + 1):
                prob += x[(j, d, t)] == 0

        # C. Concurrency Constraint
        for t in times:
            prob += pulp.lpSum([x[(j, d, t)] for d in dcs]) <= 1

    # D. RAM Capacity Constraint
    for d in dcs:
        for t in times:
            prob += (
                pulp.lpSum([ram_reqs[j] * x[(j, d, t)] for j in jobs]) <= ram_caps[d]
            )

    # 7. Solve the Problem
    print("🚀 Solving Preemptible MILP (Allowing Checkpointing/Pausing)...")
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_seconds))

    # 8. Extract Results & Build JSONs
    if pulp.LpStatus[prob.status] == "Optimal":
        print("\n✅ OPTIMAL PAUSEABLE SCHEDULE FOUND!\n")

        # --- JSON 1: CO2 Emissions per Data Center per Hour ---
        json1_data = {}
        for d in dcs:
            json1_data[d] = []
            for t in times:
                json1_data[d].append(
                    {
                        "time_utc": times_list[t].strftime("%Y-%m-%d %H:00 UTC"),
                        "co2_emission": C[(d, t)],
                    }
                )

        with open("datacenter_emissions.json", "w") as f1:
            json.dump(json1_data, f1, indent=2)
        print("💾 Saved JSON 1 to 'datacenter_emissions.json'")

        # --- JSON 2: Optimal Job Schedule (List of running tuples) ---
        json2_data = {}
        for j in jobs:
            # We only record the times the job is actively running
            active_schedule = []
            for t in times:
                for d in dcs:
                    if x[(j, d, t)].varValue == 1.0:
                        active_schedule.append(
                            [times_list[t].strftime("%Y-%m-%d %H:00 UTC"), d]
                        )

            # Sort chronologically just to be safe
            active_schedule.sort(key=lambda item: item[0])
            json2_data[j] = active_schedule

        with open("job_schedule.json", "w") as f2:
            json.dump(json2_data, f2, indent=2)
        print("💾 Saved JSON 2 to 'job_schedule.json'")

        # Calculate total emissions for terminal output
        total_carbon = (
            sum(
                ram_reqs[j] * C[(d, t)] * x[(j, d, t)].varValue
                for j in jobs
                for d in dcs
                for t in times
            )
            / 1000.0
        )
        print(f"🌱 Total Fleet CO₂ Emissions: {total_carbon:.2f} kg CO₂")

    else:
        print(
            f"❌ Could not find an optimal schedule. Status: {pulp.LpStatus[prob.status]}"
        )


if __name__ == "__main__":
    solve_preemptible_batch_scheduling()
