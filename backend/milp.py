import pulp
import pandas as pd
from cost_estimation import get_carbon_forecast


def calculate_naive_assignment(
    estimated_hours, job_power_kw, raw_df, default_dc="AWS US East VA"
):
    """
    Simulates what happens today: A user clicks "Train" and the job runs immediately
    in the default cloud region (t=0 to t=H).
    """
    print(
        "\n[ baseline ] Evaluating Naive Assignment (Run Immediately in Default Region)"
    )
    times = sorted(raw_df["Time_UTC"].unique())

    if len(times) < estimated_hours:
        print("❌ Not enough forecast data to complete the job.")
        return None

    total_cost_gco2 = 0.0
    print(
        f"             Simulating {estimated_hours}-hour execution starting at {times[0].strftime('%H:00 GMT')} in {default_dc}..."
    )

    for i in range(estimated_hours):
        t = times[i]
        # Get the carbon intensity for the default data center at this specific hour
        c_intensity = raw_df[
            (raw_df["Data_Center"] == default_dc) & (raw_df["Time_UTC"] == t)
        ]["Effective_Carbon_gCO2"].values[0]

        # Power (kW) * 1 hour * Carbon Intensity (gCO2/kWh) = grams of CO2
        step_cost = job_power_kw * c_intensity
        total_cost_gco2 += step_cost
        print(
            f"             > Hour {i+1} ({t.strftime('%H:00 GMT')}): Intensity = {c_intensity:>5.1f} gCO₂/kWh | Emitted = {step_cost/1000:>6.3f} kg"
        )

    total_kg_co2 = total_cost_gco2 / 1000.0
    print(f"             👉 Total Naive Emissions: {total_kg_co2:.3f} kg CO₂")
    return total_kg_co2


def schedule_job_milp(estimated_hours, deadline_hour_index, job_power_kw, raw_df):
    """
    The MILP Scheduler: Finds the absolute mathematical minimum carbon footprint
    across both time (shifting) and space (geography).
    """
    print("\n[ optimize ] Initializing MILP Spatio-Temporal Solver...")
    data_centers = raw_df["Data_Center"].unique().tolist()
    times = sorted(raw_df["Time_UTC"].unique())
    time_to_idx = {t: i for i, t in enumerate(times)}
    time_indices = list(range(len(times)))

    if deadline_hour_index > len(time_indices):
        raise ValueError("Deadline index exceeds the available forecast.")

    valid_start_times = [
        t for t in time_indices if t + estimated_hours <= deadline_hour_index
    ]

    print(f"             Target constraints:")
    print(f"             - Compute Duration:  {estimated_hours} sequential hours")
    print(f"             - Hardware Draw:     {job_power_kw} kW")
    print(
        f"             - Hard Deadline:     Must finish by {times[deadline_hour_index].strftime('%H:00 GMT')}"
    )
    print(
        f"             - Valid Start Slots: {len(valid_start_times)} valid chronological windows across {len(data_centers)} locations"
    )

    # Fast lookup dictionary for carbon intensity
    C = {}
    for _, row in raw_df.iterrows():
        d = row["Data_Center"]
        t_idx = time_to_idx[row["Time_UTC"]]
        C[(d, t_idx)] = row["Effective_Carbon_gCO2"]

    prob = pulp.LpProblem("Power_Aware_Sequential_Scheduling", pulp.LpMinimize)
    s = pulp.LpVariable.dicts(
        "start", [(d, t) for d in data_centers for t in time_indices], cat="Binary"
    )
    x = pulp.LpVariable.dicts(
        "run", [(d, t) for d in data_centers for t in time_indices], cat="Binary"
    )

    # OBJECTIVE FUNCTION: Minimize absolute carbon
    prob += pulp.lpSum(
        [
            job_power_kw * C[(d, t)] * x[(d, t)]
            for d in data_centers
            for t in time_indices
        ]
    )

    # CONSTRAINTS
    prob += (
        pulp.lpSum([s[(d, t)] for d in data_centers for t in valid_start_times]) == 1
    )

    invalid_start_times = [t for t in time_indices if t not in valid_start_times]
    for d in data_centers:
        for t in invalid_start_times:
            prob += s[(d, t)] == 0

    for d in data_centers:
        for t in time_indices:
            relevant_starts = [
                s[(d, t_start)]
                for t_start in range(max(0, t - estimated_hours + 1), t + 1)
            ]
            prob += x[(d, t)] == pulp.lpSum(relevant_starts)

    print("             🧠 Running branch-and-cut optimization...")
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] == "Optimal":
        for d in data_centers:
            for t in time_indices:
                if s[(d, t)].varValue == 1.0:
                    start_time = times[t]
                    end_time = times[t + estimated_hours]
                    total_cost_kg_co2 = pulp.value(prob.objective) / 1000.0

                    print(f"             ✅ Optimal Mathematical Route Locked!")
                    print(f"             📍 Location:   {d}")
                    print(
                        f"             ⏳ Start Time: {start_time.strftime('%H:00 GMT')}"
                    )
                    print(
                        f"             🏁 End Time:   {end_time.strftime('%H:00 GMT')}"
                    )
                    print(
                        f"             👉 Total MILP Emissions: {total_cost_kg_co2:.3f} kg CO₂"
                    )

                    # Print the step-by-step breakdown of the winning schedule
                    print("             Trace of optimal run:")
                    for i in range(estimated_hours):
                        c_intensity = C[(d, t + i)]
                        step_cost = job_power_kw * c_intensity
                        print(
                            f"             > Hour {i+1} ({times[t+i].strftime('%H:00 GMT')}): Intensity = {c_intensity:>5.1f} gCO₂/kWh | Emitted = {step_cost/1000:>6.3f} kg"
                        )

                    return d, start_time, total_cost_kg_co2
    else:
        print("❌ No feasible schedule found within the deadline.")
        return None, None, None


if __name__ == "__main__":
    print("🌍 Fetching live Spatio-Temporal forecast...\n")
    pivot_df, raw_df = get_carbon_forecast()

    # --- A MASSIVE TEST CASE ---
    # Job: Train Llama-3 8B (Fine-tuning)
    # Hardware: 15 kW (e.g., Crusoe 8x H100 GPU cluster)
    # Duration: 8 hours of sequential compute
    # Deadline: Must finish within the next 22 hours
    estimated_hours = 8
    job_power_kw = 15.0
    deadline_hour_index = 22

    print("=========================================================")
    print(f"🚀 TEST CASE: {estimated_hours}-Hour Distributed LLM Fine-Tuning Job")
    print("=========================================================")

    # 1. Calculate Baseline
    naive_cost = calculate_naive_assignment(
        estimated_hours, job_power_kw, raw_df, default_dc="AWS US East VA"
    )

    # 2. Calculate MILP Optimal
    opt_dc, opt_start, opt_cost = schedule_job_milp(
        estimated_hours, deadline_hour_index, job_power_kw, raw_df
    )

    # 3. Final Pitch Comparison
    if naive_cost is not None and opt_cost is not None:
        savings_kg = naive_cost - opt_cost
        savings_pct = (savings_kg / naive_cost) * 100 if naive_cost > 0 else 0

        print("\n🏆 PERFORMANCE COMPARISON (HACKATHON RESULTS)")
        print("-" * 55)
        print(f"   Standard Cloud Assignment:   {naive_cost:>8.3f} kg CO₂")
        print(f"   Green AutoML MILP Engine:    {opt_cost:>8.3f} kg CO₂")
        print("-" * 55)
        print(f"   🔥 Total Carbon Saved:       {savings_kg:>8.3f} kg CO₂")
        print(f"   📉 Emissions Reduction:      {savings_pct:>8.1f}%")
        print("=========================================================\n")
