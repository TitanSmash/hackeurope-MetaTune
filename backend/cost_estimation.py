import requests
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os


# --- THE PHYSICS ENGINE ---
def calc_wind_megawatts(wind_speed_kmh):
    v = wind_speed_kmh / 3.6
    if v < 3.0 or v >= 25.0:
        return 0.0
    elif v >= 12.0:
        return 2.0
    else:
        return (0.5 * 1.225 * (math.pi * 45.0**2) * (v**3) * 0.40) / 10_000_000


def calc_solar_megawatts(irradiance_w_m2):
    return (10000 * 0.20 * irradiance_w_m2 * 0.75) / 10_000_000


# --------------------------


def get_carbon_forecast(csv_path="datacenters.csv", days_ahead=7):
    # Load datacenters from CSV dynamically
    dcs_df = pd.read_csv(csv_path)

    # Map CSV rows to dictionary
    DATA_CENTERS = {}
    for _, row in dcs_df.iterrows():
        name = row["data center name"]
        # Use 'co2 emissions per hour' as the base carbon intensity
        DATA_CENTERS[name] = {
            "lat": row["latitude"],
            "lon": row["longitude"],
            "base_carbon": row["base_carbon"],
        }

    forecast_data = []
    url = "https://api.open-meteo.com/v1/forecast"

    now_utc = pd.Timestamp.now("UTC").floor("h")
    # Extend timeline to 7 days ahead
    end_utc = now_utc + pd.Timedelta(days=days_ahead) - pd.Timedelta(hours=1)

    # Query 8 days to ensure full 168-hour coverage (avoids timezone cutting off the end)
    forecast_days_param = min(16, days_ahead + 1)

    for name, site in DATA_CENTERS.items():
        params = {
            "latitude": site["lat"],
            "longitude": site["lon"],
            "hourly": "wind_speed_10m,direct_normal_irradiance",
            "timezone": "GMT",
            "forecast_days": forecast_days_param,
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            times = pd.to_datetime(data["hourly"]["time"], utc=True)
            winds = data["hourly"]["wind_speed_10m"]
            solars = data["hourly"]["direct_normal_irradiance"]

            for i in range(len(times)):
                t = times[i]

                if t < now_utc or t > end_utc:
                    continue

                wind_mw = calc_wind_megawatts(winds[i])
                solar_mw = calc_solar_megawatts(solars[i])
                carbon_offset = (wind_mw + solar_mw) * 50
                effective_carbon = max(0, site["base_carbon"] - carbon_offset)

                forecast_data.append(
                    {
                        "Time_UTC": t,
                        "Data_Center": name,
                        "Effective_Carbon_gCO2": round(effective_carbon, 1),
                    }
                )

        except Exception as e:
            print(f"Failed to fetch data for {name}: {e}")

    df = pd.DataFrame(forecast_data)

    # Pivot for clean terminal output
    pivot_df = df.pivot(
        index="Time_UTC", columns="Data_Center", values="Effective_Carbon_gCO2"
    )
    data_center_cols = [col for col in pivot_df.columns]

    pivot_df["Recommended Routing"] = pivot_df[data_center_cols].idxmin(axis=1)

    # Update timestamp format to include the date (needed for a 7-day view)
    pivot_df.index = pivot_df.index.strftime("%Y-%m-%d %H:00 GMT")
    pivot_df.reset_index(inplace=True)
    pivot_df.rename(columns={"Time_UTC": "Global Time"}, inplace=True)
    pivot_df.columns.name = None

    return pivot_df, df


def plot_carbon_forecast(raw_df):
    plt.figure(figsize=(15, 6))
    sns.set_style("darkgrid")

    for center in raw_df["Data_Center"].unique():
        subset = raw_df[raw_df["Data_Center"] == center]
        plt.plot(
            subset["Time_UTC"],
            subset["Effective_Carbon_gCO2"],
            label=center,
            linewidth=2,
        )

    plt.title(
        "Global 7-Day Carbon Forecast (Synchronized to GMT)",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Coordinated Universal Time (UTC/GMT)", fontsize=12)
    plt.ylabel("Effective Carbon Intensity (gCO₂/kWh)", fontsize=12)
    plt.legend(title="Data Center Locations")
    plt.tight_layout()

    # Save the 7-day plot
    output_filename = "carbon_forecast_plot_7days.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(
        f"\n✅ Success! 7-Day Plot saved successfully as: {os.path.abspath(output_filename)}"
    )


if __name__ == "__main__":
    print("Fetching global 7-day weather data and aligning to GMT timeline...\n")
    pivot_df, raw_df = get_carbon_forecast()

    # Since a 7-day forecast has 168 rows, we will print the first 24 hours to the terminal
    # and save the rest for the plot or other files.
    print(pivot_df.head(24).to_markdown(index=False))
    print(f"\n... (Data truncated for display. {len(pivot_df)} total hours mapped.)")

    plot_carbon_forecast(raw_df)
