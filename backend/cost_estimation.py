import requests
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_CENTERS = {
    "Crusoe_Abilene_TX": {"lat": 32.4487, "lon": -99.7331, "base_carbon": 100},
    "AWS_US_East_VA": {"lat": 39.0438, "lon": -77.4874, "base_carbon": 100},
    "Google_Hamina_FI": {"lat": 60.5693, "lon": 27.1938, "base_carbon": 100},
}


# --- THE PHYSICS ENGINE ---
def calc_wind_megawatts(wind_speed_kmh):
    v = wind_speed_kmh / 3.6
    if v < 3.0 or v >= 25.0:
        return 0.0
    elif v >= 12.0:
        return 2.0
    else:
        return (0.5 * 1.225 * (math.pi * 45.0**2) * (v**3) * 0.40) / 1_000_000


def calc_solar_megawatts(irradiance_w_m2):
    return (10000 * 0.20 * irradiance_w_m2 * 0.75) / 1_000_000


# --------------------------


def get_carbon_forecast():
    forecast_data = []
    url = "https://api.open-meteo.com/v1/forecast"

    # FIX 1: Updated Pandas syntax to avoid the deprecation warning
    now_utc = pd.Timestamp.now("UTC").floor("h")
    end_utc = now_utc + pd.Timedelta(hours=23)

    for name, site in DATA_CENTERS.items():
        params = {
            "latitude": site["lat"],
            "longitude": site["lon"],
            "hourly": "wind_speed_10m,direct_normal_irradiance",
            "timezone": "GMT",
            "forecast_days": 2,
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
                        "Data_Center": name.replace("_", " "),
                        "Effective_Carbon_gCO2": round(effective_carbon, 1),
                    }
                )

        except Exception as e:
            print(f"Failed to fetch data for {name}: {e}")

    df = pd.DataFrame(forecast_data)

    pivot_df = df.pivot(
        index="Time_UTC", columns="Data_Center", values="Effective_Carbon_gCO2"
    )
    data_center_cols = [col for col in pivot_df.columns]

    pivot_df["Recommended Routing"] = pivot_df[data_center_cols].idxmin(axis=1)

    pivot_df.index = pivot_df.index.strftime("%H:00 GMT")
    pivot_df.reset_index(inplace=True)
    pivot_df.rename(columns={"Time_UTC": "Global Time"}, inplace=True)
    pivot_df.columns.name = None

    return pivot_df, df


def plot_carbon_forecast(raw_df):
    plt.figure(figsize=(12, 6))
    sns.set_style("darkgrid")

    for center in raw_df["Data_Center"].unique():
        subset = raw_df[raw_df["Data_Center"] == center]
        plt.plot(
            subset["Time_UTC"],
            subset["Effective_Carbon_gCO2"],
            label=center,
            linewidth=3,
            marker="o",
            markersize=4,
        )

    plt.title(
        "Global 24-Hour Carbon Forecast (Synchronized to GMT)",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Coordinated Universal Time (UTC/GMT)", fontsize=12)
    plt.ylabel("Effective Carbon Intensity (gCO/kWh)", fontsize=12)
    plt.legend(title="Data Center Locations")
    plt.tight_layout()

    # FIX 2: Save the plot as an image file instead of trying to open a GUI window
    output_filename = "carbon_forecast_plot.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(
        f"\n✅ Success! Plot saved successfully as: {os.path.abspath(output_filename)}"
    )


if __name__ == "__main__":
    print("Fetching global weather data and aligning to GMT timeline...\n")
    pivot_df, raw_df = get_carbon_forecast()

    print(pivot_df.to_markdown(index=False))
    plot_carbon_forecast(raw_df)
