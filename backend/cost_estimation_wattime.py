import requests
import pandas as pd
import math
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from requests.auth import HTTPBasicAuth


# --- 1. CREDENTIAL MANAGEMENT ---
def load_watttime_credentials(filepath="wattime_credentials.txt"):
    """Safely loads credentials from a local text file."""
    if not os.path.exists(filepath):
        print(f"⚠️ Warning: '{filepath}' not found. Falling back to base grid data.")
        return None, None

    creds = {}
    with open(filepath, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                creds[key] = value

    return creds.get("WATTTIME_USER"), creds.get("WATTTIME_PASS")


WATTTIME_USER, WATTTIME_PASS = load_watttime_credentials()

# --- CONFIGURATION ---
GPU_CLUSTER_LOAD_MW = 0.015  # 8x H100 Cluster Load

# We use California to prove WattTime API integration, and Texas/Finland to prove our Physics Fallback
DATA_CENTERS = {
    "Crusoe_California_CA": {
        "lat": 38.5816,
        "lon": -121.4944,
        "base_carbon": 250,
    },  # Maps to CAISO_NORTH (Free Tier)
    "Crusoe_Abilene_TX": {
        "lat": 32.4487,
        "lon": -99.7331,
        "base_carbon": 350,
    },  # Maps to ERCOT (Requires Fallback)
    "Google_Hamina_FI": {
        "lat": 60.5693,
        "lon": 27.1938,
        "base_carbon": 50,
    },  # Maps to Finland (Requires Fallback)
}


# --- 2. WATTTIME API INTEGRATION ---
def get_watttime_token():
    if not WATTTIME_USER or not WATTTIME_PASS:
        return None
    try:
        response = requests.get(
            "https://api2.watttime.org/v2/login",
            auth=HTTPBasicAuth(WATTTIME_USER, WATTTIME_PASS),
        )
        response.raise_for_status()
        return response.json()["token"]
    except Exception:
        return None


def fetch_watttime_forecast(lat, lon, token):
    if not token:
        return None
    headers = {"Authorization": f"Bearer {token}"}
    ba_abbrev = "Unknown"
    try:
        # Get Balancing Authority
        ba_response = requests.get(
            "https://api2.watttime.org/v2/ba-from-loc",
            headers=headers,
            params={"latitude": lat, "longitude": lon},
        )
        ba_response.raise_for_status()
        ba_abbrev = ba_response.json()["abbrev"]

        # Fetch Forecast
        forecast_response = requests.get(
            "https://api2.watttime.org/v2/forecast",
            headers=headers,
            params={"ba": ba_abbrev},
        )
        forecast_response.raise_for_status()

        forecast_data = forecast_response.json()["forecast"]
        return {item["point_time"]: item["value"] for item in forecast_data}

    except json.JSONDecodeError:
        print(
            f"⚠️ WattTime Free Tier: Forecast locked for {ba_abbrev}. Triggering Physics Fallback."
        )
        return None
    except requests.exceptions.HTTPError:
        print(f"⚠️ WattTime Access Denied for {ba_abbrev}. Triggering Physics Fallback.")
        return None
    except Exception:
        return None


# --- 3. SPATIO-TEMPORAL PHYSICS ENGINE ---
def calc_wind_megawatts(wind_speed_kmh_10m):
    """Calculates aerodynamic power output using the Wind Profile Power Law."""
    v_10 = wind_speed_kmh_10m / 3.6
    # Extrapolate to 100m hub height using Hellmann exponent (0.143)
    v_100 = v_10 * ((100.0 / 10.0) ** 0.143)

    if v_100 < 3.0 or v_100 >= 25.0:
        return 0.0
    elif v_100 >= 12.0:
        return 2.0
    # Power equation: P = 0.5 * rho * A * v^3 * Cp
    else:
        return (0.5 * 1.225 * (math.pi * 45.0**2) * (v_100**3) * 0.40) / 1_000_000


def calc_solar_megawatts(irradiance_w_m2):
    return (10000 * 0.20 * irradiance_w_m2 * 0.75) / 1_000_000


# --- 4. THE CORE EXECUTION LOOP ---
def get_absolute_carbon_forecast():
    forecast_data = []
    wt_token = get_watttime_token()

    now_utc = pd.Timestamp.now("UTC").floor("h")
    end_utc = now_utc + pd.Timedelta(hours=23)

    for name, site in DATA_CENTERS.items():
        wt_forecast_dict = fetch_watttime_forecast(site["lat"], site["lon"], wt_token)

        params = {
            "latitude": site["lat"],
            "longitude": site["lon"],
            "hourly": "wind_speed_10m,direct_normal_irradiance",
            "timezone": "GMT",
            "forecast_days": 2,
        }

        try:
            weather_resp = requests.get(
                "https://api.open-meteo.com/v1/forecast", params=params
            )
            weather_resp.raise_for_status()
            w_data = weather_resp.json()

            times = pd.to_datetime(w_data["hourly"]["time"], utc=True)
            winds = w_data["hourly"]["wind_speed_10m"]
            solars = w_data["hourly"]["direct_normal_irradiance"]

            for i in range(len(times)):
                t = times[i]
                if t < now_utc or t > end_utc:
                    continue

                wind_mw = calc_wind_megawatts(winds[i])
                solar_mw = calc_solar_megawatts(solars[i])
                total_green_mw = wind_mw + solar_mw

                # NET-LOAD MATH: Prevent 100% utilization illusion
                net_grid_pull_mw = max(0.0, GPU_CLUSTER_LOAD_MW - total_green_mw)

                time_str = t.strftime("%Y-%m-%dT%H:%M:%SZ")
                if wt_forecast_dict and time_str in wt_forecast_dict:
                    # WattTime returns lbs/MWh. Convert to kg/MWh.
                    grid_intensity_kg_mwh = wt_forecast_dict[time_str] * 0.453592
                else:
                    grid_intensity_kg_mwh = site["base_carbon"]

                absolute_kg_co2 = net_grid_pull_mw * grid_intensity_kg_mwh

                forecast_data.append(
                    {
                        "Time_UTC": t,
                        "Data_Center": name.replace("_", " "),
                        "Absolute_Emissions_kgCO2": round(absolute_kg_co2, 4),
                    }
                )

        except Exception as e:
            print(f"Failed to process {name}: {e}")

    df = pd.DataFrame(forecast_data)

    # Pivot for terminal display
    pivot_df = df.copy()
    pivot_df["Time_UTC"] = pivot_df["Time_UTC"].dt.strftime("%H:00 GMT")
    pivot_df = pivot_df.pivot(
        index="Time_UTC", columns="Data_Center", values="Absolute_Emissions_kgCO2"
    )
    pivot_df.reset_index(inplace=True)
    pivot_df.columns.name = None

    return pivot_df, df


def plot_carbon_forecast(raw_df):
    plt.figure(figsize=(12, 6))
    sns.set_style("darkgrid")

    for center in raw_df["Data_Center"].unique():
        subset = raw_df[raw_df["Data_Center"] == center]
        plt.plot(
            subset["Time_UTC"],
            subset["Absolute_Emissions_kgCO2"],
            label=center,
            linewidth=3,
            marker="o",
            markersize=4,
        )

    plt.title(
        "Absolute AI Training Emissions (WattTime API + Physics Fallback)",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Coordinated Universal Time (UTC/GMT)", fontsize=12)
    plt.ylabel("Absolute Emissions (kg CO₂ per hour)", fontsize=12)
    plt.legend(title="Data Center Locations")
    plt.tight_layout()

    output_filename = "carbon_forecast_plot.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"\n✅ Plot saved successfully as: {os.path.abspath(output_filename)}")


if __name__ == "__main__":
    print(f"🚀 Initializing Green AutoML Spatio-Temporal Engine...")
    print(f"⚙️ Target Load: {GPU_CLUSTER_LOAD_MW} MW (8x H100 Cluster)")
    print("📡 Pulling live forecasts...\n")

    pivot_df, raw_df = get_absolute_carbon_forecast()

    print("\n=== ABSOLUTE HOURLY CARBON EMISSIONS (kg CO₂) ===")
    print(pivot_df.to_markdown(index=False))

    plot_carbon_forecast(raw_df)
