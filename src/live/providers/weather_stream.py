"""
ChronosGuard: Real-Time Live Environmental & Energy Demand Stream Provider
Ingests real atmospheric sensor readings and solar radiation metrics from Open-Meteo API.
Features: Temperature, humidity, barometric pressure, wind speed, solar radiation, temperature anomaly.
Target: Energy Grid Peak Strain Event (Extreme temperature / solar power generation deficit).
"""

import urllib.request
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

from src.live.providers.base import BaseLiveProvider

OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"

# City Coordinates for real-world environmental monitoring
LOCATIONS = {
    "NYC": {"lat": 40.7128, "lon": -74.0060, "name": "New York Metro Grid"},
    "PHOENIX": {"lat": 33.4484, "lon": -112.0740, "name": "Phoenix Desert Grid"},
    "LONDON": {"lat": 51.5074, "lon": -0.1278, "name": "UK National Grid"},
    "TOKYO": {"lat": 35.6762, "lon": 139.6503, "name": "Tokyo Metro Grid"}
}

class EnvironmentalEnergyLiveProvider(BaseLiveProvider):
    def __init__(self, location_key: str = "NYC"):
        self.location_key = location_key if location_key in LOCATIONS else "NYC"
        self.coords = LOCATIONS[self.location_key]

    @property
    def name(self) -> str:
        return f"weather_grid_{self.location_key.lower()}"

    @property
    def display_name(self) -> str:
        return f"Live Energy Grid Monitor ({self.coords['name']})"

    @property
    def feature_columns(self) -> List[str]:
        return [
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "wind_speed_10m",
            "direct_radiation",
            "temp_demand_index",
            "dew_point_2m",
            "cloud_cover"
        ]

    @property
    def target_column(self) -> str:
        return "is_peak_strain"

    @property
    def cost_parameters(self) -> Dict[str, float]:
        return {
            "cost_fn": 8500.0,    # Unprepared grid outage / emergency spinning reserve
            "cost_fp": 350.0,     # Peaker plant standby reservation fee
            "retrain_expense": 1200.0
        }

    def _request_json(self, url: str) -> Any:
        req = urllib.request.Request(url, headers={"User-Agent": "ChronosWeather/1.0"})
        with urllib.request.urlopen(req, timeout=8.0) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def fetch_baseline_data(self, n_samples: int = 250) -> pd.DataFrame:
        """
        Fetches the past 14 days (336 hourly observations) from Open-Meteo as baseline T0.
        """
        url = (
            f"{OPEN_METEO_BASE_URL}?latitude={self.coords['lat']}&longitude={self.coords['lon']}"
            f"&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,direct_radiation,dew_point_2m,cloud_cover"
            f"&past_days=14&forecast_days=1"
        )
        data = self._request_json(url)
        hourly = data["hourly"]

        df = pd.DataFrame({
            "temperature_2m": hourly["temperature_2m"],
            "relative_humidity_2m": hourly["relative_humidity_2m"],
            "surface_pressure": hourly["surface_pressure"],
            "wind_speed_10m": hourly["wind_speed_10m"],
            "direct_radiation": hourly["direct_radiation"],
            "dew_point_2m": hourly.get("dew_point_2m", hourly["temperature_2m"]),
            "cloud_cover": hourly.get("cloud_cover", [45.0] * len(hourly["temperature_2m"]))
        })

        df["temp_demand_index"] = (df["temperature_2m"] - 18.5).abs().round(2)

        high_demand = (
            0.45 * (df["temp_demand_index"] / 10.0) +
            0.25 * (df["direct_radiation"] / 800.0) -
            0.15 * (df["wind_speed_10m"] / 35.0) +
            0.15 * (df["relative_humidity_2m"] / 100.0)
        )
        p_thresh = np.percentile(high_demand.dropna(), 78)
        df["is_peak_strain"] = (high_demand > p_thresh).astype(int)

        clean_df = df.dropna().tail(n_samples).reset_index(drop=True)
        return clean_df[self.feature_columns + [self.target_column]]

    def fetch_live_tick(self) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Fetches current real-time atmospheric sensor readings.
        """
        url = (
            f"{OPEN_METEO_BASE_URL}?latitude={self.coords['lat']}&longitude={self.coords['lon']}"
            f"&current=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,dew_point_2m,cloud_cover"
        )
        data = self._request_json(url)
        curr = data["current"]

        temp = float(curr["temperature_2m"])
        humidity = float(curr["relative_humidity_2m"])
        pressure = float(curr["surface_pressure"])
        wind = float(curr["wind_speed_10m"])
        dew = float(curr.get("dew_point_2m", temp - (100 - humidity) / 5.0))
        cloud = float(curr.get("cloud_cover", 40.0))
        
        # Estimate solar radiation based on time of day and cloud dampening
        utc_hour = datetime.now(timezone.utc).hour
        daylight_factor = max(0.0, np.sin((utc_hour - 6) * np.pi / 12)) if 6 <= utc_hour <= 18 else 0.0
        cloud_factor = max(0.1, 1.0 - (cloud / 100.0) * 0.75)
        radiation = float(daylight_factor * 850.0 * cloud_factor)
        demand_index = float(abs(temp - 18.5))

        features = {
            "temperature_2m": round(temp, 2),
            "relative_humidity_2m": round(humidity, 1),
            "surface_pressure": round(pressure, 1),
            "wind_speed_10m": round(wind, 1),
            "direct_radiation": round(radiation, 1),
            "temp_demand_index": round(demand_index, 2),
            "dew_point_2m": round(dew, 2),
            "cloud_cover": round(cloud, 1)
        }

        meta = {
            "grid_name": self.coords["name"],
            "lat": self.coords["lat"],
            "lon": self.coords["lon"],
            "temp_c": temp,
            "timestamp": curr.get("time", datetime.now(timezone.utc).isoformat())
        }

        return features, meta

    def resolve_delayed_ground_truth(
        self,
        tick_id: int,
        features: Dict[str, float],
        age_seconds: float
    ) -> Optional[int]:
        """
        Resolves ground truth after 15 seconds: Checks if environmental demand caused peak strain.
        """
        if age_seconds < 15.0:
            return None

        demand = features.get("temp_demand_index", 0.0)
        is_strain = 1 if demand > 8.0 else 0
        return is_strain
