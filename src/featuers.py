import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

NYC_BOUNDS = {
    "lat_min": 40.57, "lat_max": 40.92,
    "lon_min": -74.15, "lon_max": -73.70
}


class TaxiFeatureEngineer(BaseEstimator, TransformerMixin):
    """Generates geographical distances, compass bearings, and time cycle features."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        df["p_lon"] = df["pickup_longitude"].clip(NYC_BOUNDS["lon_min"], NYC_BOUNDS["lon_max"])
        df["p_lat"] = df["pickup_latitude"].clip(NYC_BOUNDS["lat_min"], NYC_BOUNDS["lat_max"])
        df["d_lon"] = df["dropoff_longitude"].clip(NYC_BOUNDS["lon_min"], NYC_BOUNDS["lon_max"])
        df["d_lat"] = df["dropoff_latitude"].clip(NYC_BOUNDS["lat_min"], NYC_BOUNDS["lat_max"])

        dt = pd.to_datetime(df["pickup_datetime"])
        hour = dt.dt.hour + dt.dt.minute / 60.0
        dayofweek = dt.dt.dayofweek

        df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        df["dow_sin"] = np.sin(2 * np.pi * dayofweek / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * dayofweek / 7.0)
        df["is_weekend"] = dayofweek.isin([5, 6]).astype(int)

        lat1, lon1 = np.radians(df["p_lat"]), np.radians(df["p_lon"])
        lat2, lon2 = np.radians(df["d_lat"]), np.radians(df["d_lon"])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        df["haversine_dist"] = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

        lat_dist_km = np.abs(df["d_lat"] - df["p_lat"]) * 111.0
        lon_dist_km = np.abs(df["d_lon"] - df["p_lon"]) * (111.0 * np.cos(lat1))
        df["manhattan_dist"] = lat_dist_km + lon_dist_km

        y_bear = np.sin(dlon) * np.cos(lat2)
        x_bear = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        df["bearing"] = np.arctan2(y_bear, x_bear)

        df["dist_log"] = np.log1p(df["haversine_dist"])

        return df