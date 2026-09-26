import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import NYC_BOUNDS, TaxiFeatureEngineer

TRAIN_DATA_PATH = "Data/train.csv"
SAVED_MODEL_PATH = "models/ridge_taxi_model.joblib"


def clean_train_data(df: pd.DataFrame) -> pd.DataFrame:
    """Removes outliers and sensor noise from training data."""
    valid_mask = (
        df["trip_duration"].between(60, 24 * 3600) &
        df["pickup_latitude"].between(NYC_BOUNDS["lat_min"], NYC_BOUNDS["lat_max"]) &
        df["pickup_longitude"].between(NYC_BOUNDS["lon_min"], NYC_BOUNDS["lon_max"]) &
        df["dropoff_latitude"].between(NYC_BOUNDS["lat_min"], NYC_BOUNDS["lat_max"]) &
        df["dropoff_longitude"].between(NYC_BOUNDS["lon_min"], NYC_BOUNDS["lon_max"])
    )
    return df[valid_mask].copy()


def build_pipeline() -> Pipeline:
    numeric_features = [
        "p_lon", "p_lat", "d_lon", "d_lat", "passenger_count",
        "haversine_dist", "manhattan_dist", "bearing", "dist_log",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"
    ]
    categorical_features = ["vendor_id", "store_and_fwd_flag"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    return Pipeline(
        steps=[
            ("feature_engineer", TaxiFeatureEngineer()),
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=1.0, random_state=42))
        ]
    )


def main():
    print(f"Loading training data from {TRAIN_DATA_PATH}...")
    df_raw = pd.read_csv(TRAIN_DATA_PATH)
    df_clean = clean_train_data(df_raw)

    print(f"Cleaned samples: {len(df_clean)} / {len(df_raw)}")

    y_train = np.log1p(df_clean["trip_duration"].values)
    X_train = df_clean.drop(columns=["trip_duration"])

    pipeline = build_pipeline()
    print("Fitting Ridge(alpha=1.0) pipeline...")
    pipeline.fit(X_train, y_train)

    print(f"Saving trained pipeline to {SAVED_MODEL_PATH}...")
    joblib.dump(pipeline, SAVED_MODEL_PATH)
    print("Model training and saving successfully completed.")


if __name__ == "__main__":
    main()