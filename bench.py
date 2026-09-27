import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import NYC_BOUNDS, TaxiFeatureEngineer


TRAIN_DATA_PATH = "Data/train.csv"
VAL_DATA_PATH = "Data/val.csv"
MODEL_DIR = "models"
RESULTS_PATH = "models/model_comparison.csv"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same basic data-quality rules used by the original project."""
    valid_mask = (
        df["trip_duration"].between(60, 24 * 3600)
        & df["pickup_latitude"].between(
            NYC_BOUNDS["lat_min"], NYC_BOUNDS["lat_max"]
        )
        & df["pickup_longitude"].between(
            NYC_BOUNDS["lon_min"], NYC_BOUNDS["lon_max"]
        )
        & df["dropoff_latitude"].between(
            NYC_BOUNDS["lat_min"], NYC_BOUNDS["lat_max"]
        )
        & df["dropoff_longitude"].between(
            NYC_BOUNDS["lon_min"], NYC_BOUNDS["lon_max"]
        )
    )
    return df.loc[valid_mask].copy()


def build_preprocessor() -> ColumnTransformer:
    numeric_features = [
        "p_lon",
        "p_lat",
        "d_lon",
        "d_lat",
        "passenger_count",
        "haversine_dist",
        "manhattan_dist",
        "bearing",
        "dist_log",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "is_weekend",
    ]

    categorical_features = ["vendor_id", "store_and_fwd_flag"]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )


def build_pipeline(regressor) -> Pipeline:
    return Pipeline(
        steps=[
            ("feature_engineer", TaxiFeatureEngineer()),
            ("preprocessor", build_preprocessor()),
            ("regressor", regressor),
        ]
    )


def get_models():
    models = {
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=18,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=5,
            loss="huber",
            random_state=42,
        ),
    }

    # XGBoost is optional so the project still runs if the package is not installed.
    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            eval_metric="rmse",
            n_jobs=-1,
            random_state=42,
        )
    except ImportError:
        print("XGBoost not installed -> skipping XGBoost benchmark.")

    return models


def evaluate_predictions(y_true_log, y_pred_log):
    """
    Metrics are reported on the original seconds scale where appropriate.
    R²/RMSE(log) are also retained because the model is trained on log1p(target).
    """
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    # Keep predictions inside the same sensible range used by evaluate.py.
    y_pred = np.clip(y_pred, 30, 24 * 3600)
    y_pred_log = np.log1p(y_pred)

    return {
        "R2": r2_score(y_true, y_pred),
        "MAE_seconds": mean_absolute_error(y_true, y_pred),
        "MAE_minutes": mean_absolute_error(y_true, y_pred) / 60.0,
        "RMSE_seconds": np.sqrt(mean_squared_error(y_true, y_pred)),
        "RMSE_minutes": np.sqrt(mean_squared_error(y_true, y_pred)) / 60.0,
        "RMSLE": np.sqrt(mean_squared_error(y_true_log, y_pred_log)),
        "R2_log": r2_score(y_true_log, y_pred_log),
        "RMSE_log": np.sqrt(mean_squared_error(y_true_log, y_pred_log)),
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Loading training data: {TRAIN_DATA_PATH}")
    train_df = clean_data(pd.read_csv(TRAIN_DATA_PATH))

    print(f"Loading validation data: {VAL_DATA_PATH}")
    val_df = clean_data(pd.read_csv(VAL_DATA_PATH))

    print(f"Training samples after cleaning : {len(train_df):,}")
    print(f"Validation samples after cleaning: {len(val_df):,}")

    y_train_log = np.log1p(train_df["trip_duration"].to_numpy())
    y_val_log = np.log1p(val_df["trip_duration"].to_numpy())

    X_train = train_df.drop(columns=["trip_duration"])
    X_val = val_df.drop(columns=["trip_duration"])

    results = []

    for name, regressor in get_models().items():
        print("\n" + "=" * 70)
        print(f"Training: {name}")
        print("=" * 70)

        start = time.perf_counter()

        pipeline = build_pipeline(regressor)
        pipeline.fit(X_train, y_train_log)

        train_seconds = time.perf_counter() - start

        y_pred_log = pipeline.predict(X_val)
        metrics = evaluate_predictions(y_val_log, y_pred_log)

        safe_name = name.lower().replace(" ", "_")
        model_path = os.path.join(MODEL_DIR, f"{safe_name}_taxi_model.joblib")
        joblib.dump(pipeline, model_path)

        row = {
            "Model": name,
            "Training_Time_sec": round(train_seconds, 2),
            **{k: round(v, 5) for k, v in metrics.items()},
            "Model_Path": model_path,
        }
        results.append(row)

        print(f"R²              : {metrics['R2']:.5f}")
        print(f"MAE             : {metrics['MAE_minutes']:.2f} min")
        print(f"RMSE            : {metrics['RMSE_minutes']:.2f} min")
        print(f"RMSLE           : {metrics['RMSLE']:.5f}")
        print(f"Training time   : {train_seconds:.2f} sec")
        print(f"Saved to        : {model_path}")

    results_df = pd.DataFrame(results)

    # For this project, RMSLE is a particularly useful comparison metric
    # because the target is trained in log1p space.
    results_df = results_df.sort_values("RMSLE", ascending=True)
    results_df.to_csv(RESULTS_PATH, index=False)

    print("\n" + "=" * 70)
    print("MODEL BENCHMARK RESULTS")
    print("=" * 70)
    print(
        results_df[
            ["Model", "R2", "MAE_minutes", "RMSE_minutes", "RMSLE", "Training_Time_sec"]
        ].to_string(index=False)
    )
    print(f"\nComparison saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
