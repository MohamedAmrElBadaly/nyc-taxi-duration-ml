import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import NYC_BOUNDS, TaxiFeatureEngineer


TRAIN_DATA_PATH = "Data/train.csv"
VAL_DATA_PATH = "Data/val.csv"
MODEL_DIR = "models"

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR, "final_gradient_boosting_taxi_model.joblib"
)
FINAL_CONFIG_PATH = os.path.join(
    MODEL_DIR, "final_model_config.json"
)
TRAINING_REPORT_PATH = os.path.join(
    MODEL_DIR, "final_training_report.csv"
)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
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
        "p_lon", "p_lat", "d_lon", "d_lat", "passenger_count",
        "haversine_dist", "manhattan_dist", "bearing", "dist_log",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
    ]
    categorical_features = ["vendor_id", "store_and_fwd_flag"]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_features,
            ),
        ]
    )


def build_pipeline(model: GradientBoostingRegressor) -> Pipeline:
    return Pipeline(
        steps=[
            ("feature_engineer", TaxiFeatureEngineer()),
            ("preprocessor", build_preprocessor()),
            ("regressor", model),
        ]
    )


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 78)
    print("FINAL TRAINING - TRAIN + VALIDATION")
    print("=" * 78)

    train_df = clean_data(pd.read_csv(TRAIN_DATA_PATH))
    val_df = clean_data(pd.read_csv(VAL_DATA_PATH))

    print(f"Train samples after cleaning      : {len(train_df):,}")
    print(f"Validation samples after cleaning : {len(val_df):,}")

    # These parameters were selected in the previous tuning stage.
    best_params = {
        "n_estimators": 400,
        "learning_rate": 0.02,
        "max_depth": 6,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
        "subsample": 0.8,
        "loss": "huber",
    }

    # Final fitting uses TRAIN + VALIDATION only.
    # The TEST SET is intentionally never loaded here.
    full_train = pd.concat(
        [train_df, val_df],
        axis=0,
        ignore_index=True,
    )

    y_full_log = np.log1p(full_train["trip_duration"].to_numpy())
    X_full = full_train.drop(columns=["trip_duration"])

    final_regressor = GradientBoostingRegressor(
        random_state=42,
        **best_params,
    )

    pipeline = build_pipeline(final_regressor)

    print("\nTraining final model on Train + Validation...")
    pipeline.fit(X_full, y_full_log)

    joblib.dump(pipeline, FINAL_MODEL_PATH)

    config = {
        "model": "GradientBoostingRegressor",
        "target_transformation": "log1p",
        "feature_engineering": "TaxiFeatureEngineer",
        "training_data": [
            TRAIN_DATA_PATH,
            VAL_DATA_PATH,
        ],
        "test_data_used_for_training": False,
        "hyperparameters": best_params,
    }

    with open(FINAL_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Training-set diagnostics only; not a generalization estimate.
    train_pred_log = pipeline.predict(X_full)
    y_full_seconds = np.expm1(y_full_log)
    pred_seconds = np.clip(
        np.expm1(train_pred_log),
        30,
        24 * 3600,
    )

    report = pd.DataFrame(
        [{
            "Model": "Final Gradient Boosting",
            "Samples_TrainPlusVal": len(full_train),
            "MAE_minutes_train_diagnostic": round(
                mean_absolute_error(y_full_seconds, pred_seconds) / 60.0,
                5,
            ),
            "RMSE_minutes_train_diagnostic": round(
                np.sqrt(
                    mean_squared_error(y_full_seconds, pred_seconds)
                ) / 60.0,
                5,
            ),
            "R2_train_diagnostic": round(
                r2_score(y_full_seconds, pred_seconds), 5
            ),
        }]
    )
    report.to_csv(TRAINING_REPORT_PATH, index=False)

    print("\nFinal model saved:")
    print(FINAL_MODEL_PATH)
    print("\nConfig saved:")
    print(FINAL_CONFIG_PATH)
    print("\nTraining diagnostics saved:")
    print(TRAINING_REPORT_PATH)
    print("\nTEST SET WAS NOT LOADED OR USED.")


if __name__ == "__main__":
    main()
