import os
import json
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import NYC_BOUNDS, TaxiFeatureEngineer

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError(
        "XGBoost is required. Install it with: pip install xgboost"
    ) from exc


TRAIN_DATA_PATH = "Data/train.csv"
VAL_DATA_PATH = "Data/val.csv"
MODEL_DIR = "models"

TUNED_RESULTS_PATH = os.path.join(MODEL_DIR, "tuned_model_comparison.csv")
XGB_SEARCH_RESULTS_PATH = os.path.join(
    MODEL_DIR, "xgboost_random_search.csv"
)
GB_SEARCH_RESULTS_PATH = os.path.join(
    MODEL_DIR, "gradient_boosting_random_search.csv"
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


def build_pipeline(model) -> Pipeline:
    return Pipeline(
        steps=[
            ("feature_engineer", TaxiFeatureEngineer()),
            ("preprocessor", build_preprocessor()),
            ("regressor", model),
        ]
    )


def get_xgboost_search():
    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
    )

    params = {
        "regressor__n_estimators": [200, 300, 400, 500, 700, 900],
        "regressor__max_depth": [3, 4, 5, 6, 7, 8, 10],
        "regressor__learning_rate": [0.02, 0.03, 0.05, 0.07, 0.1],
        "regressor__subsample": [0.7, 0.8, 0.9, 1.0],
        "regressor__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "regressor__min_child_weight": [1, 2, 3, 5, 8],
        "regressor__gamma": [0, 0.05, 0.1, 0.2, 0.4],
        "regressor__reg_alpha": [0, 0.01, 0.1, 0.5, 1.0],
        "regressor__reg_lambda": [1, 1.5, 2, 3, 5],
    }
    return model, params


def get_gradient_boosting_search():
    model = GradientBoostingRegressor(random_state=42)

    params = {
        "regressor__n_estimators": [100, 150, 200, 250, 300, 400, 500],
        "regressor__learning_rate": [0.02, 0.03, 0.05, 0.07, 0.1],
        "regressor__max_depth": [2, 3, 4, 5, 6],
        "regressor__min_samples_split": [2, 5, 10, 20],
        "regressor__min_samples_leaf": [1, 2, 5, 10],
        "regressor__subsample": [0.7, 0.8, 0.9, 1.0],
        "regressor__max_features": [None, "sqrt", "log2"],
        "regressor__loss": ["huber", "squared_error"],
    }
    return model, params


def evaluate_model(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val_log: np.ndarray,
):
    y_pred_log = pipeline.predict(X_val)

    y_true_seconds = np.expm1(y_val_log)
    y_pred_seconds = np.expm1(y_pred_log)

    y_pred_seconds = np.clip(y_pred_seconds, 30, 24 * 3600)
    y_pred_log = np.log1p(y_pred_seconds)

    rmse_seconds = np.sqrt(
        mean_squared_error(y_true_seconds, y_pred_seconds)
    )

    return {
        "R2": r2_score(y_true_seconds, y_pred_seconds),
        "MAE_minutes": mean_absolute_error(
            y_true_seconds, y_pred_seconds
        ) / 60.0,
        "RMSE_minutes": rmse_seconds / 60.0,
        "RMSLE": np.sqrt(
            mean_squared_error(y_val_log, y_pred_log)
        ),
        "R2_log": r2_score(y_val_log, y_pred_log),
        "RMSE_log": np.sqrt(
            mean_squared_error(y_val_log, y_pred_log)
        ),
    }


def make_search(pipeline, params, n_iter=25):
    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=params,
        n_iter=n_iter,
        scoring="neg_mean_squared_error",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        refit=True,
        return_train_score=True,
    )


def save_search_results(search, path):
    results = pd.DataFrame(search.cv_results_).copy()
    results["cv_RMSE_log"] = np.sqrt(-results["mean_test_score"])
    results["train_RMSE_log"] = np.sqrt(-results["mean_train_score"])
    results = results.sort_values("rank_test_score")
    results.to_csv(path, index=False)


def run_search(name, model, params, X_train, y_train_log, X_val, y_val_log):
    print("\n" + "=" * 78)
    print(f"TUNING {name.upper()}")
    print("=" * 78)

    pipeline = build_pipeline(model)
    search = make_search(pipeline, params, n_iter=25)

    start = time.perf_counter()
    search.fit(X_train, y_train_log)
    tuning_time = time.perf_counter() - start

    metrics = evaluate_model(
        search.best_estimator_, X_val, y_val_log
    )

    safe_name = name.lower().replace(" ", "_")
    model_path = os.path.join(
        MODEL_DIR, f"{safe_name}_tuned_taxi_model.joblib"
    )
    joblib.dump(search.best_estimator_, model_path)

    search_results_path = (
        XGB_SEARCH_RESULTS_PATH
        if name == "XGBoost"
        else GB_SEARCH_RESULTS_PATH
    )
    save_search_results(search, search_results_path)

    print("\nBest parameters:")
    print(json.dumps(search.best_params_, indent=2))

    print("\nValidation:")
    print(f"R2              : {metrics['R2']:.5f}")
    print(f"MAE             : {metrics['MAE_minutes']:.2f} min")
    print(f"RMSE            : {metrics['RMSE_minutes']:.2f} min")
    print(f"RMSLE           : {metrics['RMSLE']:.5f}")
    print(f"R2 (log)        : {metrics['R2_log']:.5f}")
    print(f"RMSE (log)      : {metrics['RMSE_log']:.5f}")
    print(f"Best CV RMSE    : {np.sqrt(-search.best_score_):.5f}")
    print(f"Tuning time     : {tuning_time:.2f} sec")
    print(f"Saved model     : {model_path}")

    return {
        "Model": f"{name} Tuned",
        **{k: round(v, 5) for k, v in metrics.items()},
        "Best_CV_RMSE_log": round(
            np.sqrt(-search.best_score_), 5
        ),
        "Tuning_Time_sec": round(tuning_time, 2),
        "Model_Path": model_path,
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 78)
    print("HYPERPARAMETER TUNING - NYC TAXI TRIP DURATION")
    print("=" * 78)

    print(f"\nLoading training data: {TRAIN_DATA_PATH}")
    train_df = clean_data(pd.read_csv(TRAIN_DATA_PATH))

    print(f"Loading validation data: {VAL_DATA_PATH}")
    val_df = clean_data(pd.read_csv(VAL_DATA_PATH))

    print(f"Training samples after cleaning  : {len(train_df):,}")
    print(f"Validation samples after cleaning: {len(val_df):,}")

    y_train_log = np.log1p(train_df["trip_duration"].to_numpy())
    y_val_log = np.log1p(val_df["trip_duration"].to_numpy())

    X_train = train_df.drop(columns=["trip_duration"])
    X_val = val_df.drop(columns=["trip_duration"])

    results = []

    xgb_model, xgb_params = get_xgboost_search()
    results.append(
        run_search(
            "XGBoost",
            xgb_model,
            xgb_params,
            X_train,
            y_train_log,
            X_val,
            y_val_log,
        )
    )

    gb_model, gb_params = get_gradient_boosting_search()
    results.append(
        run_search(
            "Gradient Boosting",
            gb_model,
            gb_params,
            X_train,
            y_train_log,
            X_val,
            y_val_log,
        )
    )

    results_df = pd.DataFrame(results).sort_values(
        "RMSLE", ascending=True
    )

    results_df.to_csv(TUNED_RESULTS_PATH, index=False)

    print("\n" + "=" * 78)
    print("TUNED MODEL COMPARISON")
    print("=" * 78)

    columns = [
        "Model",
        "R2",
        "MAE_minutes",
        "RMSE_minutes",
        "RMSLE",
        "R2_log",
        "RMSE_log",
        "Best_CV_RMSE_log",
        "Tuning_Time_sec",
    ]
    print(results_df[columns].to_string(index=False))

    print(f"\nComparison saved to: {TUNED_RESULTS_PATH}")


if __name__ == "__main__":
    main()
