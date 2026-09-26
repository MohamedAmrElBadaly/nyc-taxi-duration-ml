import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

SAVED_MODEL_PATH = "models/ridge_taxi_model.joblib"
DEFAULT_EVAL_PATH = "Data/val.csv"


def evaluate_dataset(csv_path: str, model_path: str):
    print(f"Loading model from: {model_path}")
    pipeline = joblib.load(model_path)

    print(f"Loading evaluation dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    if "trip_duration" not in df.columns:
        raise ValueError("Evaluation CSV must include 'trip_duration' column.")

    y_true_log = np.log1p(df["trip_duration"].values)
    X = df.drop(columns=["trip_duration"])

    print("Running model inference...")
    y_pred_log = pipeline.predict(X)
    y_pred_clipped = np.clip(y_pred_log, np.log1p(30), np.log1p(24 * 3600))

    r2 = r2_score(y_true_log, y_pred_clipped)
    rmse = np.sqrt(mean_squared_error(y_true_log, y_pred_clipped))

    print("\n" + "=" * 45)
    print("         MODEL PERFORMANCE REPORT           ")
    print("=" * 45)
    print(f"Evaluated File : {csv_path}")
    print(f"Total Samples  : {len(df):,}")
    print(f"R2-Score       : {r2:.5f}")
    print(f"RMSE (log)     : {rmse:.5f}")
    print("=" * 45)


if __name__ == "__main__":
    target_csv = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EVAL_PATH
    evaluate_dataset(target_csv, SAVED_MODEL_PATH)