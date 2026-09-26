# NYC Taxi Trip Duration Prediction

An end-to-end machine learning project for predicting New York City taxi trip duration. The pipeline combines spatial and temporal feature engineering with a regularized linear regression model.

## Project Overview

- **Estimator:** Ridge regression with `alpha=1.0`
- **Prediction target:** `log1p(trip_duration)`
- **Primary focus:** Domain-informed spatial and temporal features in a reusable scikit-learn pipeline
- **Evaluation metrics:** R² and RMSE on the log-duration scale

## Repository Structure

```text
.
├── Data/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── models/
│   └── ridge_taxi_model.joblib   # Created by training; excluded from Git
├── notebooks/
│   └── notebooks/
│       └── 01_eda.ipynb
├── src/
│   ├── __init__.py
│   ├── evaluate.py
│   ├── featuers.py               # Feature engineering transformer
│   └── train.py
├── .gitignore
├── requirements.txt
└── README.md
```

The feature module is currently named `featuers.py`; the training script imports it using that filename.

## Feature Engineering

The pipeline adds domain-specific features to the pickup and drop-off coordinates and pickup timestamp:

- **Haversine distance:** Great-circle distance between pickup and drop-off locations.
- **Manhattan distance:** Approximation of distance along a grid-based street network.
- **Bearing:** Direction of travel from pickup to drop-off.
- **Cyclical time features:** Sine and cosine encodings for pickup hour and day of week, plus a weekend indicator.
- **Coordinate bounds:** Coordinates are clipped to core NYC bounds before spatial features are calculated.
- **Preprocessing:** Numeric features are standardized and categorical features are one-hot encoded.

Training also removes rows with trip durations outside 60 seconds to 24 hours or coordinates outside the configured NYC bounds.

## Performance

Reported performance on the held-out datasets:

| Dataset | Samples | R² | RMSE (log scale) |
| --- | ---: | ---: | ---: |
| Validation | 3,000 | 0.66285 | 0.48380 |
| Test | 2,500 | 0.56157 | 0.48885 |

## Quickstart

### 1. Install dependencies

From the repository root, install the packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Prepare the data

Place `train.csv`, `val.csv`, and `test.csv` in the `Data/` directory. The training and evaluation scripts expect to be run from the repository root. Evaluation files must contain a `trip_duration` column.

### 3. Train the model

```bash
python src/train.py
```

Training reads `Data/train.csv` and saves the fitted pipeline to `models/ridge_taxi_model.joblib`.

### 4. Evaluate the model

Evaluate on the validation set (the default):

```bash
python src/evaluate.py
```

Or pass a dataset path, such as the test set:

```bash
python src/evaluate.py Data/test.csv
```

The evaluator reports sample count, R², and RMSE for `log1p(trip_duration)`. Predictions are clipped to a duration range of 30 seconds to 24 hours before metrics are calculated.
