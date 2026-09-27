# NYC Taxi Trip Duration Prediction

An end-to-end machine learning project for predicting NYC taxi trip duration from pickup and drop-off coordinates, pickup time, passenger information, and trip metadata. The workflow includes spatial and temporal feature engineering, model benchmarking, hyperparameter tuning, final model training, and a Streamlit inference application.

## Project Overview

The final estimator is a `GradientBoostingRegressor` trained to predict `log1p(trip_duration)`. The scikit-learn pipeline applies the custom `TaxiFeatureEngineer`, scales numeric features, one-hot encodes categorical features, and then fits the regressor.

| Stage | Data used | Purpose |
| --- | --- | --- |
| Benchmarking | Train, then Validation | Compare baseline estimators |
| Hyperparameter tuning | Train with 3-fold cross-validation; Validation for evaluation | Search XGBoost and Gradient Boosting configurations |
| Final training | Train + Validation | Refit the selected Gradient Boosting model |
| Final evaluation | Held-out Test Set, kept separately | Unbiased evaluation after model development |

The final training script does not load or use the Test Set.

## Pipeline

```text
Raw trip records
	|
	v
Data cleaning and NYC coordinate filtering
	|
	v
Spatial and temporal feature engineering
	|
	v
Scaling and one-hot encoding
	|
	v
Benchmarking and cross-validated tuning
	|
	v
Final Gradient Boosting fit on Train + Validation
	|
	v
Saved model pipeline and Streamlit inference app
```

## Feature Engineering

The reusable transformer in [`src/features.py`](src/features.py) derives features from trip coordinates and pickup timestamps:

- Haversine distance, in kilometers, between pickup and drop-off.
- Manhattan distance approximation for grid-based travel.
- Bearing from pickup to drop-off.
- Log-transformed distance using `log1p(haversine_dist)`.
- Cyclical sine/cosine encodings for pickup hour and day of week.
- Weekend indicator.
- Bounded pickup and drop-off coordinates for spatial feature calculations.

Training and evaluation scripts keep trips with durations from 60 seconds to 24 hours and coordinates within the configured NYC bounds.

## Benchmark Results

The following are the recorded validation-stage results from a project benchmark run. Metrics are reported on the original duration scale except RMSLE, which is calculated in log-target space.

| Model | R² | MAE (minutes) | RMSE (minutes) | RMSLE |
| --- | ---: | ---: | ---: | ---: |
| Ridge | 0.69047 | 3.88 | 6.66 | 0.44232 |
| Random Forest | 0.80737 | 3.37 | 5.25 | 0.38609 |
| Gradient Boosting | 0.79988 | 3.24 | 5.35 | 0.36693 |
| XGBoost | 0.81409 | 3.38 | 5.16 | 0.37200 |

These values describe one validation run and are not results from the held-out Test Set.

## Hyperparameter Tuning

[`tune.py`](tune.py) runs `RandomizedSearchCV` for XGBoost and Gradient Boosting. Each search evaluates 25 randomly selected configurations with 3-fold cross-validation, `random_state=42`, and negative mean squared error on the log-transformed target. The script then reports each best estimator's validation metrics.

The selected Gradient Boosting configuration used for final training is:

| Parameter | Value |
| --- | ---: |
| `n_estimators` | 400 |
| `learning_rate` | 0.02 |
| `max_depth` | 6 |
| `min_samples_split` | 20 |
| `min_samples_leaf` | 10 |
| `max_features` | `sqrt` |
| `subsample` | 0.8 |
| `loss` | `huber` |

### Tuned Gradient Boosting Validation Results

| Metric | Value |
| --- | ---: |
| R² | 0.80191 |
| MAE | 3.25 min |
| RMSE | 5.32 min |
| RMSLE | 0.37098 |
| R² (log target) | 0.77562 |
| RMSE (log target) | 0.37098 |

Validation results are part of model development and must not be interpreted as final Test Set performance. After the configuration is selected, [`final_train.py`](final_train.py) refits the model using the combined Train and Validation datasets.

## Repository Structure

```text
nyc_taxi_project/
├── Data/
│   ├── train.csv
│   └── val.csv
├── models/
│   ├── final_gradient_boosting_taxi_model.joblib
│   └── final_model_config.json
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── evaluate.py
│   ├── features.py
│   └── train.py
├── app.py
├── bench.py
├── final_train.py
├── tune.py
├── requirements.txt
└── README.md
```

The held-out Test Set is intentionally kept separate and is not required to train the final model or run the application. Generated benchmark and tuning outputs are written to `models/`.

## Quickstart

Run commands from the project root.

### 1. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

The benchmark skips XGBoost if it is unavailable. Hyperparameter tuning requires XGBoost. Installing dependencies requires access to the configured Python package index.

### 2. Prepare the data

Place the development datasets in `Data/`:

```text
Data/
├── train.csv
└── val.csv
```

Both files must contain the trip target and the input columns used by the pipeline. Keep the held-out Test Set outside model training and selection.

### 3. Benchmark models

```powershell
python bench.py
```

Compares Ridge, Random Forest, Gradient Boosting, and (when installed) XGBoost. The comparison CSV is written to `models/model_comparison.csv`.

### 4. Tune model hyperparameters

```powershell
python tune.py
```

Runs randomized cross-validated searches for XGBoost and Gradient Boosting and writes the search results and tuned model artifacts under `models/`. XGBoost must be installed for this step.

### 5. Train the final model

```powershell
python final_train.py
```

Fits the selected Gradient Boosting configuration on Train + Validation and saves:

- `models/final_gradient_boosting_taxi_model.joblib`
- `models/final_model_config.json`
- `models/final_training_report.csv` (training-set diagnostics, not generalization metrics)

The script intentionally does not read the Test Set.

### 6. Launch the Streamlit application

```powershell
streamlit run app.py
```

The application loads the saved final pipeline and provides route presets, coordinate inputs, trip metadata, a map, and an estimated trip duration in minutes and seconds. Train the final model before launching the app if the model artifact is not present.

## Technologies

- Python, pandas, and NumPy
- scikit-learn and XGBoost
- Joblib
- Streamlit
- Jupyter

## Skills Demonstrated

- Data cleaning and leakage-aware Train/Validation/Test separation.
- Spatial and temporal feature engineering.
- scikit-learn pipelines and `ColumnTransformer` preprocessing.
- Model benchmarking and regression evaluation.
- Cross-validation and randomized hyperparameter search.
- Model persistence and interactive inference deployment.
