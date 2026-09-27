import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Add src directory to Python path so custom feature engineering is available.
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NYC Taxi Trip Duration Predictor",
    page_icon="🚕",
    layout="wide",
)

MODEL_PATH = "models/final_gradient_boosting_taxi_model.joblib"

NYC_BOUNDS = {
    "lat_min": 40.57,
    "lat_max": 40.92,
    "lon_min": -74.15,
    "lon_max": -73.70,
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Final model not found at: {MODEL_PATH}"
        )
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🚕 NYC Taxi Trip Duration Predictor")
st.caption(
    "End-to-end machine learning system with spatial-temporal feature "
    "engineering and a tuned Gradient Boosting regression model."
)

st.info(
    "The final model was trained on the cleaned Train + Validation datasets. "
    "The held-out Test Set is not used by this application."
)


# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------
st.sidebar.header("🕒 Trip Settings")

trip_date = st.sidebar.date_input(
    "Pickup Date",
    value=pd.Timestamp("2016-03-15").date(),
)

trip_time = st.sidebar.time_input(
    "Pickup Time",
    value=pd.Timestamp("14:30:00").time(),
)

passenger_count = st.sidebar.slider(
    "Passenger Count",
    min_value=1,
    max_value=6,
    value=1,
)

vendor_id = st.sidebar.selectbox(
    "Vendor ID",
    options=[1, 2],
    index=0,
)

store_and_fwd = st.sidebar.selectbox(
    "Store and Forward Flag",
    options=["N", "Y"],
    index=0,
)


# ---------------------------------------------------------------------------
# Route presets
# ---------------------------------------------------------------------------
st.subheader("📍 Trip Route")

preset = st.selectbox(
    "Choose a preset route or enter custom coordinates:",
    options=[
        "Times Square to JFK Airport",
        "Central Park to Wall Street",
        "LaGuardia Airport to Midtown",
        "Custom",
    ],
)

presets_dict = {
    "Times Square to JFK Airport": {
        "p_lat": 40.7580,
        "p_lon": -73.9855,
        "d_lat": 40.6413,
        "d_lon": -73.7781,
    },
    "Central Park to Wall Street": {
        "p_lat": 40.785091,
        "p_lon": -73.968285,
        "d_lat": 40.707491,
        "d_lon": -74.011276,
    },
    "LaGuardia Airport to Midtown": {
        "p_lat": 40.7769,
        "p_lon": -73.8740,
        "d_lat": 40.7549,
        "d_lon": -73.9840,
    },
}


if preset != "Custom":
    selected_route = presets_dict[preset]
else:
    selected_route = {
        "p_lat": 40.7580,
        "p_lon": -73.9855,
        "d_lat": 40.6413,
        "d_lon": -73.7781,
    }


col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🟢 Pickup")
    pickup_lat = st.number_input(
        "Pickup Latitude",
        value=float(selected_route["p_lat"]),
        format="%.6f",
    )
    pickup_lon = st.number_input(
        "Pickup Longitude",
        value=float(selected_route["p_lon"]),
        format="%.6f",
    )

with col2:
    st.markdown("### 🔴 Dropoff")
    dropoff_lat = st.number_input(
        "Dropoff Latitude",
        value=float(selected_route["d_lat"]),
        format="%.6f",
    )
    dropoff_lon = st.number_input(
        "Dropoff Longitude",
        value=float(selected_route["d_lon"]),
        format="%.6f",
    )


# ---------------------------------------------------------------------------
# Coordinate validation
# ---------------------------------------------------------------------------
def coordinates_are_valid(lat, lon):
    return (
        NYC_BOUNDS["lat_min"] <= lat <= NYC_BOUNDS["lat_max"]
        and NYC_BOUNDS["lon_min"] <= lon <= NYC_BOUNDS["lon_max"]
    )


pickup_valid = coordinates_are_valid(pickup_lat, pickup_lon)
dropoff_valid = coordinates_are_valid(dropoff_lat, dropoff_lon)

if not pickup_valid:
    st.warning("Pickup coordinates are outside the configured NYC bounds.")

if not dropoff_valid:
    st.warning("Dropoff coordinates are outside the configured NYC bounds.")


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------
map_data = pd.DataFrame(
    {
        "lat": [pickup_lat, dropoff_lat],
        "lon": [pickup_lon, dropoff_lon],
    }
)

st.map(map_data, zoom=11)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if st.button(
    "🚀 Calculate Estimated Duration",
    type="primary",
    use_container_width=True,
):
    if not pickup_valid or not dropoff_valid:
        st.error("Please enter valid NYC coordinates before predicting.")
        st.stop()

    try:
        pipeline = load_model()

        pickup_datetime = f"{trip_date} {trip_time}"

        input_data = pd.DataFrame(
            [
                {
                    "id": "app_query",
                    "vendor_id": vendor_id,
                    "pickup_datetime": pickup_datetime,
                    "passenger_count": passenger_count,
                    "pickup_longitude": pickup_lon,
                    "pickup_latitude": pickup_lat,
                    "dropoff_longitude": dropoff_lon,
                    "dropoff_latitude": dropoff_lat,
                    "store_and_fwd_flag": store_and_fwd,
                }
            ]
        )

        log_prediction = float(pipeline.predict(input_data)[0])

        predicted_seconds = float(np.expm1(log_prediction))
        predicted_seconds = float(
            np.clip(predicted_seconds, 30, 24 * 3600)
        )
        predicted_minutes = predicted_seconds / 60.0

        hours = int(predicted_minutes // 60)
        minutes = int(round(predicted_minutes % 60))

        st.success("### ✅ Prediction Ready")

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        metric_col1.metric(
            "Estimated Time",
            f"{predicted_minutes:.1f} min",
        )

        metric_col2.metric(
            "Estimated Seconds",
            f"{int(round(predicted_seconds)):,}",
        )

        metric_col3.metric(
            "Passengers",
            f"{passenger_count}",
        )

        if hours > 0:
            st.write(
                f"**Estimated trip duration:** approximately "
                f"{hours} h {minutes} min."
            )
        else:
            st.write(
                f"**Estimated trip duration:** approximately "
                f"{minutes} minutes."
            )

    except Exception as exc:
        st.error(f"Prediction Error: {exc}")


# ---------------------------------------------------------------------------
# Model information
# ---------------------------------------------------------------------------
with st.expander("ℹ️ Model & Project Information"):
    st.markdown(
        """
**Final Model:** Gradient Boosting Regressor

**Target Transformation:** `log1p(trip_duration)`

**Feature Engineering:**
- Haversine distance
- Manhattan distance
- Bearing
- Log-transformed distance
- Cyclical hour features
- Cyclical day-of-week features
- Weekend indicator

**Training Strategy:** Train + Validation

**Held-out Test Set:** Not used by the app or final training.
"""
    )
