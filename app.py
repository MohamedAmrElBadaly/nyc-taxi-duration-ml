import os
import sys

# إضافة مجلد src لمسار بايثون أولاً
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="NYC Taxi Trip Duration Predictor",
    page_icon="🚕",
    layout="wide"
)

MODEL_PATH = "models/ridge_taxi_model.joblib"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

st.title("🚕 NYC Taxi Trip Duration Predictor")
st.markdown(
    "Predict expected taxi travel time across New York City using regularized linear regression (`Ridge(alpha=1.0)`) "
    "and real-time spatial-temporal feature extraction."
)

st.sidebar.header("🕒 Trip Date & Settings")
trip_date = st.sidebar.date_input("Pickup Date", pd.to_datetime("2016-03-15"))
trip_time = st.sidebar.time_input("Pickup Time", pd.to_datetime("14:30:00").time())
passenger_count = st.sidebar.slider("Passenger Count", min_value=1, max_value=6, value=1)
vendor_id = st.sidebar.selectbox("Vendor ID", [1, 2], index=0)
store_and_fwd = st.sidebar.selectbox("Store and Forward Flag", ["N", "Y"], index=0)

st.subheader("📍 Coordinates Selection")
preset = st.selectbox(
    "Choose a Preset Route or Enter Custom Coordinates:",
    [
        "Times Square to JFK Airport",
        "Central Park to Wall Street",
        "LaGuardia Airport to Midtown",
        "Custom"
    ]
)

presets_dict = {
    "Times Square to JFK Airport": {
        "p_lat": 40.7580, "p_lon": -73.9855,
        "d_lat": 40.6413, "d_lon": -73.7781
    },
    "Central Park to Wall Street": {
        "p_lat": 40.785091, "p_lon": -73.968285,
        "d_lat": 40.707491, "d_lon": -74.011276
    },
    "LaGuardia Airport to Midtown": {
        "p_lat": 40.7769, "p_lon": -73.8740,
        "d_lat": 40.7549, "d_lon": -73.9840
    }
}

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🟢 Pickup Location")
    default_plat = presets_dict[preset]["p_lat"] if preset != "Custom" else 40.7580
    default_plon = presets_dict[preset]["p_lon"] if preset != "Custom" else -73.9855
    pickup_lat = st.number_input("Pickup Latitude", value=default_plat, format="%.6f")
    pickup_lon = st.number_input("Pickup Longitude", value=default_plon, format="%.6f")

with col2:
    st.markdown("### 🔴 Dropoff Location")
    default_dlat = presets_dict[preset]["d_lat"] if preset != "Custom" else 40.6413
    default_dlon = presets_dict[preset]["d_lon"] if preset != "Custom" else -73.7781
    dropoff_lat = st.number_input("Dropoff Latitude", value=default_dlat, format="%.6f")
    dropoff_lon = st.number_input("Dropoff Longitude", value=default_dlon, format="%.6f")

map_data = pd.DataFrame({
    "lat": [pickup_lat, dropoff_lat],
    "lon": [pickup_lon, dropoff_lon]
})
st.map(map_data, zoom=11)

if st.button("🚀 Calculate Estimated Duration", type="primary"):
    try:
        pipeline = load_model()

        pickup_datetime = f"{trip_date} {trip_time}"

        input_data = pd.DataFrame([{
            "id": "app_query",
            "vendor_id": vendor_id,
            "pickup_datetime": pickup_datetime,
            "passenger_count": passenger_count,
            "pickup_longitude": pickup_lon,
            "pickup_latitude": pickup_lat,
            "dropoff_longitude": dropoff_lon,
            "dropoff_latitude": dropoff_lat,
            "store_and_fwd_flag": store_and_fwd
        }])

        log_pred = pipeline.predict(input_data)[0]
        pred_seconds = np.expm1(log_pred)
        pred_minutes = pred_seconds / 60.0

        st.success("### Prediction Result")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Estimated Time", f"{pred_minutes:.1f} mins")
        metric_col2.metric("Total Seconds", f"{int(pred_seconds)} sec")
        metric_col3.metric("Passenger(s)", f"{passenger_count}")

    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")