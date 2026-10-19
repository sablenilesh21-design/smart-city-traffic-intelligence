
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- Page setup ----------------
st.set_page_config(
    page_title="Smart City Traffic Intelligence",
    page_icon="🚦",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "traffic_5min.csv"
MODEL_PATH = BASE_DIR / "traffic_congestion_model.joblib"

st.title("🚦 Smart City Traffic Intelligence")
st.caption(
    "Traffic analysis and count-based congestion prediction "
    "using Pune traffic camera data from January 2023."
)

# ---------------- Load saved files ----------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_resource
def load_model(path):
    return joblib.load(path)

try:
    df = load_data(DATA_PATH)
    saved = load_model(MODEL_PATH)
    model = saved["model"]
except Exception as e:
    st.error(f"Could not load the project files: {e}")
    st.info(
        "Ensure traffic_5min.csv and "
        "traffic_congestion_model.joblib are beside app.py."
    )
    st.stop()

# ---------------- Sidebar ----------------
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Choose a section",
    ["Dataset Overview", "Traffic Analysis", "Congestion Prediction"]
)

# ---------------- Dataset Overview ----------------
if page == "Dataset Overview":
    st.header("Dataset Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Processed Traffic Intervals", f"{len(df):,}")
    c1.caption("Each row represents a five-minute interval for a camera and direction.")
    c2.metric("Intersections", df["Intersection"].nunique())
    c3.metric("Camera IDs", df["Camera"].nunique())

    st.subheader("Congestion category distribution")
    category_counts = (
        df["Congestion"].value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
    )
    st.bar_chart(category_counts)

    st.subheader("Sample records")
    st.dataframe(df.head(20), use_container_width=True)

    st.caption(
        "Categories are defined using vehicle-count thresholds, "
        "not independently measured road congestion."
    )

# ---------------- Traffic Analysis ----------------
elif page == "Traffic Analysis":
    st.header("Traffic Analysis")

    intersection = st.selectbox(
        "Select intersection",
        sorted(df["Intersection"].unique())
    )

    subset = df[df["Intersection"] == intersection].copy()

    st.subheader(f"Vehicle counts by hour — {intersection}")
    hourly = subset.groupby("Hour")["new_vehicles"].sum()
    st.line_chart(hourly)

    st.subheader("Vehicle types")
    vehicle_cols = [
        "car_change",
        "motorbike_change",
        "bus_change",
        "truck_change"
    ]
    vehicle_totals = subset[vehicle_cols].sum().rename({
        "car_change": "Cars",
        "motorbike_change": "Motorbikes",
        "bus_change": "Buses",
        "truck_change": "Trucks"
    })
    st.bar_chart(vehicle_totals)

    st.subheader("Congestion categories at this intersection")
    category_counts = (
        subset["Congestion"].value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
    )
    st.bar_chart(category_counts)

    st.subheader("Records over time")
    daily = subset.groupby(subset["Timestamp"].dt.date)["new_vehicles"].sum()
    st.line_chart(daily)

# ---------------- Congestion Prediction ----------------
else:
    st.header("Predict a congestion category")
    st.write(
        "Choose the conditions to estimate a count-based category. "
        "This is a model prediction, not a live traffic reading."
    )

    intersections = sorted(df["Intersection"].unique())
    directions = sorted(df["Direction"].unique())

    with st.form("prediction_form"):
        selected_intersection = st.selectbox(
            "Intersection", intersections
        )
        selected_direction = st.selectbox(
            "Direction", directions
        )
        hour = st.slider("Hour of day", 0, 23, 10)
        day_name = st.selectbox(
            "Day of week",
            ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"],
            index=2
        )
        previous_traffic = st.number_input(
            "Vehicle count in the previous five-minute interval",
            min_value=0.0,
            value=25.0,
            step=1.0
        )

        submitted = st.form_submit_button("Predict congestion")

    if submitted:
        day_of_week = [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"
        ].index(day_name)

        input_data = pd.DataFrame([{
            "Hour": hour,
            "DayOfWeek": day_of_week,
            "IsWeekend": int(day_of_week >= 5),
            "PreviousTraffic": previous_traffic,
            "Intersection": selected_intersection,
            "Direction": selected_direction
        }])

        prediction = model.predict(input_data)[0]

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = model.named_steps["classifier"].classes_
            confidence = probabilities[list(classes).index(prediction)]
        else:
            confidence = None

        st.subheader("Prediction result")
        st.metric("Predicted category", prediction)

        if confidence is not None:
            st.write(f"Model confidence for this category: {confidence:.1%}")

        st.caption(
            "The model predicts categories derived from historical "
            "vehicle counts. It does not measure current road conditions."
        )
