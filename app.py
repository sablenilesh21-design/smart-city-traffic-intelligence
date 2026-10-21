
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Smart City Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "traffic_5min.csv"
MODEL_PATH = BASE_DIR / "traffic_congestion_model.joblib"

# =========================================================
# CUSTOM DESIGN
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0b1220 0%, #101b2d 100%);
        color: #e8eef8;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 18, 32, 0.85);
    }

    [data-testid="stSidebar"] {
        background: #0a1322;
        border-right: 1px solid #24334a;
    }

    [data-testid="stSidebar"] * {
        color: #e8eef8;
    }

    .hero {
        padding: 28px 30px;
        border: 1px solid #2b405b;
        border-radius: 20px;
        background: linear-gradient(115deg, #14243a, #123b45);
        margin-bottom: 24px;
    }

    .hero h1 {
        font-size: 2.25rem;
        font-weight: 800;
        color: #f2f7ff;
        margin-bottom: 8px;
    }

    .hero p {
        color: #b9cbdc;
        font-size: 1rem;
        margin-bottom: 0;
    }

    .section-heading {
        color: #66e3ce;
        font-size: 1.3rem;
        font-weight: 700;
        margin-top: 16px;
        margin-bottom: 12px;
    }

    div[data-testid="stMetric"] {
        background: #14243a;
        border: 1px solid #2b405b;
        padding: 18px;
        border-radius: 15px;
    }

    div[data-testid="stMetricLabel"] {
        color: #a9bdd2;
    }

    div[data-testid="stMetricValue"] {
        color: #66e3ce;
        font-weight: 800;
    }

    div[data-testid="stForm"] {
        background: #14243a;
        border: 1px solid #2b405b;
        border-radius: 16px;
        padding: 22px;
    }

    .result-card {
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #2b405b;
        background: #14243a;
        text-align: center;
        margin-top: 16px;
    }

    .result-value {
        font-size: 2.3rem;
        font-weight: 800;
        margin: 10px 0;
    }

    .result-caption {
        color: #b9cbdc;
        font-size: 0.9rem;
    }

    .info-card {
        background: #14243a;
        border: 1px solid #2b405b;
        border-radius: 14px;
        padding: 18px;
        height: 100%;
    }

    .info-card h3 {
        color: #66e3ce;
        font-size: 1.05rem;
    }

    .info-card p {
        color: #c1cede;
        font-size: 0.92rem;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        background: #20bfa9;
        color: #071521;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background: #66e3ce;
        color: #071521;
        border: none;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #2b405b;
        border-radius: 12px;
    }

    hr {
        border-color: #2b405b;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA AND MODEL
# =========================================================
@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    data["Timestamp"] = pd.to_datetime(data["Timestamp"])
    data["Date"] = pd.to_datetime(data["Date"])
    return data

@st.cache_resource
def load_model(path):
    return joblib.load(path)

try:
    df = load_data(DATA_PATH)
    saved = load_model(MODEL_PATH)
    model = saved["model"]
except Exception as e:
    st.error(f"Could not load project files: {e}")
    st.info(
        "Ensure traffic_5min.csv and "
        "traffic_congestion_model.joblib are in the same "
        "GitHub repository folder as app.py."
    )
    st.stop()

required_columns = [
    "Timestamp", "Intersection", "Camera", "Direction",
    "Hour", "DayOfWeek", "IsWeekend", "new_vehicles",
    "Congestion", "car_change", "motorbike_change",
    "bus_change", "truck_change"
]

missing_columns = [c for c in required_columns if c not in df.columns]
if missing_columns:
    st.error(f"Required dataset columns are missing: {missing_columns}")
    st.stop()

categories = ["Low", "Medium", "High"]
vehicle_cols = [
    "car_change", "motorbike_change",
    "bus_change", "truck_change"
]

# Derive previous-interval traffic using the same five-minute rule.
# Do not overwrite the dataset if this feature already exists.
if "PreviousTraffic" not in df.columns:
    df = df.sort_values(["Camera", "Direction", "Timestamp"]).copy()
    grouped = df.groupby(["Camera", "Direction"], sort=False)
    previous_count = grouped["new_vehicles"].shift(1)
    previous_time = grouped["Timestamp"].shift(1)

    valid_gap = (
        df["Timestamp"] - previous_time
    ) == pd.Timedelta(minutes=5)

    df["PreviousTraffic"] = previous_count.where(valid_gap, 0).fillna(0)

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
with st.sidebar:
    st.markdown("## 🚦 Traffic Intelligence")
    st.caption("Pune • Historical Traffic Analytics")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "📊 Traffic Analysis",
            "🤖 Congestion Prediction",
            "ℹ️ About Project"
        ],
        label_visibility="visible"
    )

    st.divider()
    st.caption("Data period: January 2023")
    st.caption("Prediction type: Count-based ML")

# =========================================================
# HERO HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <h1>🚦 Smart City Traffic Intelligence</h1>
    <p>Explore historical traffic patterns, understand vehicle flow,
    and estimate count-based congestion categories using machine learning.</p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# DASHBOARD HOME
# =========================================================
if page == "🏠 Dashboard":
    st.markdown('<div class="section-heading">City Traffic at a Glance</div>',
                unsafe_allow_html=True)

    total_vehicles = int(df["new_vehicles"].sum())
    intersection_count = df["Intersection"].nunique()
    camera_count = df["Camera"].nunique()
    high_pct = (df["Congestion"].eq("High").mean() * 100)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Processed Intervals", f"{len(df):,}")
    c2.metric("Intersections", intersection_count)
    c3.metric("Camera IDs", camera_count)
    c4.metric("High-Category Records", f"{high_pct:.1f}%")

    st.caption(
        "Each record represents a five-minute interval for one camera "
        "and direction. Vehicle totals are reconstructed from the source data."
    )

    left, right = st.columns([1.15, 0.85])

    with left:
        st.markdown(
            '<div class="section-heading">Traffic by Hour</div>',
            unsafe_allow_html=True
        )
        hourly = df.groupby("Hour")["new_vehicles"].sum().reindex(
            range(24), fill_value=0
        )
        st.line_chart(hourly, color="#20bfa9")
        st.caption("Total recorded vehicle counts grouped by hour of day.")

    with right:
        st.markdown(
            '<div class="section-heading">Category Distribution</div>',
            unsafe_allow_html=True
        )
        counts = df["Congestion"].value_counts().reindex(
            categories, fill_value=0
        )
        st.bar_chart(counts, color="#4f8cff")
        st.caption(
            "Categories are based on vehicle-count thresholds, "
            "not independent measurements of road congestion."
        )

    st.markdown('<div class="section-heading">Explore the System</div>',
                unsafe_allow_html=True)

    a, b, c = st.columns(3)

    with a:
        st.markdown("""
        <div class="info-card">
            <h3>📍 Historical Data</h3>
            <p>Explore traffic observations from selected Pune intersections,
            cameras, and directions.</p>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="info-card">
            <h3>📈 Traffic Insights</h3>
            <p>Compare hourly patterns, vehicle types, daily totals,
            and count-based categories.</p>
        </div>
        """, unsafe_allow_html=True)

    with c:
        st.markdown("""
        <div class="info-card">
            <h3>🧠 ML Prediction</h3>
            <p>Estimate a category from the selected location, time,
            day, direction, and previous traffic count.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Recent Records</div>',
                unsafe_allow_html=True)
    st.dataframe(
        df.sort_values("Timestamp", ascending=False).head(10),
        use_container_width=True,
        hide_index=True
    )

# =========================================================
# TRAFFIC ANALYSIS
# =========================================================
elif page == "📊 Traffic Analysis":
    st.markdown('<div class="section-heading">Explore Traffic Patterns</div>',
                unsafe_allow_html=True)

    intersections = sorted(df["Intersection"].dropna().unique())
    selected_intersection = st.selectbox(
        "Choose an intersection", intersections
    )

    subset = df[df["Intersection"] == selected_intersection].copy()

    direction_options = ["All directions"] + sorted(
        subset["Direction"].dropna().unique()
    )
    selected_direction = st.selectbox(
        "Filter by direction", direction_options
    )

    if selected_direction != "All directions":
        subset = subset[subset["Direction"] == selected_direction]

    if subset.empty:
        st.warning("No records match the selected filters.")
        st.stop()

    total = int(subset["new_vehicles"].sum())
    average = subset["new_vehicles"].mean()
    peak_hour = int(subset.groupby("Hour")["new_vehicles"].sum().idxmax())

    c1, c2, c3 = st.columns(3)
    c1.metric("Vehicle Count Total", f"{total:,}")
    c2.metric("Average per Record", f"{average:.1f}")
    c3.metric("Busiest Recorded Hour", f"{peak_hour:02d}:00")

    left, right = st.columns(2)

    with left:
        st.markdown("#### Vehicle counts by hour")
        hourly = subset.groupby("Hour")["new_vehicles"].sum().reindex(
            range(24), fill_value=0
        )
        st.bar_chart(hourly, color="#20bfa9")

    with right:
        st.markdown("#### Vehicle type breakdown")
        totals = subset[vehicle_cols].sum().rename({
            "car_change": "Cars",
            "motorbike_change": "Motorbikes",
            "bus_change": "Buses",
            "truck_change": "Trucks"
        })
        st.bar_chart(totals, color="#4f8cff")

    left, right = st.columns(2)

    with left:
        st.markdown("#### Count-based categories")
        category_counts = subset["Congestion"].value_counts().reindex(
            categories, fill_value=0
        )
        st.bar_chart(category_counts, color="#e5a94d")

    with right:
        st.markdown("#### Daily traffic trend")
        daily = (
            subset.groupby(subset["Timestamp"].dt.date)["new_vehicles"]
            .sum()
        )
        st.line_chart(daily, color="#a78bfa")

    st.markdown("#### Filtered data preview")
    st.dataframe(
        subset.sort_values("Timestamp", ascending=False).head(100),
        use_container_width=True,
        hide_index=True
    )
    st.caption("Showing up to 100 recent matching records.")

# =========================================================
# CONGESTION PREDICTION
# =========================================================
elif page == "🤖 Congestion Prediction":
    st.markdown('<div class="section-heading">Estimate a Traffic Category</div>',
                unsafe_allow_html=True)

    st.info(
        "This tool predicts a category derived from historical vehicle counts. "
        "It does not access live traffic-camera feeds or real-time road speeds."
    )

    intersections = sorted(df["Intersection"].dropna().unique())
    directions = sorted(df["Direction"].dropna().unique())
    day_names = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ]

    with st.form("prediction_form"):
        st.markdown("#### 1. Select location and time")
        col1, col2 = st.columns(2)

        with col1:
            selected_intersection = st.selectbox(
                "Intersection", intersections
            )
            hour = st.slider("Hour of day", 0, 23, 18)

        with col2:
            selected_direction = st.selectbox(
                "Direction", directions
            )
            day_name = st.selectbox(
                "Day of week", day_names, index=2
            )

        day_of_week = day_names.index(day_name)

        # Estimate a typical previous-interval count from matching
        # historical observations. The user can still override it.
        historical = df[
            (df["Intersection"] == selected_intersection)
            & (df["Direction"] == selected_direction)
            & (df["Hour"] == hour)
            & (df["DayOfWeek"] == day_of_week)
        ]

        if not historical.empty:
            automatic_count = float(historical["PreviousTraffic"].median())
        else:
            fallback = df[
                (df["Intersection"] == selected_intersection)
                & (df["Direction"] == selected_direction)
            ]
            automatic_count = (
                float(fallback["PreviousTraffic"].median())
                if not fallback.empty else float(df["PreviousTraffic"].median())
            )

        st.markdown("#### 2. Previous traffic count")
        use_automatic = st.checkbox(
            "Use estimated historical count automatically",
            value=True
        )

        if use_automatic:
            previous_traffic = automatic_count
            st.metric(
                "Estimated previous five-minute count",
                f"{previous_traffic:.0f}"
            )
            st.caption(
                "Calculated from matching historical records where available. "
                "It is an estimate, not a live count."
            )
        else:
            previous_traffic = st.number_input(
                "Enter previous five-minute vehicle count",
                min_value=0.0,
                value=max(0.0, automatic_count),
                step=1.0
            )

        submitted = st.form_submit_button(
            "✨ Predict congestion", use_container_width=True
        )

    if submitted:
        input_data = pd.DataFrame([{
            "Hour": hour,
            "DayOfWeek": day_of_week,
            "IsWeekend": int(day_of_week >= 5),
            "PreviousTraffic": previous_traffic,
            "Intersection": selected_intersection,
            "Direction": selected_direction
        }])

        try:
            prediction = model.predict(input_data)[0]
            confidence = None

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0]

                if hasattr(model, "named_steps"):
                    classifier = model.named_steps.get("classifier")
                    classes = classifier.classes_ if classifier is not None else model.classes_
                else:
                    classes = model.classes_

                confidence = probabilities[list(classes).index(prediction)]

            result_colors = {
                "Low": "#20bfa9",
                "Medium": "#e5a94d",
                "High": "#ff6b6b"
            }
            result_color = result_colors.get(prediction, "#66e3ce")

            st.markdown(f"""
            <div class="result-card">
                <div class="result-caption">PREDICTED CATEGORY</div>
                <div class="result-value" style="color:{result_color}">
                    {prediction}
                </div>
                <div class="result-caption">
                    {selected_intersection} · {selected_direction}
                    · {hour:02d}:00 · {day_name}
                </div>
            </div>
            """, unsafe_allow_html=True)

            if confidence is not None:
                st.markdown("#### Model confidence")
                st.progress(float(confidence))
                st.write(f"Estimated class probability: **{confidence:.1%}**")
                st.caption(
                    "This is the model's estimated probability for its "
                    "predicted class, not a guarantee of real-world accuracy."
                )

            with st.expander("View model inputs"):
                st.dataframe(input_data, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# =========================================================
# ABOUT PROJECT
# =========================================================
elif page == "ℹ️ About Project":
    st.markdown('<div class="section-heading">About This Project</div>',
                unsafe_allow_html=True)

    st.markdown("""
    This project explores historical traffic counts collected at selected
    intersections in Pune and uses a machine-learning classifier to estimate
    a traffic category from time, location, direction, and previous traffic.
    """)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="info-card">
            <h3>📂 Dataset</h3>
            <p>Historical traffic-camera observations from January 2023.
            Processed records represent five-minute intervals for individual
            cameras and directions.</p>
            <p>Source: Heterogeneous Traffic Count Dataset, Pune.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="info-card">
            <h3>🧠 Machine Learning</h3>
            <p>A saved Decision Tree pipeline uses historical traffic features
            to classify records into Low, Medium, or High count-based categories.</p>
            <p>Reported test accuracy during development: approximately 87%.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Important limitations")
    st.warning(
        "The categories are generated from vehicle-count thresholds. "
        "They are not ground-truth congestion labels measured from road speed "
        "or travel time. The data is historical, not live. The reported model "
        "accuracy describes performance on the project's held-out test data."
    )

    st.markdown("#### Project workflow")
    st.markdown("""
    1. Clean and process the traffic-camera data.
    2. Aggregate observations into five-minute intervals.
    3. Explore hourly, daily, and vehicle-type patterns.
    4. Train and evaluate a classification model.
    5. Present the analysis and predictions in this dashboard.
    """)

# =========================================================
# FOOTER
# =========================================================
st.divider()
st.caption(
    "Smart City Traffic Intelligence • Academic data science project • "
    "Historical Pune traffic data • Not a live traffic monitoring service"
)
