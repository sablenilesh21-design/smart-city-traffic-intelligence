
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Smart City Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "traffic_5min.csv"
MODEL_PATH = BASE_DIR / "traffic_congestion_model.joblib"

PAGES = [
    "🏠 Dashboard",
    "📊 Traffic Analysis",
    "🤖 Congestion Prediction",
    "ℹ️ About Project"
]

# =====================================================
# 2. CUSTOM STYLING
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0b1220, #101b2d);
    color: #e8eef8;
}

[data-testid="stHeader"] {
    background: rgba(11, 18, 32, 0.9);
}

[data-testid="stSidebar"] {
    background: #0a1322;
    border-right: 1px solid #24334a;
}

[data-testid="stSidebar"] * {
    color: #e8eef8;
}

.hero {
    padding: 30px;
    border: 1px solid #2b405b;
    border-radius: 20px;
    background: linear-gradient(115deg, #14243a, #123b45);
    margin-bottom: 24px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}

.hero h1 {
    font-size: clamp(1.7rem, 3vw, 2.4rem);
    font-weight: 800;
    color: #f2f7ff;
    margin: 0 0 10px 0;
}

.hero p {
    color: #b9cbdc;
    font-size: 1rem;
    line-height: 1.6;
    margin: 0;
}

.section-heading {
    color: #66e3ce;
    font-size: 1.25rem;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 16px;
    padding-left: 12px;
    border-left: 4px solid #20bfa9;
}

[data-testid="stMetric"] {
    background: #14243a;
    border: 1px solid #2b405b;
    border-radius: 15px;
    padding: 18px;
}

[data-testid="stMetricLabel"] {
    color: #a9bdd2;
}

[data-testid="stMetricValue"] {
    color: #66e3ce;
    font-weight: 800;
}

.info-card {
    background: #14243a;
    border: 1px solid #2b405b;
    border-radius: 15px;
    padding: 20px;
    min-height: 155px;
    margin-bottom: 10px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    transition: all 0.2s ease-in-out;
}

.info-card:hover {
    border-color: #20bfa9;
    transform: translateY(-3px);
}

.info-card h3 {
    color: #66e3ce;
    font-size: 1.05rem;
    margin-top: 0;
}

.info-card p {
    color: #c1cede;
    font-size: 0.92rem;
    line-height: 1.55;
}

div.stButton > button,
div.stFormSubmitButton > button {
    width: 100%;
    min-height: 44px;
    background: #20bfa9;
    color: #071521;
    font-weight: 700;
    border: 1px solid #20bfa9;
    border-radius: 10px;
    transition: all 0.2s ease-in-out;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background: #66e3ce;
    color: #071521;
    border-color: #66e3ce;
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(32,191,169,0.2);
}

[data-testid="stForm"] {
    background: #14243a;
    border: 1px solid #2b405b;
    border-radius: 16px;
    padding: 22px;
}

.result-card {
    padding: 25px;
    border-radius: 16px;
    border: 1px solid #2b405b;
    background: #14243a;
    text-align: center;
    margin: 18px 0;
}

.result-caption {
    color: #b9cbdc;
    font-size: 0.9rem;
}

.result-value {
    font-size: 2.4rem;
    font-weight: 800;
    margin: 12px 0;
}

hr {
    border-color: #2b405b;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD DATA AND MODEL
# =====================================================
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)

    required = [
        "Timestamp", "Intersection", "Camera", "Direction",
        "Hour", "DayOfWeek", "IsWeekend", "new_vehicles",
        "Congestion", "car_change", "motorbike_change",
        "bus_change", "truck_change"
    ]

    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError(
            f"Dataset is missing these columns: {missing}"
        )

    data["Timestamp"] = pd.to_datetime(
        data["Timestamp"], errors="coerce"
    )

    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(
            data["Date"], errors="coerce"
        )

    numeric_columns = [
        "Hour", "DayOfWeek", "IsWeekend", "new_vehicles",
        "car_change", "motorbike_change", "bus_change", "truck_change"
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data = data.dropna(
        subset=[
            "Timestamp", "Intersection", "Camera", "Direction",
            "Hour", "DayOfWeek", "IsWeekend", "new_vehicles",
            "Congestion"
        ]
    ).copy()

    for column in numeric_columns:
        data[column] = data[column].fillna(0)

    data["Hour"] = data["Hour"].astype(int)
    data["DayOfWeek"] = data["DayOfWeek"].astype(int)
    data["IsWeekend"] = data["IsWeekend"].astype(int)

    return data


@st.cache_resource
def load_model(file_path):
    saved_object = joblib.load(file_path)

    # Support either a model saved directly or a dictionary
    # containing the model under the "model" key.
    if isinstance(saved_object, dict) and "model" in saved_object:
        return saved_object["model"]

    return saved_object


if not DATA_PATH.exists():
    st.error("Dataset file not found: traffic_5min.csv")
    st.info("Upload traffic_5min.csv beside app.py in your repository.")
    st.stop()

if not MODEL_PATH.exists():
    st.error("Model file not found: traffic_congestion_model.joblib")
    st.info(
        "Upload traffic_congestion_model.joblib beside app.py "
        "in your repository."
    )
    st.stop()

try:
    df = load_data(str(DATA_PATH))
    model = load_model(str(MODEL_PATH))
except Exception as error:
    st.error(f"Unable to load the project files: {error}")
    st.stop()

if df.empty:
    st.error("The dataset has no usable records.")
    st.stop()


# =====================================================
# 4. PREPARE PREVIOUS TRAFFIC FEATURE
# =====================================================
# The trained model expects PreviousTraffic. If it is already
# present in the CSV, keep it. Otherwise calculate it from the
# preceding five-minute record for the same camera and direction.

if "PreviousTraffic" not in df.columns:
    df = df.sort_values(
        ["Camera", "Direction", "Timestamp"]
    ).copy()

    group_columns = ["Camera", "Direction"]

    previous_count = df.groupby(group_columns)["new_vehicles"].shift(1)
    previous_timestamp = df.groupby(group_columns)["Timestamp"].shift(1)

    valid_previous_interval = (
        df["Timestamp"] - previous_timestamp
    ) == pd.Timedelta(minutes=5)

    df["PreviousTraffic"] = (
        previous_count.where(valid_previous_interval, 0).fillna(0)
    )

df["PreviousTraffic"] = pd.to_numeric(
    df["PreviousTraffic"], errors="coerce"
).fillna(0)

CATEGORIES = ["Low", "Medium", "High"]

VEHICLE_COLUMNS = [
    "car_change",
    "motorbike_change",
    "bus_change",
    "truck_change"
]

VEHICLE_LABELS = {
    "car_change": "Cars",
    "motorbike_change": "Motorbikes",
    "bus_change": "Buses",
    "truck_change": "Trucks"
}

DAY_NAMES = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]


# =====================================================
# 5. NAVIGATION HELPERS
# =====================================================
def navigate_to(page_name):
    st.session_state["navigation"] = page_name


if "navigation" not in st.session_state:
    st.session_state["navigation"] = PAGES[0]


# =====================================================
# 6. SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("## 🚦 Traffic Intelligence")
    st.caption("Pune | Historical Traffic Analytics")
    st.divider()

    selected_page = st.radio(
        "NAVIGATION",
        PAGES,
        key="navigation"
    )

    st.divider()

    st.markdown("### 📌 Project Information")
    st.caption("Dataset: January 2023")
    st.caption("Model: Machine learning classifier")
    st.caption("Mode: Historical data analysis")


# =====================================================
# 7. HEADER
# =====================================================
st.markdown("""
<div class="hero">
    <h1>🚦 Smart City Traffic Intelligence</h1>
    <p>
        Explore historical traffic patterns, analyse vehicle counts,
        and estimate traffic categories using machine learning.
    </p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# 8. DASHBOARD HOME
# =====================================================
if selected_page == "🏠 Dashboard":

    st.markdown(
        '<div class="section-heading">City Traffic at a Glance</div>',
        unsafe_allow_html=True
    )

    total_records = len(df)
    total_intersections = df["Intersection"].nunique()
    total_cameras = df["Camera"].nunique()

    high_percentage = (
        df["Congestion"].astype(str).eq("High").mean() * 100
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Processed Records", f"{total_records:,}")
    c2.metric("Intersections", f"{total_intersections:,}")
    c3.metric("Camera IDs", f"{total_cameras:,}")
    c4.metric("High-Category Records", f"{high_percentage:.1f}%")

    st.caption(
        "Each record represents a five-minute interval for one camera "
        "and direction. These figures describe the available dataset, "
        "not live city-wide traffic."
    )

    left, right = st.columns([1.2, 0.8])

    with left:
        st.markdown(
            '<div class="section-heading">Traffic by Recorded Hour</div>',
            unsafe_allow_html=True
        )

        hourly = (
            df.groupby("Hour")["new_vehicles"]
            .sum()
            .sort_index()
        )

        st.line_chart(hourly, color="#20bfa9")

        st.caption(
            "Only hours with observations are shown. Missing hours do not "
            "automatically mean zero traffic."
        )

    with right:
        st.markdown(
            '<div class="section-heading">Category Distribution</div>',
            unsafe_allow_html=True
        )

        category_counts = (
            df["Congestion"].astype(str)
            .value_counts()
            .reindex(CATEGORIES, fill_value=0)
        )

        st.bar_chart(category_counts, color="#4f8cff")

        st.caption(
            "Categories are based on vehicle-count thresholds rather than "
            "independently measured road speed."
        )

    st.markdown(
        '<div class="section-heading">Explore the System</div>',
        unsafe_allow_html=True
    )

    card1, card2, card3 = st.columns(3)

    with card1:
        st.markdown("""
        <div class="info-card">
            <h3>📍 Historical Data</h3>
            <p>
                Explore historical observations by intersection,
                camera, direction, and time.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.button(
            "Explore Historical Data →",
            key="historical_button",
            on_click=navigate_to,
            args=("📊 Traffic Analysis",),
            use_container_width=True
        )

    with card2:
        st.markdown("""
        <div class="info-card">
            <h3>📈 Traffic Insights</h3>
            <p>
                Compare recorded hourly patterns, vehicle types,
                and count-based categories.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.button(
            "View Traffic Insights →",
            key="insights_button",
            on_click=navigate_to,
            args=("📊 Traffic Analysis",),
            use_container_width=True
        )

    with card3:
        st.markdown("""
        <div class="info-card">
            <h3>🧠 ML Prediction</h3>
            <p>
                Estimate a Low, Medium, or High category using
                the saved machine-learning model.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.button(
            "Try Prediction →",
            key="prediction_button",
            on_click=navigate_to,
            args=("🤖 Congestion Prediction",),
            use_container_width=True
        )

    st.markdown(
        '<div class="section-heading">Recent Records</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        df.sort_values("Timestamp", ascending=False).head(10),
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# 9. TRAFFIC ANALYSIS
# =====================================================
elif selected_page == "📊 Traffic Analysis":

    st.markdown(
        '<div class="section-heading">Explore Traffic Patterns</div>',
        unsafe_allow_html=True
    )

    intersections = sorted(
        df["Intersection"].dropna().astype(str).unique().tolist()
    )

    selected_intersection = st.selectbox(
        "Choose an intersection",
        intersections
    )

    subset = df[
        df["Intersection"].astype(str) == selected_intersection
    ].copy()

    directions = sorted(
        subset["Direction"].dropna().astype(str).unique().tolist()
    )

    selected_direction = st.selectbox(
        "Choose direction",
        ["All directions"] + directions
    )

    if selected_direction != "All directions":
        subset = subset[
            subset["Direction"].astype(str) == selected_direction
        ]

    if subset.empty:
        st.warning("No records match these filters.")
        st.stop()

    total_count = subset["new_vehicles"].sum()
    average_count = subset["new_vehicles"].mean()

    hourly_totals = (
        subset.groupby("Hour")["new_vehicles"]
        .sum()
        .sort_index()
    )

    busiest_hour = int(hourly_totals.idxmax())

    c1, c2, c3 = st.columns(3)

    c1.metric("Sum of Recorded Counts", f"{total_count:,.0f}")
    c2.metric("Average per Record", f"{average_count:.1f}")
    c3.metric("Busiest Recorded Hour", f"{busiest_hour:02d}:00")

    left, right = st.columns(2)

    with left:
        st.markdown("#### Vehicle counts by recorded hour")
        st.bar_chart(hourly_totals, color="#20bfa9")

        observed_hours = sorted(
            subset["Hour"].dropna().astype(int).unique()
        )

        st.caption(
            "Recorded hours: "
            + ", ".join(f"{h:02d}:00" for h in observed_hours)
            + ". Missing hours indicate unavailable observations."
        )

    with right:
        st.markdown("#### Vehicle type breakdown")

        vehicle_columns = [
            col for col in VEHICLE_COLUMNS if col in subset.columns
        ]

        vehicle_totals = (
            subset[vehicle_columns]
            .sum()
            .rename(VEHICLE_LABELS)
        )

        st.bar_chart(vehicle_totals, color="#4f8cff")

    left, right = st.columns(2)

    with left:
        st.markdown("#### Count-based category distribution")

        category_counts = (
            subset["Congestion"].astype(str)
            .value_counts()
            .reindex(CATEGORIES, fill_value=0)
        )

        st.bar_chart(category_counts, color="#e5a94d")

        st.caption(
            "This chart counts records in each category. It does not measure "
            "the actual number of hours a road was congested."
        )

    with right:
        st.markdown("#### Daily recorded traffic")

        daily_totals = (
            subset.groupby(subset["Timestamp"].dt.date)["new_vehicles"]
            .sum()
            .sort_index()
        )

        st.line_chart(daily_totals, color="#a78bfa")

        st.caption(
            "Daily totals include only intervals available in the dataset "
            "and may not represent a complete 24-hour day."
        )

    st.markdown("#### Filtered data preview")

    st.dataframe(
        subset.sort_values("Timestamp", ascending=False).head(100),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download filtered data as CSV",
        data=subset.to_csv(index=False).encode("utf-8"),
        file_name="filtered_traffic_data.csv",
        mime="text/csv"
    )


# =====================================================
# 10. CONGESTION PREDICTION
# =====================================================
elif selected_page == "🤖 Congestion Prediction":

    st.markdown(
        '<div class="section-heading">Estimate a Traffic Category</div>',
        unsafe_allow_html=True
    )

    st.info(
        "This is a historical-data-based estimate. The application does not "
        "read live cameras, current road speeds, or real-time traffic feeds."
    )

    intersections = sorted(
        df["Intersection"].dropna().astype(str).unique().tolist()
    )

    directions = sorted(
        df["Direction"].dropna().astype(str).unique().tolist()
    )

    with st.form("prediction_form"):

        st.markdown("#### 1. Select location and time")

        col1, col2 = st.columns(2)

        with col1:
            selected_intersection = st.selectbox(
                "Intersection",
                intersections
            )

            hour = st.slider(
                "Hour of day",
                min_value=0,
                max_value=23,
                value=18
            )

        with col2:
            selected_direction = st.selectbox(
                "Direction",
                directions
            )

            day_name = st.selectbox(
                "Day of week",
                DAY_NAMES,
                index=2
            )

        day_of_week = DAY_NAMES.index(day_name)

        matching_history = df[
            (df["Intersection"].astype(str) == selected_intersection)
            & (df["Direction"].astype(str) == selected_direction)
            & (df["Hour"] == hour)
            & (df["DayOfWeek"] == day_of_week)
        ]

        if not matching_history.empty:
            automatic_count = float(
                matching_history["PreviousTraffic"].median()
            )
            estimate_note = (
                "Estimated from historical records matching the selected "
                "intersection, direction, hour, and weekday."
            )
        else:
            fallback_history = df[
                (df["Intersection"].astype(str) == selected_intersection)
                & (df["Direction"].astype(str) == selected_direction)
            ]

            if not fallback_history.empty:
                automatic_count = float(
                    fallback_history["PreviousTraffic"].median()
                )
                estimate_note = (
                    "No exact time/day match was found. Using the historical "
                    "median for this intersection and direction."
                )
            else:
                automatic_count = float(
                    df["PreviousTraffic"].median()
                )
                estimate_note = (
                    "No location-specific match was found. Using the "
                    "overall historical median."
                )

        st.markdown("#### 2. Previous five-minute vehicle count")

        use_automatic = st.checkbox(
            "Use estimated historical count automatically",
            value=True
        )

        if use_automatic:
            previous_traffic = automatic_count

            st.metric(
                "Estimated previous-interval count",
                f"{previous_traffic:.0f}"
            )

            st.caption(estimate_note + " This is not a live count.")

        else:
            previous_traffic = st.number_input(
                "Enter previous five-minute vehicle count",
                min_value=0.0,
                value=max(0.0, automatic_count),
                step=1.0
            )

        submitted = st.form_submit_button(
            "✨ Predict category",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame([{
            "Hour": int(hour),
            "DayOfWeek": int(day_of_week),
            "IsWeekend": int(day_of_week >= 5),
            "PreviousTraffic": float(previous_traffic),
            "Intersection": selected_intersection,
            "Direction": selected_direction
        }])

        try:
            prediction = model.predict(input_data)[0]

            probabilities = None
            model_classes = None

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0]

                if hasattr(model, "named_steps"):
                    classifier = model.named_steps.get("classifier")

                    if (
                        classifier is not None
                        and hasattr(classifier, "classes_")
                    ):
                        model_classes = list(classifier.classes_)

                if (
                    model_classes is None
                    and hasattr(model, "classes_")
                ):
                    model_classes = list(model.classes_)

            colors = {
                "Low": "#20bfa9",
                "Medium": "#e5a94d",
                "High": "#ff6b6b"
            }

            result_color = colors.get(prediction, "#66e3ce")

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

            if probabilities is not None and model_classes is not None:

                probability_data = pd.DataFrame({
                    "Category": model_classes,
                    "Estimated probability (%)": [
                        float(p) * 100 for p in probabilities
                    ]
                }).set_index("Category")

                probability_data = probability_data.reindex(
                    CATEGORIES,
                    fill_value=0
                )

                st.markdown("#### Probability assigned to each category")

                st.bar_chart(
                    probability_data,
                    horizontal=True,
                    color="#20bfa9"
                )

                predicted_probability = None

                if prediction in model_classes:
                    predicted_probability = float(
                        probabilities[model_classes.index(prediction)]
                    )

                if predicted_probability is not None:
                    st.write(
                        "Probability assigned to the predicted category: "
                        f"**{predicted_probability:.1%}**"
                    )

                    if predicted_probability < 0.60:
                        st.warning(
                            "The highest class probability is below 60%. "
                            "Treat this estimate cautiously."
                        )
                    else:
                        st.caption(
                            "A higher model probability is not a guarantee "
                            "of real-world accuracy. Probability calibration "
                            "has not been established here."
                        )

            with st.expander("View inputs sent to the model"):
                st.dataframe(input_data, use_container_width=True)

        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.caption(
                "The saved model must use the same feature names and "
                "preprocessing expected by this application."
            )


# =====================================================
# 11. ABOUT PROJECT
# =====================================================
elif selected_page == "ℹ️ About Project":

    st.markdown(
        '<div class="section-heading">About This Project</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Smart City Traffic Intelligence explores historical traffic counts "
        "from selected Pune intersections and uses a machine-learning "
        "classifier to estimate a count-based traffic category."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>📂 Dataset</h3>
            <p>
                Historical traffic-camera observations from January 2023.
                Processed records represent five-minute intervals for
                individual cameras and directions.
            </p>
            <p><strong>Source:</strong> Heterogeneous Traffic Count Dataset, Pune.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🧠 Machine Learning</h3>
            <p>
                A saved Decision Tree pipeline uses time, location,
                direction, and previous traffic-count features to estimate
                Low, Medium, or High categories.
            </p>
            <p><strong>Previously reported test accuracy:</strong>
            approximately 87%.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Project workflow")

    st.markdown("""
    1. Clean and process the traffic-camera data.
    2. Aggregate observations into five-minute intervals.
    3. Explore hourly patterns and vehicle-type counts.
    4. Train and evaluate a classification model.
    5. Present the analysis and predictions in this dashboard.
    """)

    st.warning(
        "The categories are derived from vehicle-count thresholds, not "
        "ground-truth congestion measurements such as road speed or travel "
        "time. The dataset is historical, not live. The approximately 87% "
        "accuracy is a previously reported test result and should be "
        "confirmed against the model's evaluation."
    )


# =====================================================
# 12. FOOTER
# =====================================================
st.divider()

st.caption(
    "Smart City Traffic Intelligence | Academic data science project | "
    "Historical Pune traffic data | Not a live traffic monitoring service"
)
