
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Smart City Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_FILE = Path("traffic_5min.csv")
MODEL_FILE = Path("traffic_congestion_model.joblib")

# --------------------------------------------------
# THEME AND CSS
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg: #0b1120;
    --panel: #121c30;
    --border: #26354c;
    --text: #e8eef8;
    --muted: #94a5bd;
    --cyan: #48d7e8;
    --blue: #7297ff;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, #142640 0%, transparent 40%),
        #0b1120;
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #f1f6ff !important;
}

p, label, .stMarkdown {
    color: #d5deed;
}

[data-testid="stSidebar"] {
    background: #0e1728;
    border-right: 1px solid #25334a;
}

[data-testid="stSidebar"] * {
    color: #e8eef8;
}

/* Clean sidebar navigation: hide radio dots and style options as menu rows */
[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 5px;
}
[data-testid="stSidebar"] [role="radiogroup"] > label {
    padding: 9px 11px;
    border: 1px solid transparent;
    border-radius: 9px;
    background: transparent;
    transition: background .15s ease, border-color .15s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover {
    background: #17263b;
    border-color: #2b405a;
}
[data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child {
    display: none !important;
}
[data-testid="stSidebar"] [role="radiogroup"] > label > div:last-child {
    margin-left: 0 !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(90deg, #17354a, #14283d);
    border-color: #31566b;
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {
    color: #55dce8 !important;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #17243a, #111b2d);
    border: 1px solid #2b3c56;
    border-radius: 15px;
    padding: 18px 18px;
}

[data-testid="stMetricLabel"] {
    color: #9fb0c8 !important;
}

[data-testid="stMetricValue"] {
    color: #f2f7ff !important;
    font-family: 'Space Grotesk', sans-serif;
}

[data-testid="stMetricDelta"] {
    color: #48d7e8 !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #26354c !important;
    border-radius: 15px;
}

.stButton > button {
    border-radius: 10px;
    border: 1px solid #344964;
    background: #17263c;
    color: #eef6ff;
    min-height: 42px;
    font-weight: 600;
}

.stButton > button:hover {
    border-color: #48d7e8;
    color: #48d7e8;
}

.stDownloadButton > button {
    background: #19384a;
    color: white;
    border-radius: 10px;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: #111c2e;
    border-color: #30415b;
    border-radius: 9px;
}

hr {
    border-color: #26354c;
}

.hero {
    padding: 27px 30px;
    border-radius: 19px;
    background: linear-gradient(115deg, #172c46, #122036 65%, #173544);
    border: 1px solid #30465f;
    margin-bottom: 24px;
}

.eyebrow {
    text-transform: uppercase;
    color: #48d7e8;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
}

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(27px, 3.2vw, 39px);
    font-weight: 700;
    color: #f4f8ff;
    line-height: 1.2;
    margin: 10px 0;
}

.hero-subtitle {
    color: #b2c3d9;
    font-size: 14px;
    max-width: 720px;
    line-height: 1.7;
}

.section-label {
    color: #48d7e8;
    font-size: 11px;
    letter-spacing: 1.8px;
    font-weight: 700;
    text-transform: uppercase;
    margin-bottom: 5px;
}

.panel-title {
    color: #edf4ff;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 19px;
    font-weight: 600;
    margin-bottom: 4px;
}

.panel-subtitle {
    color: #8fa2bc;
    font-size: 12px;
    margin-bottom: 15px;
}

.insight-card {
    background: linear-gradient(145deg, #15243a, #111b2d);
    border: 1px solid #2a3c55;
    padding: 18px;
    border-radius: 14px;
    min-height: 130px;
}

.insight-title {
    color: #94a9c5;
    font-size: 12px;
    margin-bottom: 8px;
}

.insight-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 23px;
    font-weight: 700;
    color: #f2f7ff;
}

.insight-note {
    font-size: 12px;
    color: #91a7c1;
    margin-top: 8px;
}

.status-pill {
    display: inline-block;
    border: 1px solid #31556b;
    color: #64dce7;
    background: #132d3c;
    border-radius: 30px;
    padding: 5px 11px;
    font-size: 11px;
    font-weight: 600;
}

.small-note {
    color: #8fa2bc;
    font-size: 12px;
}

[data-testid="stDataFrame"] {
    border: 1px solid #2a3c55;
    border-radius: 12px;
}

footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------
@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        return None

    df = pd.read_csv(DATA_FILE)

    # Standardize column names
    df.columns = [str(c).strip() for c in df.columns]

    # Build timestamp from Date and Time when needed
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"], errors="coerce"
        )
    elif "Date" in df.columns and "Time" in df.columns:
        df["Timestamp"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            errors="coerce"
        )

    if "Timestamp" in df.columns:
        df["Hour"] = df["Timestamp"].dt.hour
        df["DateOnly"] = df["Timestamp"].dt.date
        df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
        df["DayName"] = df["Timestamp"].dt.day_name()
        df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

    # Ensure expected vehicle columns exist
    vehicle_cols = ["car_change", "motorbike_change",
                    "bus_change", "truck_change"]

    for col in vehicle_cols:
        if col not in df.columns:
            df[col] = 0

        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Use the provided traffic count if available
    if "new_vehicles" not in df.columns:
        df["new_vehicles"] = df[vehicle_cols].sum(axis=1)

    df["new_vehicles"] = pd.to_numeric(
        df["new_vehicles"], errors="coerce"
    ).fillna(0)

    # Create a category only if it is absent
    if "Congestion" not in df.columns:
        df["Congestion"] = pd.cut(
            df["new_vehicles"],
            bins=[-np.inf, 9, 52, np.inf],
            labels=["Low", "Medium", "High"]
        ).astype(str)

    df["Congestion"] = (
        df["Congestion"].astype(str).str.strip().str.title()
    )

    df = df.replace([np.inf, -np.inf], np.nan)

    return df


@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        return None

    try:
        saved = joblib.load(MODEL_FILE)

        if isinstance(saved, dict):
            return saved.get("model", saved.get("pipeline"))

        return saved
    except Exception:
        return None


df = load_data()
model = load_model()

if df is None:
    st.error(
        "The file traffic_5min.csv was not found. "
        "Make sure it is in the same GitHub folder as app.py."
    )
    st.stop()

if df.empty:
    st.error("The traffic dataset is empty.")
    st.stop()

if "Intersection" not in df.columns:
    df["Intersection"] = "Unknown"

if "Direction" not in df.columns:
    df["Direction"] = "Unknown"

df["Intersection"] = df["Intersection"].fillna("Unknown").astype(str)
df["Direction"] = df["Direction"].fillna("Unknown").astype(str)

# --------------------------------------------------
# CHART CONFIGURATION
# --------------------------------------------------
CHART_COLORS = {
    "Low": "#48d7a8",
    "Medium": "#f7bd59",
    "High": "#ff7185",
    "car_change": "#48d7e8",
    "motorbike_change": "#7297ff",
    "bus_change": "#b99aff",
    "truck_change": "#f5a76b"
}

def style_chart(fig, height=330):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c6d3e5", family="DM Sans"),
        margin=dict(l=12, r=12, t=35, b=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        hoverlabel=dict(
            bgcolor="#17263c",
            bordercolor="#48d7e8",
            font_color="#ffffff"
        ),
        xaxis=dict(
            gridcolor="#25344a",
            zerolinecolor="#25344a"
        ),
        yaxis=dict(
            gridcolor="#25344a",
            zerolinecolor="#25344a"
        )
    )
    return fig


def panel_heading(title, subtitle):
    st.markdown(
        f'<div class="panel-title">{title}</div>'
        f'<div class="panel-subtitle">{subtitle}</div>',
        unsafe_allow_html=True
    )


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 🚦 Traffic Intelligence")
    st.markdown(
        '<span class="status-pill">PUNE · HISTORICAL DATA</span>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Traffic Explorer",
            "Congestion Prediction",
            "About Project"
        ],
        key="navigation"
    )

    st.markdown("---")
    st.markdown("### Dataset snapshot")

    st.metric("Records", f"{len(df):,}")

    if "Timestamp" in df.columns and df["Timestamp"].notna().any():
        start_date = df["Timestamp"].min().strftime("%d %b %Y")
        end_date = df["Timestamp"].max().strftime("%d %b %Y")
        st.caption(f"Available dates: {start_date} – {end_date}")

    st.caption(
        "This app analyses historical traffic-count data. "
        "It does not receive live traffic feeds."
    )


# --------------------------------------------------
# PAGE 1: OVERVIEW
# --------------------------------------------------
if page == "Overview":

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">SMART MOBILITY · ANALYTICS PLATFORM</div>
        <div class="hero-title">Understand traffic.<br>Make smarter decisions.</div>
        <div class="hero-subtitle">
            Explore historical vehicle-count patterns across selected Pune
            intersections, identify busy periods, and estimate congestion
            categories from the available data.
        </div>
        <br>
        <span class="status-pill">● HISTORICAL DATA MODE</span>
    </div>
    """, unsafe_allow_html=True)

    # Global filters
    st.markdown('<div class="section-label">CONTROL CENTRE</div>',
                unsafe_allow_html=True)
    st.markdown("### Your traffic overview")

    f1, f2 = st.columns([1, 1])

    with f1:
        intersections = sorted(df["Intersection"].unique().tolist())
        selected_intersection = st.selectbox(
            "Select intersection",
            ["All intersections"] + intersections,
            key="overview_intersection"
        )

    with f2:
        directions = sorted(df["Direction"].unique().tolist())
        selected_direction = st.selectbox(
            "Select direction",
            ["All directions"] + directions,
            key="overview_direction"
        )

    view = df.copy()

    if selected_intersection != "All intersections":
        view = view[view["Intersection"] == selected_intersection]

    if selected_direction != "All directions":
        view = view[view["Direction"] == selected_direction]

    if view.empty:
        st.warning("No records match these filters.")
        st.stop()

    # KPI cards
    total_records = len(view)
    recorded_count = view["new_vehicles"].sum()
    avg_count = view["new_vehicles"].mean()
    high_pct = (view["Congestion"] == "High").mean() * 100

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Traffic observations", f"{total_records:,}")
    k2.metric("Recorded vehicle-count total", f"{recorded_count:,.0f}")
    k3.metric("Average count / record", f"{avg_count:.1f}")
    k4.metric("High-category records", f"{high_pct:.1f}%")

    st.caption(
        "Totals are sums of recorded observations, not unique vehicles "
        "across the entire city. Multiple cameras and directions may overlap."
    )

    st.markdown("---")

    left, right = st.columns([1.55, 1])

    with left:
        panel_heading(
            "Traffic through the day",
            "Average recorded vehicle count by hour"
        )

        if "Hour" in view.columns:
            hourly = (
                view.groupby("Hour", as_index=False)["new_vehicles"]
                .mean()
                .sort_values("Hour")
            )

            fig = px.area(
                hourly,
                x="Hour",
                y="new_vehicles",
                markers=True,
                labels={
                    "Hour": "Hour of day",
                    "new_vehicles": "Average count"
                }
            )
            fig.update_traces(
                line=dict(color="#48d7e8", width=3),
                fillcolor="rgba(72,215,232,0.12)",
                marker=dict(size=7)
            )
            fig.update_xaxes(dtick=1)
            st.plotly_chart(
                style_chart(fig, 360),
                use_container_width=True
            )
        else:
            st.info("Timestamp information is unavailable.")

    with right:
        panel_heading(
            "Congestion mix",
            "Distribution of recorded categories"
        )

        category_counts = (
            view["Congestion"].value_counts()
            .reindex(["Low", "Medium", "High"], fill_value=0)
            .reset_index()
        )
        category_counts.columns = ["Category", "Records"]

        fig = px.pie(
            category_counts,
            names="Category",
            values="Records",
            hole=0.68,
            color="Category",
            color_discrete_map=CHART_COLORS
        )
        fig.update_traces(
            textposition="outside",
            textinfo="percent",
            marker=dict(line=dict(color="#0b1120", width=3))
        )
        fig.update_layout(showlegend=True)
        st.plotly_chart(
            style_chart(fig, 360),
            use_container_width=True
        )

    left, right = st.columns(2)

    with left:
        panel_heading(
            "Vehicle composition",
            "Sum of recorded counts by vehicle type"
        )

        vehicle_columns = [
            "car_change", "motorbike_change",
            "bus_change", "truck_change"
        ]

        vehicle_values = (
            view[vehicle_columns].sum()
            .rename_axis("Vehicle type")
            .reset_index(name="Recorded count")
        )
        vehicle_values["Vehicle type"] = vehicle_values[
            "Vehicle type"
        ].map({
            "car_change": "Cars",
            "motorbike_change": "Motorbikes",
            "bus_change": "Buses",
            "truck_change": "Trucks"
        })

        fig = px.bar(
            vehicle_values,
            x="Vehicle type",
            y="Recorded count",
            color="Vehicle type",
            color_discrete_map={
                "Cars": "#48d7e8",
                "Motorbikes": "#7297ff",
                "Buses": "#b99aff",
                "Trucks": "#f5a76b"
            }
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(
            style_chart(fig, 320),
            use_container_width=True
        )

    with right:
        panel_heading(
            "Busiest intersections",
            "Ranked by average recorded count per observation"
        )

        location_data = (
            view.groupby("Intersection", as_index=False)
            .agg(
                Average_Count=("new_vehicles", "mean"),
                Observations=("new_vehicles", "size")
            )
            .sort_values("Average_Count", ascending=True)
        )

        fig = px.bar(
            location_data,
            x="Average_Count",
            y="Intersection",
            orientation="h",
            color="Average_Count",
            color_continuous_scale=["#23465e", "#48d7e8"],
            hover_data=["Observations"]
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(
            style_chart(fig, 320),
            use_container_width=True
        )

    st.markdown("---")
    panel_heading("Recent observations", "Latest records in the selected view")

    if "Timestamp" in view.columns:
        recent = view.sort_values("Timestamp", ascending=False).head(10)
    else:
        recent = view.tail(10)

    display_cols = [
        c for c in [
            "Timestamp", "Intersection", "Direction",
            "new_vehicles", "Congestion"
        ] if c in recent.columns
    ]

    st.dataframe(
        recent[display_cols],
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# PAGE 2: TRAFFIC EXPLORER
# --------------------------------------------------
elif page == "Traffic Explorer":

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">INTERACTIVE DATA EXPLORATION</div>
        <div class="hero-title">Traffic Explorer</div>
        <div class="hero-subtitle">
            Filter the historical dataset and compare traffic patterns
            by location, direction, date, time, and vehicle type.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Filter your data")

    f1, f2, f3 = st.columns(3)

    with f1:
        location_options = sorted(df["Intersection"].unique().tolist())
        locations = st.multiselect(
            "Intersections",
            location_options,
            default=location_options
        )

    with f2:
        direction_options = sorted(df["Direction"].unique().tolist())
        selected_directions = st.multiselect(
            "Directions",
            direction_options,
            default=direction_options
        )

    with f3:
        category_options = ["Low", "Medium", "High"]
        selected_categories = st.multiselect(
            "Congestion categories",
            category_options,
            default=category_options
        )

    filtered = df[
        df["Intersection"].isin(locations)
        & df["Direction"].isin(selected_directions)
        & df["Congestion"].isin(selected_categories)
    ].copy()

    if "Timestamp" in filtered.columns and filtered["Timestamp"].notna().any():
        valid_dates = filtered["Timestamp"].dropna()

        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()

        date_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
            start, end = date_range
            filtered = filtered[
                (filtered["Timestamp"].dt.date >= start)
                & (filtered["Timestamp"].dt.date <= end)
            ]

    if filtered.empty:
        st.warning("No observations match the selected filters.")
        st.stop()

    st.markdown("---")

    a, b, c = st.columns(3)
    a.metric("Filtered observations", f"{len(filtered):,}")
    b.metric("Average count", f"{filtered['new_vehicles'].mean():.2f}")
    c.metric("Locations selected", f"{filtered['Intersection'].nunique()}")

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        panel_heading(
            "Hourly traffic pattern",
            "Compare average counts by hour"
        )

        hourly = (
            filtered.groupby("Hour", as_index=False)["new_vehicles"]
            .mean()
            .sort_values("Hour")
        )

        fig = px.line(
            hourly,
            x="Hour",
            y="new_vehicles",
            markers=True,
            labels={
                "Hour": "Hour of day",
                "new_vehicles": "Average count"
            }
        )
        fig.update_traces(line=dict(color="#48d7e8", width=3))
        fig.update_xaxes(dtick=1)
        st.plotly_chart(
            style_chart(fig, 340),
            use_container_width=True
        )

    with right:
        panel_heading(
            "Traffic by location",
            "Average count for each selected intersection"
        )

        location_hour = (
            filtered.groupby("Intersection", as_index=False)
            ["new_vehicles"].mean()
            .sort_values("new_vehicles", ascending=False)
        )

        fig = px.bar(
            location_hour,
            x="Intersection",
            y="new_vehicles",
            color="Intersection",
            labels={"new_vehicles": "Average count"}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(
            style_chart(fig, 340),
            use_container_width=True
        )

    left, right = st.columns(2)

    with left:
        panel_heading(
            "Vehicle type comparison",
            "Recorded counts within the selected filters"
        )

        vehicle_cols = [
            "car_change", "motorbike_change",
            "bus_change", "truck_change"
        ]

        vehicle_totals = (
            filtered[vehicle_cols].sum()
            .rename_axis("Vehicle")
            .reset_index(name="Count")
        )

        vehicle_totals["Vehicle"] = vehicle_totals["Vehicle"].map({
            "car_change": "Cars",
            "motorbike_change": "Motorbikes",
            "bus_change": "Buses",
            "truck_change": "Trucks"
        })

        fig = px.bar(
            vehicle_totals,
            x="Vehicle",
            y="Count",
            color="Vehicle"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(
            style_chart(fig, 320),
            use_container_width=True
        )

    with right:
        panel_heading(
            "Hourly category distribution",
            "Number of observations in each category by hour"
        )

        heat_data = (
            filtered.groupby(["Hour", "Congestion"])
            .size()
            .reset_index(name="Records")
        )

        if not heat_data.empty:
            heat_pivot = heat_data.pivot(
                index="Congestion",
                columns="Hour",
                values="Records"
            ).fillna(0)

            heat_pivot = heat_pivot.reindex(
                ["Low", "Medium", "High"]
            ).fillna(0)

            fig = px.imshow(
                heat_pivot,
                aspect="auto",
                color_continuous_scale=["#17243a", "#23657b", "#48d7e8"],
                labels=dict(
                    x="Hour of day",
                    y="Category",
                    color="Records"
                )
            )
            st.plotly_chart(
                style_chart(fig, 320),
                use_container_width=True
            )

    st.markdown("---")
    panel_heading("Filtered records", "Inspect and export the data")

    st.dataframe(
        filtered.head(1000),
        use_container_width=True,
        hide_index=True
    )

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download filtered CSV",
        data=csv,
        file_name="filtered_traffic_data.csv",
        mime="text/csv"
    )

    st.caption("The preview shows at most 1,000 rows; the CSV export contains all filtered rows.")


# --------------------------------------------------
# PAGE 3: CONGESTION PREDICTION
# --------------------------------------------------
elif page == "Congestion Prediction":

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">MACHINE LEARNING WORKSPACE</div>
        <div class="hero-title">Congestion Prediction</div>
        <div class="hero-subtitle">
            Estimate a congestion category for a selected intersection,
            direction, and time using the trained model when available.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.warning(
            "The trained model could not be loaded. Check that "
            "traffic_congestion_model.joblib exists in your repository."
        )

    st.markdown("### Configure a prediction")

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)

        with c1:
            prediction_location = st.selectbox(
                "Intersection",
                sorted(df["Intersection"].unique().tolist())
            )

            prediction_direction = st.selectbox(
                "Direction",
                sorted(df["Direction"].unique().tolist())
            )

            prediction_hour = st.slider(
                "Hour of day",
                min_value=0,
                max_value=23,
                value=18,
                format="%d:00"
            )

        with c2:
            day_name_to_num = {
                "Monday": 0, "Tuesday": 1, "Wednesday": 2,
                "Thursday": 3, "Friday": 4, "Saturday": 5,
                "Sunday": 6
            }

            prediction_day_name = st.selectbox(
                "Day of week",
                list(day_name_to_num.keys()),
                index=4
            )

            use_custom_count = st.checkbox(
                "Enter a custom previous traffic count"
            )

            historical = df[
                (df["Intersection"] == prediction_location)
                & (df["Direction"] == prediction_direction)
                & (df["Hour"] == prediction_hour)
                & (df["DayOfWeek"] == day_name_to_num[prediction_day_name])
            ]

            if historical.empty:
                historical = df[
                    (df["Intersection"] == prediction_location)
                    & (df["Direction"] == prediction_direction)
                ]

            if historical.empty:
                default_previous = float(df["new_vehicles"].median())
            else:
                default_previous = float(historical["new_vehicles"].median())

            if use_custom_count:
                previous_traffic = st.number_input(
                    "Previous traffic count",
                    min_value=0.0,
                    value=round(default_previous, 1),
                    step=1.0
                )
            else:
                previous_traffic = st.number_input(
                    "Historical typical count (editable)",
                    min_value=0.0,
                    value=round(default_previous, 1),
                    step=1.0
                )

        submitted = st.form_submit_button(
            "⚡ Estimate congestion",
            use_container_width=True
        )

    if submitted:
        day_num = day_name_to_num[prediction_day_name]

        input_df = pd.DataFrame([{
            "Intersection": prediction_location,
            "Direction": prediction_direction,
            "Hour": int(prediction_hour),
            "DayOfWeek": int(day_num),
            "IsWeekend": int(day_num >= 5),
            "PreviousTraffic": float(previous_traffic)
        }])

        prediction = None
        probabilities = None
        prediction_error = None

        if model is not None:
            try:
                prediction = str(model.predict(input_df)[0]).title()

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(input_df)[0]
                    classes = getattr(model, "classes_", None)

                    if classes is None and hasattr(model, "named_steps"):
                        final_step = list(model.named_steps.values())[-1]
                        classes = getattr(final_step, "classes_", None)

                    if classes is not None:
                        probabilities = {
                            str(label).title(): float(prob)
                            for label, prob in zip(classes, probs)
                        }
            except Exception as exc:
                prediction_error = str(exc)

        if prediction is not None:
            color = {
                "Low": "#48d7a8",
                "Medium": "#f7bd59",
                "High": "#ff7185"
            }.get(prediction, "#48d7e8")

            st.markdown("---")
            st.markdown("### Estimated result")

            st.markdown(
                f"""
                <div style="
                    padding:26px;
                    border:1px solid {color};
                    border-radius:17px;
                    background:linear-gradient(120deg,#14243a,#111b2d);
                    margin-bottom:15px;">
                    <div style="color:#9fb1c8;font-size:12px;
                                letter-spacing:1.5px;text-transform:uppercase;">
                        MODEL PREDICTION
                    </div>
                    <div style="color:{color};font-size:36px;
                                font-weight:700;margin-top:8px;">
                        {prediction}
                    </div>
                    <div style="color:#9fb1c8;font-size:13px;margin-top:8px;">
                        {prediction_location} · {prediction_direction} ·
                        {prediction_hour:02d}:00 · {prediction_day_name}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if probabilities:
                panel_heading(
                    "Model confidence scores",
                    "Predicted class probabilities, not guaranteed accuracy"
                )

                prob_df = pd.DataFrame({
                    "Category": list(probabilities.keys()),
                    "Probability": list(probabilities.values())
                })

                fig = px.bar(
                    prob_df,
                    x="Category",
                    y="Probability",
                    color="Category",
                    color_discrete_map=CHART_COLORS,
                    text=prob_df["Probability"].map(
                        lambda x: f"{x:.1%}"
                    )
                )
                fig.update_yaxes(
                    tickformat=".0%",
                    range=[0, 1]
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(
                    style_chart(fig, 300),
                    use_container_width=True
                )

            st.info(
                "This is a model-based estimate from historical data, "
                "not a live traffic forecast. The congestion labels in this "
                "project are derived from vehicle-count thresholds, so they "
                "should not be interpreted as measured road speed or delay."
            )

        else:
            st.error(
                "Prediction failed. The saved model may expect different "
                "feature names or input columns."
            )
            if prediction_error:
                with st.expander("Technical details"):
                    st.code(prediction_error)

            st.caption(
                "Check the model's training feature names and ensure they "
                "match the input fields used by this page."
            )


# --------------------------------------------------
# PAGE 4: ABOUT PROJECT
# --------------------------------------------------
elif page == "About Project":

    st.markdown("""
    <div class="hero">
        <div class="eyebrow">PROJECT OVERVIEW</div>
        <div class="hero-title">Inside the intelligence layer</div>
        <div class="hero-subtitle">
            A student data-science project exploring historical traffic
            patterns and machine-learning-based congestion categories.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### What this platform does")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="insight-card">
            <div class="insight-title">01 · DATA PROCESSING</div>
            <div class="insight-value">Clean & organise</div>
            <div class="insight-note">
                Reads the processed traffic CSV, handles missing numeric
                values, and extracts time-based information where possible.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-card">
            <div class="insight-title">02 · EXPLORATORY ANALYSIS</div>
            <div class="insight-value">Find patterns</div>
            <div class="insight-note">
                Compares hourly traffic, intersections, vehicle types,
                and the distribution of congestion categories.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="insight-card">
            <div class="insight-title">03 · MACHINE LEARNING</div>
            <div class="insight-value">Estimate categories</div>
            <div class="insight-note">
                Uses the saved classification model when available to
                estimate Low, Medium, or High congestion.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-card">
            <div class="insight-title">04 · INTERACTIVE DASHBOARD</div>
            <div class="insight-value">Explore & export</div>
            <div class="insight-note">
                Supports filtering, interactive chart exploration,
                record previews, and CSV downloads.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Dataset details")

    st.write(
        "The project uses a processed version of the Heterogeneous Traffic "
        "Count Dataset, Pune. The source dataset contains traffic observations "
        "from selected intersections and camera directions."
    )

    st.markdown(
        "[Open the original dataset on Mendeley Data](https://data.mendeley.com/datasets/xnf2k6n288/1)"
    )

    st.markdown("### Important limitations")

    st.warning("""
    - This dashboard is not connected to a live traffic feed.
    - Weather and rainfall are not displayed because the current app does
      not integrate verified weather observations.
    - Vehicle-count totals across cameras or directions may overlap.
    - Congestion categories are based on vehicle-count thresholds and do
      not directly measure speed, travel time, or road capacity.
    - Model predictions depend on the training data and feature definitions.
    """)

    st.markdown("---")
    st.markdown(
        '<div class="small-note">'
        'SMART CITY TRAFFIC INTELLIGENCE · DATA SCIENCE PROJECT'
        '</div>',
        unsafe_allow_html=True
    )
