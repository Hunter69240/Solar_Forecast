import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import plotly.graph_objects as go
from utils import fetch_weather_forecast

st.set_page_config(page_title="Weather Data Viewer", page_icon="🌤️", layout="wide")

# ── Back button ────────────────────────────────────────────
if st.button("← Back to Forecast", type="secondary"):
    st.switch_page("app.py")

st.title("🌤️ Fetched Weather Data Viewer")
st.caption("📍 Bengaluru, India — Live 7-day hourly forecast from Open-Meteo")

# ── Fetch data ─────────────────────────────────────────────
weather_df = fetch_weather_forecast()

if weather_df is None:
    st.error("❌ Could not fetch weather data. Please check your internet connection.")
    st.stop()

# ── Friendly column names ──────────────────────────────────
COLUMN_MAP = {
    "shortwave_radiation": "Shortwave Radiation (W/m²)",
    "direct_radiation":    "Direct Radiation (W/m²)",
    "temperature":         "Temperature (°C)",
    "cloudcover":          "Cloud Cover (%)",
    "windspeed":           "Wind Speed (km/h)"
}

display_df = weather_df.rename(columns=COLUMN_MAP).reset_index()
display_df = display_df.rename(columns={"datetime": "Datetime"})

# ── Unique dates for filter ────────────────────────────────
available_dates = sorted(display_df["Datetime"].dt.date.unique())
date_labels = [d.strftime("%a, %d %b %Y") for d in available_dates]

# ── Sidebar filter ─────────────────────────────────────────
st.sidebar.header("🔍 Filter")
selected_labels = st.sidebar.multiselect(
    "Select days to view",
    options=date_labels,
    default=date_labels,
    help="Filter the table and chart by specific days"
)

selected_dates = [
    available_dates[date_labels.index(lbl)]
    for lbl in selected_labels
]

if not selected_dates:
    st.warning("No days selected. Please pick at least one day from the sidebar.")
    st.stop()

filtered_df = display_df[display_df["Datetime"].dt.date.isin(selected_dates)].copy()

# ── Summary metric cards ───────────────────────────────────
st.subheader("📊 Summary — Selected Period")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("☀️ Avg Shortwave", f"{filtered_df['Shortwave Radiation (W/m²)'].mean():.1f} W/m²")
col2.metric("🔆 Avg Direct Rad", f"{filtered_df['Direct Radiation (W/m²)'].mean():.1f} W/m²")
col3.metric("🌡️ Avg Temp", f"{filtered_df['Temperature (°C)'].mean():.1f} °C")
col4.metric("☁️ Avg Cloud Cover", f"{filtered_df['Cloud Cover (%)'].mean():.1f} %")
col5.metric("💨 Avg Wind Speed", f"{filtered_df['Wind Speed (km/h)'].mean():.1f} km/h")

st.divider()

# ── Chart ──────────────────────────────────────────────────
st.subheader("📈 Radiation & Weather Overview")

chart_tab1, chart_tab2, chart_tab3 = st.tabs(["☀️ Radiation", "🌡️ Temperature", "☁️ Cloud & Wind"])

with chart_tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered_df["Datetime"],
        y=filtered_df["Shortwave Radiation (W/m²)"],
        mode="lines",
        name="Shortwave Radiation",
        line=dict(color="#ffd54f", width=2),
        fill="tozeroy",
        fillcolor="rgba(255, 213, 79, 0.15)"
    ))
    fig.add_trace(go.Scatter(
        x=filtered_df["Datetime"],
        y=filtered_df["Direct Radiation (W/m²)"],
        mode="lines",
        name="Direct Radiation",
        line=dict(color="#ff8a65", width=2),
        fill="tozeroy",
        fillcolor="rgba(255, 138, 101, 0.10)"
    ))
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Datetime",
        yaxis_title="Radiation (W/m²)",
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", y=1.08)
    )
    st.plotly_chart(fig, use_container_width=True)

with chart_tab2:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=filtered_df["Datetime"],
        y=filtered_df["Temperature (°C)"],
        mode="lines",
        name="Temperature",
        line=dict(color="#4fc3f7", width=2),
        fill="tozeroy",
        fillcolor="rgba(79, 195, 247, 0.12)"
    ))
    fig2.update_layout(
        template="plotly_dark",
        xaxis_title="Datetime",
        yaxis_title="Temperature (°C)",
        height=380,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig2, use_container_width=True)

with chart_tab3:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=filtered_df["Datetime"],
        y=filtered_df["Cloud Cover (%)"],
        mode="lines",
        name="Cloud Cover",
        line=dict(color="#b0bec5", width=2),
        fill="tozeroy",
        fillcolor="rgba(176, 190, 197, 0.12)"
    ))
    fig3.add_trace(go.Scatter(
        x=filtered_df["Datetime"],
        y=filtered_df["Wind Speed (km/h)"],
        mode="lines",
        name="Wind Speed",
        line=dict(color="#81c784", width=2),
        yaxis="y2"
    ))
    fig3.update_layout(
        template="plotly_dark",
        xaxis_title="Datetime",
        yaxis=dict(title="Cloud Cover (%)", range=[0, 105]),
        yaxis2=dict(title="Wind Speed (km/h)", overlaying="y", side="right"),
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", y=1.08)
    )
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── Full data table ────────────────────────────────────────
st.subheader(f"📋 Hourly Data Table — {len(filtered_df)} rows")

# Style the table: highlight rows where radiation = 0 (night-time)
st.dataframe(
    filtered_df.set_index("Datetime"),
    use_container_width=True,
    height=450
)

# ── Download ───────────────────────────────────────────────
csv_data = filtered_df.to_csv(index=False).encode("utf-8")
date_range = (
    f"{selected_dates[0].strftime('%Y-%m-%d')}"
    if len(selected_dates) == 1
    else f"{selected_dates[0].strftime('%Y-%m-%d')}_to_{selected_dates[-1].strftime('%Y-%m-%d')}"
)
st.download_button(
    label="⬇️ Download as CSV",
    data=csv_data,
    file_name=f"weather_data_{date_range}.csv",
    mime="text/csv"
)

st.caption("Data source: [Open-Meteo](https://open-meteo.com) · Refreshed every hour · Timezone: Asia/Kolkata")
