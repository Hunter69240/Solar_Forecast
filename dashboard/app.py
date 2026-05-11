import sys
import os
import json
sys.path.append(os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from utils import fetch_weather_forecast, fetch_sunrise_sunset

# ── Path to pre-computed forecast file (written by GitHub Actions hourly) ──
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORECAST_PATH = os.path.join(BASE_DIR, "data", "forecasts.json")

st.set_page_config(page_title="Solar Forecast", page_icon="☀️")
st.title("☀️ Solar Power Generation Forecast")
st.caption("📍 Location: Bengaluru, India — Plant 1, Inverter 1BY6WEcLGh8j5v7")

# ── Session state defaults ─────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None
if "last_num_days" not in st.session_state:
    st.session_state.last_num_days = 3

# ── Fetch live weather data (used for chart overlays & weather table) ──
weather_forecast = fetch_weather_forecast()
sunrise_sunset   = fetch_sunrise_sunset()

if weather_forecast is None:
    st.warning("⚠️ Could not fetch live weather data. Sunrise/sunset markers and weather table will be unavailable.")


# ── Load pre-computed forecast from forecasts.json ────────
def load_forecast(forecast_dates):
    """
    Reads data/forecasts.json written by scripts/predict_and_save.py.
    Filters to the requested forecast_dates and returns a results df
    plus the generated_at timestamp.
    """
    if not os.path.exists(FORECAST_PATH):
        return None, None

    with open(FORECAST_PATH, "r") as f:
        data = json.load(f)

    generated_at = datetime.fromisoformat(data["generated_at"])

    df_all = pd.DataFrame(data["predictions"])
    df_all["datetime_obj"] = pd.to_datetime(df_all["datetime"])

    # Keep only the rows that fall on the requested forecast dates
    df_filtered = df_all[
        df_all["datetime_obj"].dt.date.isin(forecast_dates)
    ].copy().reset_index(drop=True)

    if df_filtered.empty:
        return None, generated_at

    df_results = pd.DataFrame({
        "Datetime":               df_filtered["datetime_obj"].tolist(),
        "Date":                   [h.strftime("%d %b") for h in df_filtered["datetime_obj"]],
        "Hour":                   [h.hour               for h in df_filtered["datetime_obj"]],
        "Time":                   [h.strftime("%H:%M")  for h in df_filtered["datetime_obj"]],
        "Predicted AC Power (W)": df_filtered["ac_power"].tolist()
    })

    return df_results, generated_at


# ── Slider ─────────────────────────────────────────────────
today = datetime.now().date()

num_days = st.slider(
    "Number of days to forecast",
    min_value=1,
    max_value=7,
    value=st.session_state.last_num_days,
    step=1
)

# Reset results if slider value changed
if num_days != st.session_state.last_num_days:
    st.session_state.last_num_days = num_days
    st.session_state.results = None

forecast_dates = [today + timedelta(days=d) for d in range(num_days)]

# ── Button ─────────────────────────────────────────────────
if st.button("☀️ Make Prediction", type="primary", use_container_width=True):
    df_results, generated_at = load_forecast(forecast_dates)

    if df_results is None and generated_at is None:
        # forecasts.json does not exist yet (cold start before first Actions run)
        st.error(
            "❌ No forecast data found. "
            "Run `python scripts/predict_and_save.py` locally or wait for the first GitHub Actions run."
        )
    elif df_results is None:
        # File exists but has no rows for the requested dates (script ran on a different day)
        st.error(
            "❌ Forecast file exists but contains no data for the selected dates. "
            "This can happen if the forecast file is more than 24 hours old. "
            "Trigger a manual GitHub Actions run to refresh it."
        )
    else:
        st.session_state.results = {
            "df":             df_results,
            "forecast_dates": forecast_dates,
            "num_days":       num_days,
            "generated_at":   generated_at
        }

# ── Display results if available ──────────────────────────
if st.session_state.results is not None:
    r              = st.session_state.results
    df_results     = r["df"]
    forecast_dates = r["forecast_dates"]
    result_num_days = r["num_days"]
    generated_at   = r["generated_at"]

    # ── Freshness indicator ────────────────────────────────
    age_minutes = (datetime.now() - generated_at).total_seconds() / 60
    st.caption(
        f"🕐 Forecast last updated: **{generated_at.strftime('%d %b %Y, %H:%M')}** "
        f"({int(age_minutes)} min ago)"
    )
    if age_minutes > 90:
        st.warning(
            "⚠️ Forecast data is more than 90 minutes old. "
            "GitHub Actions may have missed a scheduled run."
        )

    # ── Daily total metric cards ───────────────────────────
    daily_totals = (
        df_results.groupby("Date")["Predicted AC Power (W)"]
        .sum()
        .reindex([d.strftime("%d %b") for d in forecast_dates])
    )

    cols = st.columns(result_num_days)
    for i, (date_label, total) in enumerate(daily_totals.items()):
        cols[i].metric(label=date_label, value=f"{total:,.0f} W")

    date_range_label = (
        forecast_dates[0].strftime("%d %b %Y")
        if result_num_days == 1
        else f"{forecast_dates[0].strftime('%d %b')} – {forecast_dates[-1].strftime('%d %b %Y')}"
    )
    st.subheader(f"Forecast — {date_range_label}")

    if result_num_days == 1:
        chart_df = df_results.set_index("Hour")[["Predicted AC Power (W)"]]
    else:
        chart_df = df_results.pivot(index="Hour", columns="Date", values="Predicted AC Power (W)")
        chart_df = chart_df[[d.strftime("%d %b") for d in forecast_dates]]

    # ── Chart options checkboxes beside the chart ──────────
    chart_col, opts_col = st.columns([5, 1])

    with opts_col:
        st.markdown("**Chart options**")
        show_peak = st.checkbox("Peak hour", value=True)
        show_sun  = st.checkbox("Sunrise / Sunset", value=True)
        st.divider()
        if st.button("📊 View", use_container_width=True, help="View raw fetched weather data"):
            st.switch_page("pages/view.py")

    with chart_col:
        fig = go.Figure()

        if result_num_days == 1:
            fig.add_trace(go.Scatter(
                x=chart_df.index,
                y=chart_df["Predicted AC Power (W)"],
                mode="lines",
                name="AC Power",
                line=dict(color="#4fc3f7", width=2)
            ))
            if show_peak:
                peak_hour = int(chart_df["Predicted AC Power (W)"].idxmax())
                peak_val  = chart_df["Predicted AC Power (W)"].max()
                fig.add_vline(
                    x=peak_hour,
                    line_dash="dot", line_color="#ffd54f",
                    annotation_text=f"⬆ Peak {peak_hour:02d}:00 ({peak_val:.0f} W)",
                    annotation_position="top left",
                    annotation_font_color="#ffd54f"
                )
        else:
            colors = ["#4fc3f7", "#81c784", "#ffb74d", "#f06292",
                      "#ba68c8", "#4db6ac", "#fff176"]
            for idx, col in enumerate(chart_df.columns):
                fig.add_trace(go.Scatter(
                    x=chart_df.index,
                    y=chart_df[col],
                    mode="lines",
                    name=col,
                    line=dict(color=colors[idx % len(colors)], width=2)
                ))
                if show_peak:
                    peak_hour = int(chart_df[col].idxmax())
                    peak_val  = chart_df[col].max()
                    fig.add_annotation(
                        x=peak_hour, y=peak_val,
                        text=f"⬆ {peak_val:.0f} W",
                        showarrow=True, arrowhead=2,
                        arrowcolor=colors[idx % len(colors)],
                        font=dict(color=colors[idx % len(colors)], size=11),
                        bgcolor="rgba(0,0,0,0.5)"
                    )

        # ── Sunrise / sunset lines ─────────────────────────
        if show_sun and sunrise_sunset is not None:
            ref_date = forecast_dates[0]
            if ref_date in sunrise_sunset.index:
                sr = sunrise_sunset.loc[ref_date, "sunrise"]
                ss = sunrise_sunset.loc[ref_date, "sunset"]
                sr_hour  = sr.hour + sr.minute / 60
                ss_hour  = ss.hour + ss.minute / 60
                sr_label = sr.strftime("%H:%M")
                ss_label = ss.strftime("%H:%M")
                fig.add_vline(
                    x=sr_hour,
                    line_dash="dash", line_color="#ffcc80",
                    annotation_text=f"🌅 Sunrise {sr_label}",
                    annotation_position="top right",
                    annotation_font_color="#ffcc80"
                )
                fig.add_vline(
                    x=ss_hour,
                    line_dash="dash", line_color="#ef9a9a",
                    annotation_text=f"🌇 Sunset {ss_label}",
                    annotation_position="top left",
                    annotation_font_color="#ef9a9a"
                )

        fig.update_layout(
            xaxis_title="Hour",
            yaxis_title="Predicted AC Power (W)",
            template="plotly_dark",
            legend_title="Date",
            height=420,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        fig.update_xaxes(
            tickmode="array",
            tickvals=list(range(24)),
            ticktext=[f"{h:02d}:00" for h in range(24)]
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Hourly table ───────────────────────────────────────
    st.subheader("Hourly Breakdown")
    display_df = df_results[["Date", "Time", "Predicted AC Power (W)"]].reset_index(drop=True)
    st.dataframe(display_df, use_container_width=True)

    # ── Download button ────────────────────────────────────
    date_range_str = (
        forecast_dates[0].strftime("%Y-%m-%d")
        if result_num_days == 1
        else f"{forecast_dates[0].strftime('%Y-%m-%d')}_to_{forecast_dates[-1].strftime('%Y-%m-%d')}"
    )
    st.download_button(
        label="⬇️ Download Forecast as CSV",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name=f"solar_forecast_{date_range_str}.csv",
        mime="text/csv"
    )

    # ── Live weather table ─────────────────────────────────
    if weather_forecast is not None:
        day_weather = weather_forecast[weather_forecast.index.date == forecast_dates[0]]
        if not day_weather.empty:
            st.subheader("🌤️ Live Weather Inputs Used")
            weather_display = day_weather.rename(columns={
                "shortwave_radiation": "Shortwave (W/m²)",
                "direct_radiation":    "Direct (W/m²)",
                "temperature":         "Temp (°C)",
                "cloudcover":          "Cloud Cover (%)",
                "windspeed":           "Wind (km/h)"
            })
            st.dataframe(
                weather_display.reset_index().rename(columns={"datetime": "Hour"}),
                use_container_width=True
            )


# ── About the Model ────────────────────────────────────────
st.divider()
st.subheader("📖 About the Model")

st.markdown("""
**What is this model?**
This forecast is generated by a Long Short-Term Memory (LSTM) neural network built using TensorFlow
and Keras. It learns solar generation patterns from real historical weather data and applies them
to live weather forecasts to produce genuine hour-by-hour predictions.

**What data was it trained on?**
The model is trained on the last **92 days** of hourly ERA5 reanalysis weather data for Bengaluru,
India, fetched fresh from [Open-Meteo](https://open-meteo.com) each time preprocessing is run.
This means the model always reflects recent seasonal and weather patterns rather than stale
historical averages. AC power output is derived from shortwave radiation using a linear proxy
(1000 W/m² ≈ 5000 W peak output).

**How does it make predictions?**
The model uses **10 input features** per hour:

| Category | Features |
|---|---|
| Time | Hour of day, day of week, month |
| Lag | Previous hour's output, 3-hour rolling average |
| Weather | Shortwave radiation, direct radiation, temperature, cloud cover, wind speed |

Predictions are pre-computed every hour by a GitHub Actions workflow and stored in
`data/forecasts.json`. The dashboard reads from this file instantly — no model inference
happens at request time.

**Model Architecture**
- LSTM layer (64 units) → Dropout (0.2) → Dense output
- Trained with Adam optimiser, MSE loss, early stopping (patience = 5)
- Input shape: (1 timestep × 10 features)

**Limitations**
- AC power is synthetically derived from irradiance; actual inverter efficiency curves are not modelled
- The model does not account for panel degradation, dust, shading, or equipment faults
- 92 days of training data covers ~3 months; accuracy may vary outside the trained season
- A production system would use real measured generation data and retrain continuously
""")
