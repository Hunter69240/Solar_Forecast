import streamlit as st
import requests
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta
import os
import time
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Solar Forecast", page_icon="☀️")
st.title("☀️ Solar Power Generation Forecast")
st.caption("📍 Location: Bengaluru, India — Plant 1, Inverter 1BY6WEcLGh8j5v7")

# ── Session state defaults ─────────────────────────────────
if "btn_state" not in st.session_state:
    st.session_state.btn_state = "idle"        # idle | loading | done
if "results" not in st.session_state:
    st.session_state.results = None
if "last_num_days" not in st.session_state:
    st.session_state.last_num_days = 3

# ── Fetch live weather forecast from Open-Meteo ───────────
@st.cache_data(ttl=3600)
def fetch_weather_forecast():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 12.9716,
        "longitude": 77.5946,
        "hourly": "shortwave_radiation,direct_radiation,temperature_2m,cloudcover,windspeed_10m",
        "timezone": "Asia/Kolkata",
        "forecast_days": 7
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        weather_df = pd.DataFrame({
            "datetime": pd.to_datetime(data["hourly"]["time"]),
            "shortwave_radiation": data["hourly"]["shortwave_radiation"],
            "direct_radiation": data["hourly"]["direct_radiation"],
            "temperature": data["hourly"]["temperature_2m"],
            "cloudcover": data["hourly"]["cloudcover"],
            "windspeed": data["hourly"]["windspeed_10m"]
        })
        weather_df = weather_df.set_index("datetime")
        return weather_df
    except Exception:
        return None

weather_forecast = fetch_weather_forecast()

@st.cache_data(ttl=3600)
def fetch_sunrise_sunset():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 12.9716,
        "longitude": 77.5946,
        "daily": "sunrise,sunset",
        "timezone": "Asia/Kolkata",
        "forecast_days": 7
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        df = pd.DataFrame({
            "date": pd.to_datetime(data["daily"]["time"]).date,
            "sunrise": pd.to_datetime(data["daily"]["sunrise"]),
            "sunset": pd.to_datetime(data["daily"]["sunset"])
        })
        df = df.set_index("date")
        return df
    except Exception:
        return None

sunrise_sunset = fetch_sunrise_sunset()

if weather_forecast is None:
    st.warning("⚠️ Could not fetch live weather forecast. Predictions will fall back to zeros for weather inputs.")

scaler = joblib.load('models/scaler.pkl')

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
    st.session_state.btn_state = "idle"
    st.session_state.results = None

forecast_dates = [today + timedelta(days=d) for d in range(num_days)]

# ── Button rendering ───────────────────────────────────────
if st.session_state.btn_state == "idle":
    clicked = st.button(" Make Prediction", type="primary", use_container_width=True)
    if clicked:
        st.session_state.btn_state = "loading"
        st.rerun()

elif st.session_state.btn_state == "loading":
    st.button(" Predicting...", disabled=True, type="primary", use_container_width=True)

    # ── Run predictions ────────────────────────────────────
    hours = [
        datetime(d.year, d.month, d.day, h)
        for d in forecast_dates
        for h in range(24)
    ]

    predictions_real = []
    predictions_scaled = []

    for i, hour in enumerate(hours):
        lag_1 = predictions_scaled[i-1] if i > 0 else 0.0
        rolling_mean_3 = float(np.mean(predictions_scaled[max(0, i-3):i])) if i > 0 else 0.0

        if weather_forecast is not None and hour in weather_forecast.index:
            w = weather_forecast.loc[hour]
            shortwave = float(w["shortwave_radiation"] or 0)
            direct = float(w["direct_radiation"] or 0)
            temperature = float(w["temperature"] or 25)
            cloudcover = float(w["cloudcover"] or 0)
            windspeed = float(w["windspeed"] or 0)
        else:
            shortwave = direct = cloudcover = windspeed = 0.0
            temperature = 25.0

        payload = {
            "hour": hour.hour,
            "day_of_week": hour.weekday(),
            "month": hour.month,
            "lag_1": lag_1,
            "rolling_mean_3": rolling_mean_3,
            "shortwave_radiation": shortwave,
            "direct_radiation": direct,
            "temperature": temperature,
            "cloudcover": cloudcover,
            "windspeed": windspeed
        }

        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            result = response.json()
            real_val = result["predicted_ac_power"]
            scaled_val = float(scaler.transform([[real_val]])[0][0])
            predictions_real.append(round(max(0.0, real_val), 2))
            predictions_scaled.append(max(0.0, scaled_val))
        except Exception:
            predictions_real.append(0.0)
            predictions_scaled.append(0.0)

    df_results = pd.DataFrame({
        "Datetime": hours,
        "Date": [h.strftime("%d %b") for h in hours],
        "Hour": [h.hour for h in hours],
        "Time": [h.strftime("%H:%M") for h in hours],
        "Predicted AC Power (W)": predictions_real
    })

    st.session_state.results = {
        "df": df_results,
        "forecast_dates": forecast_dates,
        "num_days": num_days
    }
    st.session_state.btn_state = "done"
    st.rerun()

elif st.session_state.btn_state == "done":
    st.button("✅ Predicted!", disabled=True, type="primary", use_container_width=True)

# ── Display results if available ──────────────────────────
if st.session_state.results is not None:
    r = st.session_state.results
    df_results = r["df"]
    forecast_dates = r["forecast_dates"]
    result_num_days = r["num_days"]

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
        show_sun = st.checkbox("Sunrise / Sunset", value=True)

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
                peak_val = chart_df["Predicted AC Power (W)"].max()
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
                    peak_val = chart_df[col].max()
                    fig.add_annotation(
                        x=peak_hour, y=peak_val,
                        text=f"⬆ {peak_val:.0f} W",
                        showarrow=True, arrowhead=2,
                        arrowcolor=colors[idx % len(colors)],
                        font=dict(color=colors[idx % len(colors)], size=11),
                        bgcolor="rgba(0,0,0,0.5)"
                    )

        # Sunrise / sunset vertical lines (based on first forecast day)
        if show_sun and sunrise_sunset is not None:
            ref_date = forecast_dates[0]
            if ref_date in sunrise_sunset.index:
                sr_hour = sunrise_sunset.loc[ref_date, "sunrise"].hour + sunrise_sunset.loc[ref_date, "sunrise"].minute / 60
                ss_hour = sunrise_sunset.loc[ref_date, "sunset"].hour + sunrise_sunset.loc[ref_date, "sunset"].minute / 60
                sr_label = sunrise_sunset.loc[ref_date, "sunrise"].strftime("%H:%M")
                ss_label = sunrise_sunset.loc[ref_date, "sunset"].strftime("%H:%M")
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

    st.subheader("Hourly Breakdown")
    display_df = df_results[["Date", "Time", "Predicted AC Power (W)"]].reset_index(drop=True)
    st.dataframe(display_df, use_container_width=True)

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

    if weather_forecast is not None:
        day_weather = weather_forecast[weather_forecast.index.date == forecast_dates[0]]
        if not day_weather.empty:
            st.subheader("🌤️ Live Weather Inputs Used")
            weather_display = day_weather.rename(columns={
                "shortwave_radiation": "Shortwave (W/m²)",
                "direct_radiation": "Direct (W/m²)",
                "temperature": "Temp (°C)",
                "cloudcover": "Cloud Cover (%)",
                "windspeed": "Wind (km/h)"
            })
            st.dataframe(weather_display.reset_index().rename(columns={"datetime": "Hour"}), use_container_width=True)

# ── Reset "done" button back to idle after 2 seconds ──────
if st.session_state.btn_state == "done":
    time.sleep(2)
    st.session_state.btn_state = "idle"
    st.rerun()


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

For each forecast hour, the dashboard fetches the **live Open-Meteo weather forecast** and passes 
the actual predicted conditions into the model — so the output curve you see is driven by real 
forecast irradiance and cloud cover, not just time-of-day patterns.

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