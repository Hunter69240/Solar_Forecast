import streamlit as st
import requests
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="Solar Forecast", page_icon="☀️")
st.title("☀️ Solar Power Generation Forecast")

# ── Load scaler to normalize lag values ───────────────────
scaler = joblib.load('models/scaler.pkl')

# ── Generate next 24 hours ─────────────────────────────────
now = datetime.now().replace(minute=0, second=0, microsecond=0)
hours = [now + timedelta(hours=i) for i in range(24)]

# ── Call API for each hour ─────────────────────────────────
predictions_real = []      # inverse transformed (watts)
predictions_scaled = []    # normalized (0 to 1) for lag feedback

for i, hour in enumerate(hours):
    # lag and rolling in normalized scale
    lag_1 = predictions_scaled[i-1] if i > 0 else 0.0
    rolling_mean_3 = float(np.mean(predictions_scaled[max(0, i-3):i])) if i > 0 else 0.0

    payload = {
        "hour": hour.hour,
        "day_of_week": hour.weekday(),
        "month": hour.month,
        "lag_1": lag_1,
        "rolling_mean_3": rolling_mean_3
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload)
        result = response.json()
        real_val = result["predicted_ac_power"]
        # scale it back down for lag feedback
        scaled_val = float(scaler.transform([[real_val]])[0][0])
        predictions_real.append(round(max(0.0, real_val), 2))
        predictions_scaled.append(scaled_val)
    except:
        predictions_real.append(0.0)
        predictions_scaled.append(0.0)

# ── Build dataframe ────────────────────────────────────────
df = pd.DataFrame({
    "Time": [h.strftime("%H:%M") for h in hours],
    "Predicted AC Power (W)": predictions_real
})

# ── Display ────────────────────────────────────────────────
st.subheader("24-Hour Forecast")
st.line_chart(df.set_index("Time"))

st.subheader("Hourly Breakdown")
st.dataframe(df, use_container_width=True)