import os
import json
import numpy as np
import pandas as pd
import requests
import joblib
import tensorflow as tf
from datetime import datetime, timedelta

# ── Paths (always relative to repo root, regardless of where script is run from) ──
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "lstm_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
OUTPUT_PATH = os.path.join(BASE_DIR, "data",   "forecasts.json")

# ── Location (Bengaluru) ───────────────────────────────────
LATITUDE  = 12.9716
LONGITUDE = 77.5946


# ── Step 1: Fetch 7-day hourly weather from Open-Meteo ────
def fetch_weather():
    print("  Fetching weather from Open-Meteo...")
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":  LATITUDE,
        "longitude": LONGITUDE,
        "hourly":    "shortwave_radiation,direct_radiation,temperature_2m,cloudcover,windspeed_10m",
        "timezone":  "Asia/Kolkata",
        "forecast_days": 7
    }
    response = requests.get(url, params=params, timeout=10)
    data = response.json()

    weather_df = pd.DataFrame({
        "datetime":           pd.to_datetime(data["hourly"]["time"]),
        "shortwave_radiation": data["hourly"]["shortwave_radiation"],
        "direct_radiation":    data["hourly"]["direct_radiation"],
        "temperature":         data["hourly"]["temperature_2m"],
        "cloudcover":          data["hourly"]["cloudcover"],
        "windspeed":           data["hourly"]["windspeed_10m"]
    })
    weather_df = weather_df.set_index("datetime")
    print(f"  Got {len(weather_df)} hourly weather rows.")
    return weather_df


# ── Step 2: Run the sequential 168-step prediction loop ───
def run_predictions(model, scaler, weather_df):
    today = datetime.now().date()
    forecast_dates = [today + timedelta(days=d) for d in range(7)]

    # All 168 hours: 7 days × 24 hours
    hours = [
        datetime(d.year, d.month, d.day, h)
        for d in forecast_dates
        for h in range(24)
    ]

    predictions_real   = []
    predictions_scaled = []

    for i, hour in enumerate(hours):
        # Lag features — computed from previous predictions (sequential dependency)
        lag_1          = predictions_scaled[i - 1] if i > 0 else 0.0
        rolling_mean_3 = float(np.mean(predictions_scaled[max(0, i - 3):i])) if i > 0 else 0.0

        # Weather inputs for this hour
        if weather_df is not None and hour in weather_df.index:
            w         = weather_df.loc[hour]
            shortwave = float(w["shortwave_radiation"] or 0)
            direct    = float(w["direct_radiation"]    or 0)
            temperature = float(w["temperature"]       or 25)
            cloudcover  = float(w["cloudcover"]        or 0)
            windspeed   = float(w["windspeed"]         or 0)
        else:
            shortwave = direct = cloudcover = windspeed = 0.0
            temperature = 25.0

        # Build feature vector — same order as training
        X = np.array([[
            hour.hour, hour.weekday(), hour.month,
            lag_1, rolling_mean_3,
            shortwave, direct, temperature, cloudcover, windspeed
        ]])
        X = X.reshape((1, 1, 10))  # (samples, timesteps, features)

        try:
            pred_scaled = float(model.predict(X, verbose=0)[0][0])
            real_val    = float(scaler.inverse_transform([[pred_scaled]])[0][0])
            real_val    = round(max(0.0, real_val), 2)
            scaled_val  = max(0.0, pred_scaled)
        except Exception as e:
            print(f"  Warning: prediction failed at hour {hour} — {e}")
            real_val   = 0.0
            scaled_val = 0.0

        predictions_real.append(real_val)
        predictions_scaled.append(scaled_val)

    return hours, predictions_real


# ── Step 3: Write results to data/forecasts.json ──────────
def save_forecast(hours, predictions):
    generated_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    output = {
        "generated_at": generated_at,
        "predictions": [
            {
                "datetime": h.strftime("%Y-%m-%dT%H:%M:%S"),
                "ac_power": p
            }
            for h, p in zip(hours, predictions)
        ]
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Saved {len(predictions)} predictions → {OUTPUT_PATH}")
    print(f"  Generated at: {generated_at}")


# ── Main ───────────────────────────────────────────────────
def main():
    print("=== Suprmentr Hourly Forecast Job ===")

    print("[1/3] Loading model and scaler...")
    model  = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    print("[2/3] Fetching weather forecast...")
    weather_df = fetch_weather()

    print("[3/3] Running 168-step prediction loop...")
    hours, predictions = run_predictions(model, scaler, weather_df)

    print("[4/4] Saving forecast...")
    save_forecast(hours, predictions)

    print("=== Done ===")


if __name__ == "__main__":
    main()
