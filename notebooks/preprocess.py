import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import requests
import os

# ── Fetch data from Open-Meteo ─────────────────────────────
# Bengaluru coordinates: 12.9716° N, 77.5946° E
url = "https://archive.open-meteo.com/v1/archive"

params = {
    "latitude": 12.9716,
    "longitude": 77.5946,
    "start_date": "2023-01-01",
    "end_date": "2024-12-31",
    "hourly": "shortwave_radiation,direct_radiation,temperature_2m,cloudcover,windspeed_10m",
    "timezone": "Asia/Kolkata"
}

print("Fetching data from Open-Meteo...")
response = requests.get(url, params=params)
data = response.json()

# ── Build dataframe ────────────────────────────────────────
df = pd.DataFrame({
    "datetime": data["hourly"]["time"],
    "shortwave_radiation": data["hourly"]["shortwave_radiation"],
    "direct_radiation": data["hourly"]["direct_radiation"],
    "temperature": data["hourly"]["temperature_2m"],
    "cloudcover": data["hourly"]["cloudcover"],
    "windspeed": data["hourly"]["windspeed_10m"]
})

# ── Fix datetime ───────────────────────────────────────────
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.set_index("datetime")

# ── Create synthetic AC power from irradiance ──────────────
# Since Open-Meteo gives weather not actual power output,
# we simulate AC power using shortwave radiation as a proxy
# Peak capacity assumed at 1000 W/m2 irradiance = 5000W output
df["AC_POWER"] = (df["shortwave_radiation"] / 1000) * 5000
df["AC_POWER"] = df["AC_POWER"].clip(lower=0)

# ── Handle missing values ──────────────────────────────────
df = df.fillna(0)

# ── Normalize AC_POWER ─────────────────────────────────────
scaler = MinMaxScaler()
df["AC_POWER"] = scaler.fit_transform(df[["AC_POWER"]])

# ── Save ───────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
df.to_csv("data/cleaned.csv")
joblib.dump(scaler, "models/scaler.pkl")

print("Done! Shape:", df.shape)
print(df.head(10))