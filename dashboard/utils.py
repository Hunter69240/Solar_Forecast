import requests
import pandas as pd
import streamlit as st

LATITUDE = 12.9716
LONGITUDE = 77.5946


@st.cache_data(ttl=3600)
def fetch_weather_forecast():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
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
            "direct_radiation":    data["hourly"]["direct_radiation"],
            "temperature":         data["hourly"]["temperature_2m"],
            "cloudcover":          data["hourly"]["cloudcover"],
            "windspeed":           data["hourly"]["windspeed_10m"]
        })
        weather_df = weather_df.set_index("datetime")
        return weather_df
    except Exception:
        return None


@st.cache_data(ttl=3600)
def fetch_sunrise_sunset():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "daily": "sunrise,sunset",
        "timezone": "Asia/Kolkata",
        "forecast_days": 7
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        df = pd.DataFrame({
            "date":    pd.to_datetime(data["daily"]["time"]).date,
            "sunrise": pd.to_datetime(data["daily"]["sunrise"]),
            "sunset":  pd.to_datetime(data["daily"]["sunset"])
        })
        df = df.set_index("date")
        return df
    except Exception:
        return None
