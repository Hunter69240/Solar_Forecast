## dashboard/utils.py

### Purpose
Handles all external API calls needed by the dashboard.

### Functions

**`fetch_weather_forecast()`**
- Fetches 7 days of hourly weather data for Bengaluru
- Returns dataframe with radiation, temperature, cloudcover, windspeed
- This data is sent to /predict endpoint to generate power forecasts
- Returns None if API call fails

**`fetch_sunrise_sunset()`**
- Fetches sunrise and sunset times for next 7 days
- Used for display purposes only, not for prediction
- Returns None if API call fails

### Caching
Both functions use `@st.cache_data(ttl=3600)`:
- First call fetches from API and stores result in memory
- All subsequent calls within 1 hour return stored result
- After 1 hour cache expires and fresh data is fetched
- Tradeoff: sudden weather changes won't reflect until cache expires

### Why separate from app.py
Clean separation of concerns — if the weather API changes,
only this file needs to be updated.

### Dependencies
- `requests`, `pandas`, `streamlit`