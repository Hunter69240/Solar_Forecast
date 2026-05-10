## preprocess.py

### Purpose
Fetches weather data from Open-Meteo API and prepares it for model training.

### What it does
1. Fetches 92 days of hourly weather data for Bengaluru (lat: 12.9716, lon: 77.5946) from Open-Meteo API
2. Converts response into a dataframe with datetime set as the index
3. Simulates solar AC power output using the formula:
   `AC_POWER = (shortwave_radiation / 1000) * 5000`
   (assumes peak capacity of 5000W at 1000 W/m² irradiance)
4. Fills missing values with 0
5. Normalizes AC_POWER to 0–1 range using MinMaxScaler
6. Saves outputs to disk

### Inputs
- Open-Meteo forecast API (live internet call)
- Variables fetched: `shortwave_radiation`, `direct_radiation`, `temperature_2m`, `cloudcover`, `windspeed_10m`

### Outputs
| File | Location | Description |
|------|----------|-------------|
| cleaned.csv | data/ | Processed dataframe with all features |
| scaler.pkl | models/ | Saved MinMaxScaler for use during prediction |

### Key note
`scaler.pkl` must be preserved — it is used by the API to convert normalized predictions back to real watt values. If lost, predictions become meaningless.

### Dependencies
- `pandas`, `numpy`, `sklearn`, `joblib`, `requests`