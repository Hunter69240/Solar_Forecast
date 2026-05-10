## dashboard/app.py

### Purpose
The main UI of the application. Displays solar power predictions 
as an interactive chart and table.

### What it does
1. Takes user input via a slider (1–7 days to forecast)
2. Fetches live weather data via utils.py
3. Loops through every hour of selected days and calls POST /predict
   for each hour with live weather values as raw input
4. Displays results as:
   - Daily total metric cards
   - Plotly line chart with sunrise/sunset markers
   - Hourly breakdown table
   - CSV download button
   - Live weather inputs used

### Key ML concept — Autoregressive prediction
Each hour's prediction is fed back as the next hour's lag input:
```python
lag_1 = predictions_scaled[i-1] if i > 0 else 0.0
```
The model predicts hour by hour, feeding its own output 
back as input for the next hour.

### Session state
Streamlit reruns the entire script on every interaction.
Session state preserves data across reruns:

| Key | Purpose |
|-----|---------|
| btn_state | Tracks button state: idle / loading / done |
| results | Stores prediction dataframe |
| last_num_days | Detects if slider changed |

### API communication
- Reads API_URL from .env file
- Default: http://127.0.0.1:8000
- Sends raw weather values to /predict — no scaling done here
- Scaling and inverse transform happens inside the API

### Inputs
| Source | Description |
|--------|-------------|
| utils.fetch_weather_forecast() | 7 days of hourly weather |
| utils.fetch_sunrise_sunset() | Sunrise/sunset times for display |
| POST /predict response | predicted_ac_power per hour |
| models/scaler.pkl | Used locally to compute scaled lag values |

### Outputs
| Output | Description |
|--------|-------------|
| Plotly chart | Hour by hour power curve per day |
| Metric cards | Daily total power per selected day |
| Dataframe table | Hourly breakdown |
| CSV download | Exportable forecast |

### Dependencies
- `streamlit`, `plotly`, `requests`, `pandas`, `numpy`, `joblib`