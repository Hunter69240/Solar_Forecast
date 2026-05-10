## api/main.py

### Purpose
Serves the trained LSTM model as a REST API.

### What it does
1. Loads `lstm_model.h5` and `scaler.pkl` once at startup
2. Defines a `PredictRequest` Pydantic model that validates 
   all incoming JSON — wrong types are rejected automatically
3. Exposes two routes:
   - `GET /` — health check, confirms API is running
   - `POST /predict` — accepts 10 weather/time features as JSON,
     returns predicted AC power in watts

### Prediction flow
1. Converts JSON payload to numpy array
2. Reshapes to `(1, 1, 10)` — LSTM always requires 3D input
3. Calls `model.predict()` — returns a value between 0 and 1
4. Calls `scaler.inverse_transform()` to convert back to watts
5. Returns `predicted_ac_power` rounded to 4 decimal places

### Inputs
| Source | Description |
|--------|-------------|
| models/lstm_model.h5 | Trained LSTM model |
| models/scaler.pkl | Saved scaler for inverse transform |
| POST /predict JSON | 10 feature values from the user |

### Outputs
```json
{
  "predicted_ac_power": 3847.2631
}
```

### Dependencies
- `fastapi`, `uvicorn`, `pydantic`, `tensorflow`, `joblib`, `numpy`