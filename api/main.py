from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import joblib


model = tf.keras.models.load_model('models/lstm_model.h5')
scaler = joblib.load('models/scaler.pkl')

app = FastAPI()

class PredictRequest(BaseModel):
    hour: int
    day_of_week: int
    month: int
    lag_1: float
    rolling_mean_3: float


@app.get("/")
def root():
    return {"status": "Solar Forecast API is running"}

# ── Predict endpoint ───────────────────────────────────────
@app.post("/predict")
def predict(data: PredictRequest):
    # Build input array
    X = np.array([[data.hour, data.day_of_week, data.month,
                   data.lag_1, data.rolling_mean_3]])
    
    # Reshape for LSTM (samples, timesteps, features)
    X = X.reshape((1, 1, 5))
    
    # Predict
    prediction_scaled = model.predict(X, verbose=0)[0][0]
    
    # Inverse transform to get real value
    prediction = scaler.inverse_transform([[prediction_scaled]])[0][0]
    
    return {
        "predicted_ac_power": round(float(prediction), 4)
    }