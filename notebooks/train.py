import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os


df = pd.read_csv('data/features.csv', index_col='datetime', parse_dates=True)


FEATURES = [
    'hour', 'day_of_week', 'month',
    'lag_1', 'rolling_mean_3',
    'shortwave_radiation', 'direct_radiation',
    'temperature', 'cloudcover', 'windspeed'
]

X = df[FEATURES].values
y = df['AC_POWER'].values


X = X.reshape((X.shape[0], 1, X.shape[1]))


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)


model = Sequential([
    LSTM(64, input_shape=(1, len(FEATURES)), return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()


os.makedirs('models', exist_ok=True)

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint('models/lstm_model.h5', save_best_only=True)
]


history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

loss = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest MSE: {loss:.4f}")
print("Model saved to models/lstm_model.h5")