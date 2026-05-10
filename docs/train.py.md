## train.py

### Purpose
Builds, trains and saves the LSTM model.

### What it does
1. Reads `data/features.csv` and defines 10 input features
2. Splits data into 80% train, 20% test with `shuffle=False`
   (order must be preserved — future cannot train on past)
3. Reshapes X to 3D: `(samples, timesteps=1, features)`
   because LSTM requires 3D input
4. Builds model architecture:
   - `LSTM(64)` — learns temporal patterns from sequences
   - `Dropout(0.2)` — randomly disables 20% of neurons each 
     training step to prevent overfitting
   - `Dense(1)` — outputs a single prediction value
5. Uses two callbacks:
   - `EarlyStopping(patience=5)` — stops training if validation 
     loss does not decrease for 5 consecutive epochs
   - `ModelCheckpoint` — saves only the best model, not the last
6. Evaluates final model on test set, reports MSE

### Inputs
| File | Location |
|------|----------|
| features.csv | data/ |

### Outputs
| File | Location | Description |
|------|----------|-------------|
| lstm_model.h5 | models/ | Trained LSTM model |

### Dependencies
- `pandas`, `numpy`, `tensorflow`, `sklearn`