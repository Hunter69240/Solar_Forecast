import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# ── Load data ──────────────────────────────────────────────
df = pd.read_csv('data/Plant_1_Generation_Data.csv')

# ── Pick one inverter ──────────────────────────────────────
df = df[df['SOURCE_KEY'] == '1BY6WEcLGh8j5v7'].copy()

# ── Keep only what we need ─────────────────────────────────
df = df[['DATE_TIME', 'AC_POWER']]

# ── Fix datetime ───────────────────────────────────────────
df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], dayfirst=True)
df = df.set_index('DATE_TIME')

# ── Resample to hourly (sum the 15 min intervals) ──────────
df = df.resample('h').sum()

# ── Handle missing values ──────────────────────────────────
df = df.fillna(0)

# ── Normalize ──────────────────────────────────────────────
scaler = MinMaxScaler()
df['AC_POWER'] = scaler.fit_transform(df[['AC_POWER']])

# ── Save cleaned data and scaler ──────────────────────────
os.makedirs('models', exist_ok=True)
df.to_csv('data/cleaned.csv')
joblib.dump(scaler, 'models/scaler.pkl')

print("Done! Shape:", df.shape)
print(df.head(10))