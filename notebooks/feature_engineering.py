import pandas as pd


df = pd.read_csv('data/cleaned.csv', index_col='datetime', parse_dates=True)

# ── Time features ──────────────────────────────────────────
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month

# ── Lag and rolling features on AC_POWER ──────────────────
df['lag_1'] = df['AC_POWER'].shift(1)
df['rolling_mean_3'] = df['AC_POWER'].shift(1).rolling(window=3).mean()

# ── Weather features are already present from preprocess.py ─
# shortwave_radiation, direct_radiation, temperature, cloudcover, windspeed

df = df.dropna()

df.to_csv('data/features.csv')

print("Done! Shape:", df.shape)
print(df.head(10))