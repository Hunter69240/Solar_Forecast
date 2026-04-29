import pandas as pd

# ── Load cleaned data ──────────────────────────────────────
df = pd.read_csv('data/cleaned.csv', index_col='DATE_TIME', parse_dates=True)

# ── Time features ──────────────────────────────────────────
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month

# ── Lag feature (previous hour's generation) ───────────────
df['lag_1'] = df['AC_POWER'].shift(1)

# ── Rolling average (last 3 hours) ────────────────────────
df['rolling_mean_3'] = df['AC_POWER'].shift(1).rolling(window=3).mean()

# ── Drop rows with nulls created by lag/rolling ───────────
df = df.dropna()

# ── Save ───────────────────────────────────────────────────
df.to_csv('data/features.csv')

print("Done! Shape:", df.shape)
print(df.head(10))