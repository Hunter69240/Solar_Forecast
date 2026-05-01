import pandas as pd


df = pd.read_csv('data/cleaned.csv', index_col='DATE_TIME', parse_dates=True)

df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month


df['lag_1'] = df['AC_POWER'].shift(1)

df['rolling_mean_3'] = df['AC_POWER'].shift(1).rolling(window=3).mean()

df = df.dropna()

df.to_csv('data/features.csv')

print("Done! Shape:", df.shape)
print(df.head(10))