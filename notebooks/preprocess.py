import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

df = pd.read_csv('data/Plant_1_Generation_Data.csv')

df = df[df['SOURCE_KEY'] == '1BY6WEcLGh8j5v7'].copy()

df = df[['DATE_TIME', 'AC_POWER']]

df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'], dayfirst=True)
df = df.set_index('DATE_TIME')

df = df.resample('h').sum()

df = df.fillna(0)

scaler = MinMaxScaler()
df['AC_POWER'] = scaler.fit_transform(df[['AC_POWER']])

os.makedirs('models', exist_ok=True)
df.to_csv('data/cleaned.csv')
joblib.dump(scaler, 'models/scaler.pkl')

print("Done! Shape:", df.shape)
print(df.head(10))