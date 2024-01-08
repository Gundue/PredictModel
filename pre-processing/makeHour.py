import pandas as pd

df = pd.read_csv('', encoding='cp949')

df['date'] = pd.to_datetime(df.date)

df = df.set_index('date')

resam = df.resample(rule='H').mean().round(1)

resam.to_csv('resam.csv')