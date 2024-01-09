from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sqlalchemy import create_engine
from sklearn import metrics

df = pd.read_csv('.csv', nrows=1000)
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='raise')

conn = _engine_aircok.connect()

df2 = pd.read_sql("", conn, index_col=None)


# 시계열 ts 데이터 생성
timeSeries = df.loc[:, ["date", "temperature"]]
timeSeries.index = timeSeries.date
ts = timeSeries.drop("date", axis=1)

# 정상성 확인을 위한 시계열 분해 (추세, 순환, 계절성, 불규칙 요소)
result = seasonal_decompose(ts, model='additive', period=3)
fig = plt.figure()
fig = result.plot()

# 차분
ts_diff = ts - ts.shift()

# ACF, PACF 확인 AR : 3, MA : 5
fig = plt.figure(figsize=(20, 8))
ax1 = fig.add_subplot(2, 1, 1)
fig = sm.graphics.tsa.plot_acf(ts_diff[1:], lags=20, ax=ax1)
ax2 = fig.add_subplot(2, 1, 2)
fig = sm.graphics.tsa.plot_pacf(ts_diff[1:], lags=20, ax=ax2)

#  p-value값 확인
rs = adfuller(ts['temperature'])
print('%f' % rs[1])

# model 학습
model = sm.tsa.arima.ARIMA(ts, order=(5, 1, 6))
model_fit = model.fit()

print(model_fit.summary())

# 12개 예측값
forecast = model_fit.forecast(steps=12)
print(forecast, df2['temperature'][0:12])

# 실제 데이터
plt.figure(figsize=(22, 8))
plt.plot(np.array(df2['temperature']), label="original")
plt.plot(np.array(forecast), label="predicted")
plt.xlabel("Date")
plt.ylabel("temperature")
plt.legend()

# mae값
print("MAE : %f" % mean_absolute_error(forecast, df2['temperature'][0:12]))

plt.show()

