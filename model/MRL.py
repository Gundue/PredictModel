import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

_engine = create_engine()

conn = _engine.connect()

df = pd.read_sql("", conn, index_col=None)

# 이산화탄소, 습도, 미세먼지
x = df[['co2', 'humidity', 'pm10', 'pm25']]
y = df[['temperature']]

# 데이터 분리 (학습데이터, 테스트 데이터)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2)

model_lr = LinearRegression()
model_lr.fit(x_train, y_train)

# 테스트 데이터로 예측값
y_predict = model_lr.predict(x_test)

# 기울기
print(model_lr.coef_)
print(y[:, 3])

# 잔차분석 정확도
print(model_lr.score(x, y))

# 예측 온도와 실제 온도 비교
plt.scatter(y_test, y_predict, alpha=0.4)
#plt.scatter(df[['humidity']], df[['temperature']], alpha=0.4)

plt.show()
