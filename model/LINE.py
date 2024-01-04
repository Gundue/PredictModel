from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("", encoding='euc-kr')
df = df.dropna()
df = df.reset_index(drop=True)

x = df["high"]
y = df['low']

line_fitter = LinearRegression()
line_fitter.fit(x.values.reshape(-1, 1), y)

plt.plot(x, y, 'o')
plt.plot(x, line_fitter.predict(x.values.reshape(-1, 1)))
predict = line_fitter.predict([[0.9]])
print(predict)

plt.show()
