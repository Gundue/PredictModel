import pandas as pd

df = pd.read_csv('')

print(df)
for i in range(1, 13):
    q1 = df.iloc[:, i].quantile(0.25)
    q3 = df.iloc[:, i].quantile(0.75)
    iqr = q3 - q1
    outlier = iqr * 1.5
    df = df[df.iloc[:, i] < q1 - outlier]
    # df = df[df.iloc[:, i] > (q3 + outlier)]

print(df)
