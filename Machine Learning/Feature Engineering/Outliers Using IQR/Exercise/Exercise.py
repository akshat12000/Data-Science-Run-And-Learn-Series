import pandas as pd

df = pd.read_csv('height_weight.csv')

Q1 = df.weight.quantile(0.25)
Q3 = df.weight.quantile(0.75)
IQR = Q3 - Q1

print(df[(df.weight < Q1 - 1.5 * IQR) | (df.weight > Q3 + 1.5 * IQR)])

Q1 = df.height.quantile(0.25)
Q3 = df.height.quantile(0.75)
IQR = Q3 - Q1

print(df[(df.height < Q1 - 1.5 * IQR) | (df.height > Q3 + 1.5 * IQR)])

