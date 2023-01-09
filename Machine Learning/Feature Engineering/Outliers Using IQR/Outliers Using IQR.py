import numpy as np
import pandas as pd

df = pd.read_csv('heights.csv')

Q1 = df.height.quantile(0.25)
Q3 = df.height.quantile(0.75)
IQR = Q3 - Q1

df = df[(df.height > Q1 - 1.5 * IQR) & (df.height < Q3 + 1.5 * IQR)]

print(df)