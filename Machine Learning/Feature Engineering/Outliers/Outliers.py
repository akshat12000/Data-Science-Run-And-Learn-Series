import pandas as pd

df = pd.read_csv('heights.csv')

max_threshold = df['height'].quantile(0.95)
min_threshold = df['height'].quantile(0.05)

df = df[(df['height'] < max_threshold) & (df['height'] > min_threshold)]

print(df)

df2 = pd.read_csv('bhp.csv')

min_threshold,max_threshold = df2.price_per_sqft.quantile([0.001,0.999])

df2 = df2[(df2.price_per_sqft < max_threshold) & (df2.price_per_sqft > min_threshold)]

print(df2)