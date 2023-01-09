import pandas as pd

# Read the data
df = pd.read_csv('AB_NYC_2019.csv')

# Remove rows with missing values
df = df.dropna()

df['price_per_night'] = df['price'] / df['minimum_nights']

# Remove rows with extreme values
max_threshold = df['price_per_night'].quantile(0.95)
min_threshold = df['price_per_night'].quantile(0.05)

df = df[(df['price_per_night'] < max_threshold) & (df['price_per_night'] > min_threshold)]

print(df)