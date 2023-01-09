import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

df = pd.read_csv('bhp.csv')

df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']

# Removing Outliers using percentile
min_threshold, max_threshold = df['price_per_sqft'].quantile([0.001, 0.999])

df = df[(df['price_per_sqft'] > min_threshold) & (df['price_per_sqft'] < max_threshold)]

plt.hist(df['price_per_sqft'], bins=20, rwidth=0.8, density=True)
plt.xlabel('price_per_sqft')
plt.ylabel('Count')

rng = np.arange(df['price_per_sqft'].min(), df['price_per_sqft'].max(), 0.1)
plt.plot(rng, norm.pdf(rng, df['price_per_sqft'].mean(), df['price_per_sqft'].std()))
plt.show()

# Removing Outliers using Z Score

df = df[(np.abs(df['price_per_sqft'] - df['price_per_sqft'].mean()) <= (3 * df['price_per_sqft'].std()))]

plt.hist(df['price_per_sqft'], bins=20, rwidth=0.8, density=True)
plt.xlabel('price_per_sqft')
plt.ylabel('Count')

rng = np.arange(df['price_per_sqft'].min(), df['price_per_sqft'].max(), 0.1)
plt.plot(rng, norm.pdf(rng, df['price_per_sqft'].mean(), df['price_per_sqft'].std()))
plt.show()