import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm
import numpy as np

df = pd.read_csv('heights.csv')

plt.hist(df['height'], bins=20, rwidth=0.8,density=True)
plt.xlabel('height(inches)')
plt.ylabel('Count')

rng = np.arange(df['height'].min(), df['height'].max(), 0.1)
plt.plot(rng, norm.pdf(rng, df['height'].mean(), df['height'].std()))
plt.show()

# Removing Outliers using Z Score

df = df[(np.abs(df['height'] - df['height'].mean()) <= (3 * df['height'].std()))]

plt.hist(df['height'], bins=20, rwidth=0.8,density=True)
plt.xlabel('height(inches)')
plt.ylabel('Count')

rng = np.arange(df['height'].min(), df['height'].max(), 0.1)
plt.plot(rng, norm.pdf(rng, df['height'].mean(), df['height'].std()))
plt.show()


