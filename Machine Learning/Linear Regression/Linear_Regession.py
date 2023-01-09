import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv('homeprices.csv')
print(df)

reg=linear_model.LinearRegression()
reg.fit(df[['area']],df.price)

plt.xlabel('area in sqft')
plt.ylabel('price in USD')
plt.scatter(df["area"], df["price"], color='red', marker='+')
plt.plot(df["area"], reg.predict(df[['area']]), color='blue')
plt.show()

print(reg.predict([[3300]]))
print(reg.coef_,reg.intercept_)

d=pd.read_csv('areas.csv')
p=reg.predict(d)
d['prices']=p
d.to_csv('prediction.csv',index=False)