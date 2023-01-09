import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


df = pd.read_csv("carprices.csv")
# print(df)

X = df['Mileage']
Y = df['Sell Price($)']
# print(X,Y)

plt.scatter(X,Y)
plt.show()

# From Plot it is clear that there is a linear relationship between Mileage and Sell Price
# So we can use Linear Regression to predict the Sell Price

dummy = pd.get_dummies(df['Car Model'])
df = pd.concat([df, dummy], axis='columns')
df = df.drop(['Car Model', 'Mercedez Benz C class'], axis='columns')

# print(df)

X = df.drop('Sell Price($)', axis='columns')
Y = df['Sell Price($)']

model = linear_model.LinearRegression()
model.fit(X, Y)

print(model.predict([[45000, 4, 0, 0]]))
print(model.predict([[86000, 7, 0, 1]]))

print(model.score(X, Y))

