import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model

df = pd.read_csv("carprices.csv")

X = df[['Mileage', 'Age(yrs)']]
Y = df['Sell Price($)']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = linear_model.LinearRegression()
model.fit(X_train, Y_train)

print(model.predict(X_test))
print(model.score(X_test, Y_test))


