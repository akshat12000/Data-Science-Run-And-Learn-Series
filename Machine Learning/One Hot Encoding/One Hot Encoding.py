import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv("homeprices.csv")

dummy = pd.get_dummies(df.town)
merged = pd.concat([df, dummy], axis='columns')

final = merged.drop(['town', 'west windsor'], axis='columns')
print(final)

model = linear_model.LinearRegression()

x = final.drop('price', axis='columns')
y = final.price

reg = model.fit(x, y)

print(reg.predict([[2800, 0, 1]]))
print(reg.predict([[3400, 0, 0]]))

print(reg.score(x, y))

# Hot Encoding using Scikit Learn
le = LabelEncoder()
dfle = df
dfle.town = le.fit_transform(dfle.town)

X=dfle[['town', 'area']].values
Y=dfle.price

ct=ColumnTransformer([("town", OneHotEncoder(), [0])], remainder='passthrough')
X = ct.fit_transform(X)
X=X[:,1:]

new_model = linear_model.LinearRegression()
new_model.fit(X, Y)

print(new_model.predict([[1, 0, 2800]]))
print(new_model.predict([[0, 1, 3400]]))

print(new_model.score(X, Y))

