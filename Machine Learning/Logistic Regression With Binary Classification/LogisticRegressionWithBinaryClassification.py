import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('insurance_data.csv')
plt.scatter(df.age, df.bought_insurance, marker='+', color='red')
plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(df[['age']], df.bought_insurance, test_size=0.1)

model = LogisticRegression()
model.fit(X_train, Y_train)

print(model.predict(X_test))
print(model.predict_proba(X_test))
print(model.score(X_test, Y_test))