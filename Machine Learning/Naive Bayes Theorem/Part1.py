import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Read the data
df = pd.read_csv('titanic.csv')
df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns', inplace=True)

target = df.Survived
inputs = df.drop('Survived', axis='columns')

dummies = pd.get_dummies(inputs.Sex)

inputs = pd.concat([inputs, dummies], axis='columns')
inputs.drop('Sex', axis='columns', inplace=True)

inputs.Age = inputs.Age.fillna(inputs.Age.mean())

X_train, X_test, Y_train, Y_test = train_test_split(inputs, target, test_size=0.2)

# Naive Bayes Theorem
model = GaussianNB()
model.fit(X_train, Y_train)
print(model.score(X_test, Y_test))





