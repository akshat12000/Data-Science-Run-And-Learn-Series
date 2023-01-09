import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree

df = pd.read_csv("titanic.csv")
# print(df)

df = df.drop(['PassengerId','Name','SibSp','Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns')

inputs_n = df.drop(['Survived'], axis='columns')
target = df['Survived']

le_sex = LabelEncoder()

inputs_n['Sex_n'] = le_sex.fit_transform(inputs_n['Sex'])
inputs_n = inputs_n.drop(['Sex'],axis='columns')

inputs_n['Age'] = inputs_n['Age'].fillna(inputs_n['Age'].mean())

X_train, X_test, Y_train, Y_test = train_test_split(inputs_n, target, test_size=0.2)

model = tree.DecisionTreeClassifier()
model.fit(X_train, Y_train)

print(model.score(X_test, Y_test))



