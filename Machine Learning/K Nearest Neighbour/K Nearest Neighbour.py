import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

df0 = df[df.target==0]
df1 = df[df.target==1]
df2 = df[df.target==2]

df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])

X = df.drop(['target', 'flower_name'], axis='columns')
Y = df.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, Y_train)

print(model.score(X_test, Y_test))

Y_predicted = model.predict(X_test)
print(classification_report(Y_test, Y_predicted))



