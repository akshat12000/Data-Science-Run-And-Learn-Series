import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])

X = df.drop(['target', 'flower_name'], axis='columns')
Y = df.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = SVC()
model.fit(X_train, Y_train)
print(model.score(X_test, Y_test))


