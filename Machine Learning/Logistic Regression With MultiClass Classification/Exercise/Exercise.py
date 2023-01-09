import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

iris = load_iris()
# print(iris)

X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.2)

model = LogisticRegression(solver="lbfgs",class_weight="balanced",max_iter=10000)
model.fit(X_train, Y_train)

print("Training Score: ", model.score(X_train, Y_train))
print("Testing Score: ", model.score(X_test, Y_test))

print("Predicted Value: ", model.predict(X_test))
print("Actual Value: ", Y_test)

plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test)
plt.show()

