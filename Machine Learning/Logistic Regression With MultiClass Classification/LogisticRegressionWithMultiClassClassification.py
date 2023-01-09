import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sn

digits = load_digits()

plt.gray()
plt.matshow(digits.images[0])

X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target, test_size=0.2)

model = LogisticRegression(solver="lbfgs",class_weight="balanced",max_iter=10000)
model.fit(X_train, Y_train)

print(model.score(X_test, Y_test))
print(model.predict([digits.data[67]]))

Y_Predicted = model.predict(X_test)
cm = confusion_matrix(Y_test, Y_Predicted)

plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()

