import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

digits = load_digits()

df = pd.DataFrame(digits.data)
df['target'] = digits.target

X = df.drop(['target'], axis='columns')
Y = df.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = RandomForestClassifier(n_estimators=50)
model.fit(X_train, Y_train)

print(model.score(X_test, Y_test))

Y_Predicted = model.predict(X_test)

cm = confusion_matrix(Y_test, Y_Predicted)

plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()



