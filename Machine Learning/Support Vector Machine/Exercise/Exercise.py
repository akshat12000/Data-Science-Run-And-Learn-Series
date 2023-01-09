import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

digits = load_digits()

df = pd.DataFrame(digits.data)
df['target'] = digits.target

X = df.drop(['target'], axis='columns')
Y = df.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = SVC()
model.fit(X_train, Y_train)
print(model.score(X_test, Y_test))

