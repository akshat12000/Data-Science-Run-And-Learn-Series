import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("diabetes.csv")

X = df.drop("Outcome", axis=1)
Y = df["Outcome"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=10)

print(cross_val_score(DecisionTreeClassifier(), X,Y, cv=5).mean())

bagging = BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, max_samples=0.8,oob_score=True, random_state=0)
bagging.fit(X_train, Y_train)

print(bagging.oob_score_)

print(cross_val_score(bagging, X,Y, cv=5).mean())








