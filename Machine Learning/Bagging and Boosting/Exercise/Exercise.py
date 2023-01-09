import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

df = pd.read_csv("heart.csv")

# Removing outliers based on z-score
df = df[abs((df["Age"] - df["Age"].mean()) / df["Age"].std()) < 3]
df = df[abs((df["RestingBP"] - df["RestingBP"].mean()) / df["RestingBP"].std()) < 3]
df = df[abs((df["Cholesterol"] - df["Cholesterol"].mean()) / df["Cholesterol"].std()) < 3]
df = df[abs((df["FastingBS"] - df["FastingBS"].mean()) / df["FastingBS"].std()) < 3]
df = df[abs((df["RestingECG"] - df["RestingECG"].mean()) / df["RestingECG"].std()) < 3]
df = df[abs((df["MaxHR"] - df["MaxHR"].mean()) / df["MaxHR"].std()) < 3]
df = df[abs((df["ExerciseAngina"] - df["ExerciseAngina"].mean()) / df["ExerciseAngina"].std()) < 3]
df = df[abs((df["Oldpeak"] - df["Oldpeak"].mean()) / df["Oldpeak"].std()) < 3]
df = df[abs((df["ST_Slope"] - df["ST_Slope"].mean()) / df["ST_Slope"].std()) < 3]

X = df.drop("HeartDisease",axis=1)
Y = df["HeartDisease"]

scaler = StandardScaler()
X_Scaled = scaler.fit_transform(X)

X_train,X_test,Y_train,Y_test = train_test_split(X_Scaled,Y,test_size=0.2,random_state=30)

svm = SVC()
svm.fit(X_train,Y_train)
print(svm.score(X_test,Y_test))

dt = DecisionTreeClassifier()
dt.fit(X_train,Y_train)
print(dt.score(X_test,Y_test))

bagging = BaggingClassifier(DecisionTreeClassifier(),n_estimators=100,max_samples=0.8,oob_score=True,random_state=0)
bagging.fit(X_train,Y_train)
print(bagging.oob_score_)

bagging = BaggingClassifier(SVC(),n_estimators=100,max_samples=0.8,oob_score=True,random_state=0)
bagging.fit(X_train,Y_train)
print(bagging.oob_score_)

