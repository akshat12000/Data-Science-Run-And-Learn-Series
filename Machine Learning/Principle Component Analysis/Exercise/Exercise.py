import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

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

# print(df)

X = df.drop("HeartDisease",axis=1)
Y = df["HeartDisease"]

scaler = StandardScaler()
X_Scaled = scaler.fit_transform(X)

X_train,X_test,Y_train,Y_test = train_test_split(X_Scaled,Y,test_size=0.2,random_state=30)

model = LogisticRegression()
model.fit(X_train,Y_train)

print(model.score(X_test,Y_test))

model = RandomForestClassifier(n_estimators=10)
model.fit(X_train,Y_train)

print(model.score(X_test,Y_test))

model = SVC()
model.fit(X_train,Y_train)

print(model.score(X_test,Y_test))

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_Scaled)

X_train,X_test,Y_train,Y_test = train_test_split(X_pca,Y,test_size=0.2,random_state=30)

model2 = LogisticRegression()
model2.fit(X_train,Y_train)

print(model2.score(X_test,Y_test))

model2 = RandomForestClassifier(n_estimators=10)
model2.fit(X_train,Y_train)

print(model2.score(X_test,Y_test))

model2 = SVC()
model2.fit(X_train,Y_train)

print(model2.score(X_test,Y_test))




