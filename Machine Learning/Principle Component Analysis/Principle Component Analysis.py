import pandas as pd
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

digits = load_digits()

df = pd.DataFrame(digits.data,columns=digits.feature_names)

X=df
Y=digits.target

scaler = StandardScaler()
X_Scaled=scaler.fit_transform(X)

X_train,X_test,Y_train,Y_test = train_test_split(X_Scaled,Y,test_size=0.2,random_state=30)

model = LogisticRegression()
model.fit(X_train,Y_train)

print(model.score(X_test,Y_test))

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_Scaled)

X_train,X_test,Y_train,Y_test = train_test_split(X_pca,Y,test_size=0.2,random_state=30)

model2 = LogisticRegression()
model2.fit(X_train,Y_train)

print(model2.score(X_test,Y_test))




