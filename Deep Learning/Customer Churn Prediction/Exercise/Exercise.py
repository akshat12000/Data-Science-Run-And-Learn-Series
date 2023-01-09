import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

df = pd.read_csv('Churn_Modelling.csv')

df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

df = pd.get_dummies(df, columns=['Geography'])

df['Gender'].replace({"Female":1, "Male":0}, inplace=True)

df = df.dropna()

X = df.drop('Exited', axis='columns')
Y = df['Exited']

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

model = keras.Sequential([
    keras.layers.Dense(20, input_shape=(12,), activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=100)

print(model.evaluate(X_test, Y_test))

Y_pred = model.predict(X_test)

Y_pred = [1 if y>0.5 else 0 for y in Y_pred]

cm = confusion_matrix(Y_test, Y_pred)

print(cm)

print(classification_report(Y_test, Y_pred))

sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()





