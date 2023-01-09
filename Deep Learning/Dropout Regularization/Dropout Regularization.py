import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('sonar_dataset.csv', header=None)

X = df.drop(60,axis="columns")
Y = df[60]
Y = pd.get_dummies(Y, drop_first=True)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1)

model = keras.Sequential([
    keras.layers.Dense(60, input_shape=(60,), activation='relu'),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=100, batch_size=8)

print(model.evaluate(X_test, Y_test))

Y_pred = model.predict(X_test).reshape(-1)
Y_pred = np.round(Y_pred)

print(classification_report(Y_test, Y_pred))

cm = confusion_matrix(Y_test, Y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()

model2 = keras.Sequential([
    keras.layers.Dense(60, input_shape=(60,), activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model2.fit(X_train, Y_train, epochs=100, batch_size=8)

print(model2.evaluate(X_test, Y_test))

Y_pred2 = model2.predict(X_test).reshape(-1)

Y_pred2 = np.round(Y_pred2)

print(classification_report(Y_test, Y_pred2))

cm2 = confusion_matrix(Y_test, Y_pred2)
sns.heatmap(cm2, annot=True, fmt='d')
plt.show()




