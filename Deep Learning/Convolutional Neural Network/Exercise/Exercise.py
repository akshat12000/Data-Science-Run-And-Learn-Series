import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

(X_train,Y_train),(X_test,Y_test) = keras.datasets.mnist.load_data()

class_names = ['0','1','2','3','4','5','6','7','8','9']

Y_train = Y_train.reshape(-1,)

X_train = X_train/255
X_test = X_test/255

model = keras.Sequential([
    keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train,Y_train,epochs=20)

print(model.evaluate(X_test,Y_test))

Y_pred = model.predict(X_test)

Y_pred_labels = [np.argmax(i) for i in Y_pred]

print("Classification Report: \n", classification_report(Y_test,Y_pred_labels))

