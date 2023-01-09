import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

(X_train,Y_train),(X_test,Y_test) = keras.datasets.cifar10.load_data()

classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

X_train_scaled = X_train/255
X_test_scaled = X_test/255

Y_train_categorical = keras.utils.to_categorical(Y_train,10)
Y_test_categorical = keras.utils.to_categorical(Y_test,10)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(32,32,3)),
    keras.layers.Dense(3000,activation='relu'),
    keras.layers.Dense(1000,activation='relu'),
    keras.layers.Dense(10,activation='sigmoid')
])

model.compile(optimizer='SGD',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train_scaled,Y_train_categorical,epochs=50)

model.evaluate(X_test_scaled,Y_test_categorical)



