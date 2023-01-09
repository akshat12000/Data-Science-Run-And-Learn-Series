import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import numpy as np

# Load the dataset
(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

plt.matshow(X_train[0])
plt.show()

X_Train=X_train.reshape(len(X_train), 28*28)

X_Test=X_test.reshape(len(X_test), 28*28)

# Normalize the data
X_Train=X_Train/255
X_Test=X_Test/255

# Build the model
model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_Train, Y_train, epochs=5)

print(model.evaluate(X_Test, Y_test))

