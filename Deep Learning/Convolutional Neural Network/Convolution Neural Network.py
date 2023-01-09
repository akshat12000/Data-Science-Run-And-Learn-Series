import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

(X_train,Y_train),(X_test,Y_test) = keras.datasets.cifar10.load_data()

class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

Y_train = Y_train.reshape(-1,)

def plot_sample(X,Y,index):
    plt.figure(figsize=(15,2))
    plt.imshow(X[index])
    plt.xlabel(class_names[Y[index]])
    plt.show()

plot_sample(X_train,Y_train,0)

X_train = X_train/255
X_test = X_test/255

# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(32,32,3)),
#     keras.layers.Dense(3000,activation='relu'),
#     keras.layers.Dense(1000,activation='relu'),
#     keras.layers.Dense(10,activation='sigmoid')
# ])

# model.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(X_train,Y_train,epochs=5)

# print(model.evaluate(X_test,Y_test))

# Y_pred = model.predict(X_test)
# Y_pred_labels = [np.argmax(i) for i in Y_pred]

# print("Classification Report: \n", classification_report(Y_test,Y_pred_labels))

model2 = keras.Sequential([
    keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(32,32,3)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])

model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model2.fit(X_train,Y_train,epochs=20)

print(model2.evaluate(X_test,Y_test))

Y_pred = model2.predict(X_test)

Y_pred_labels = [np.argmax(i) for i in Y_pred]

print("Classification Report: \n", classification_report(Y_test,Y_pred_labels))



