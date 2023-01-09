import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Importing the dataset
df = pd.read_csv("insurance_data.csv")

X_train, X_test, Y_train, Y_test = train_test_split(df[['age','affordibility']], df['bought_insurance'], test_size=0.2)

# Scale the data
X_train_scaled = X_train.copy()
X_train_scaled['age'] = X_train_scaled['age']/100

X_test_scaled = X_test.copy()
X_test_scaled['age'] = X_test_scaled['age']/100

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(2,), activation='sigmoid', kernel_initializer='ones', bias_initializer='zeros')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, Y_train, epochs=5000)

# Evaluate the model
print(model.evaluate(X_test_scaled, Y_test))

# Predict the model
print(model.predict(X_test_scaled))

# Plot the model
plt.scatter(X_test_scaled['age'], Y_test, color='red')
plt.show()
plt.scatter(X_test_scaled['age'], [1 if x>=0.5 else 0 for x in model.predict(X_test_scaled)], color='blue')
plt.show()

def log_loss(y_true, y_predicted):
    epsilon = 1e-15
    y_predicted_new = [max(i, epsilon) for i in y_predicted]
    y_predicted_new = [min(i, 1-epsilon) for i in y_predicted_new]
    y_predicted_new = np.array(y_predicted_new)
    return -np.mean(y_true*np.log(y_predicted_new)+ (1-y_true)*np.log(1-y_predicted_new))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def gradient_descent(age,affordability,Y_true,epochs,loss_threshold):
    w1 = w2 = 1
    bias = 0
    rate = 0.5
    n = len(age)
    for i in range(epochs):
        weighted_sum = w1*age + w2*affordability + bias
        Y_predicted = sigmoid(weighted_sum)
        loss = log_loss(Y_true,Y_predicted)
        w1d = (1/n)*np.dot(np.transpose(age),(Y_predicted-Y_true))
        w2d = (1/n)*np.dot(np.transpose(affordability),(Y_predicted-Y_true))
        bias_d = np.mean(Y_predicted-Y_true)
        w1 = w1 - rate*w1d
        w2 = w2 - rate*w2d
        bias = bias - rate*bias_d
        print(f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')
        if loss<=loss_threshold:
            break
    return w1,w2,bias

print(gradient_descent(X_train_scaled['age'],X_train_scaled['affordibility'],Y_train,10000,0.4631))
