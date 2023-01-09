import numpy as np

y_predicted = np.array([1, 1, 0, 0, 1])
y_true = np.array([0.30, 0.7, 1, 0, 0.5])
epsilon = 1e-15

# Mean Squared Error
def mse(y_true, y_predicted):
    return np.mean((y_true - y_predicted) ** 2)

print(mse(y_true, y_predicted))

# Mean Absolute Error
def mae(y_true, y_predicted):
    return np.mean(np.abs(y_true - y_predicted))

print(mae(y_true, y_predicted))

# Binary Cross Entropy
def bce(y_true, y_predicted):
    y_predicted_new = [max(i, epsilon) for i in y_predicted]
    y_predicted_new = np.array([min(i, 1 - epsilon) for i in y_predicted_new])
    return np.mean(-(y_true * np.log(y_predicted_new) + (1 - y_true) * np.log(1 - y_predicted_new)))

print(bce(y_true, y_predicted))

# Cross Entropy
def ce(y_true, y_predicted):
    y_predicted_new = [max(i, epsilon) for i in y_predicted]
    y_predicted_new = np.array([min(i, 1 - epsilon) for i in y_predicted_new])
    return np.mean(-y_true * np.log(y_predicted_new))

print(ce(y_true, y_predicted))
