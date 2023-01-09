import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing

# Read the data
df = pd.read_csv('homeprices_banglore.csv')

sx = preprocessing.MinMaxScaler()
sy = preprocessing.MinMaxScaler()

# Scale the data
scaled_x = sx.fit_transform(df.drop('price', axis='columns'))
scaled_y = sy.fit_transform(df['price'].values.reshape(df.shape[0], 1))

# Create the model
def batch_gradient_descent(X,Y_true,epochs,learning_rate=0.01):
    number_of_features = X.shape[1]
    w = np.ones(shape=(number_of_features))
    b = 0
    total_samples = X.shape[0]
    cost_list = []
    epoch_list = []
    for i in range(epochs):
        Y_predicted = np.dot(w,X.T) + b
        w_grad = -(2/total_samples)*(X.T.dot(Y_true - Y_predicted))
        b_grad = -(2/total_samples)*np.sum(Y_true - Y_predicted)
        w = w - learning_rate*w_grad
        b = b - learning_rate*b_grad
        cost = np.mean(np.square(Y_true - Y_predicted))
        if i % 10 == 0:
            cost_list.append(cost)
            epoch_list.append(i)
        print("Epoch {}: w = {}, b = {}, cost = {}, ".format(i,w,b,cost))
    return w,b,cost,cost_list,epoch_list

w,b,cost,cost_list,epoch_list = batch_gradient_descent(scaled_x,scaled_y.reshape(scaled_y.shape[0],),500)

# Plot the cost
plt.plot(epoch_list,cost_list)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.show()

def predict(area,bedrooms,w,b):
    scaled_x = sx.transform([[area,bedrooms]])[0]
    scaled_price = w[0]*scaled_x[0] + w[1]*scaled_x[1] + b
    return sy.inverse_transform([[scaled_price]])[0][0]

print(predict(2600,4,w,b))

def stochastic_gradient_descent(X,Y_true,epochs,learning_rate=0.01):
    number_of_features = X.shape[1]
    w = np.ones(shape=(number_of_features))
    b = 0
    total_samples = X.shape[0]
    cost_list = []
    epoch_list = []
    for i in range(epochs):
        random_index = np.random.randint(total_samples)
        sample_x = X[random_index]
        sample_y = Y_true[random_index]
        Y_predicted = np.dot(w,sample_x.T) + b
        w_grad = -(2/total_samples)*(sample_x.T.dot(sample_y - Y_predicted))
        b_grad = -(2/total_samples)*np.sum(sample_y - Y_predicted)
        w = w - learning_rate*w_grad
        b = b - learning_rate*b_grad
        cost = np.mean(np.square(Y_true - Y_predicted))
        if i % 10 == 0:
            cost_list.append(cost)
            epoch_list.append(i)
        print("Epoch {}: w = {}, b = {}, cost = {}, ".format(i,w,b,cost))
    return w,b,cost,cost_list,epoch_list

w,b,cost,cost_list,epoch_list = stochastic_gradient_descent(scaled_x,scaled_y.reshape(scaled_y.shape[0],),500)

# Plot the cost
plt.plot(epoch_list,cost_list)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.show()

def predict_new(area,bedrooms,w,b):
    scaled_x = sx.transform([[area,bedrooms]])[0]
    scaled_price = w[0]*scaled_x[0] + w[1]*scaled_x[1] + b
    return sy.inverse_transform([[scaled_price]])[0][0]

print(predict_new(2600,4,w,b))