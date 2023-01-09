import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split

def log_loss(y_true, y_predicted):
    epsilon = 1e-15
    y_predicted_new = [max(i, epsilon) for i in y_predicted]
    y_predicted_new = [min(i, 1-epsilon) for i in y_predicted_new]
    y_predicted_new = np.array(y_predicted_new)
    return -np.mean(y_true*np.log(y_predicted_new)+ (1-y_true)*np.log(1-y_predicted_new))

def sigmoid(x):
    return 1/(1+np.exp(-x))

class myNN:
    def __init__(self):
        self.w1 = 1
        self.w2 = 1
        self.bias = 0

    def fit(self,X,Y,epochs,loss_threshold):
        self.w1,self.w2,self.bias = self.gradient_descent(X['age'],X['affordibility'],Y,epochs,loss_threshold)

    def predict(self,X):
        weighted_sum = self.w1*X['age'] + self.w2*X['affordibility'] + self.bias
        return sigmoid(weighted_sum)
    
    def evaluate(self,X,Y):
        Y_predicted = self.predict(X)
        Y_predicted_cls = [1 if i>0.5 else 0 for i in Y_predicted]
        accuracy = np.sum(Y_predicted_cls == Y)/len(Y)
        return accuracy
    
    def gradient_descent(self,age,affordability,Y_true,epochs,loss_threshold):
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
            if i%50==0:
                print(f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')
            if loss<=loss_threshold:
                print(f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')
                break
        return w1,w2,bias

# Importing the dataset
df = pd.read_csv("insurance_data.csv")

X_train, X_test, Y_train, Y_test = train_test_split(df[['age','affordibility']], df['bought_insurance'], test_size=0.2)

# Scale the data
X_train_scaled = X_train.copy()
X_train_scaled['age'] = X_train_scaled['age']/100

X_test_scaled = X_test.copy()
X_test_scaled['age'] = X_test_scaled['age']/100

customModel = myNN()
customModel.fit(X_train_scaled,Y_train,10000,0.4631)

print(customModel.evaluate(X_test_scaled,Y_test))



