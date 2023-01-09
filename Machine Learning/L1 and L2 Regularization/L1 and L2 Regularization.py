import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.model_selection import train_test_split

# Importing the dataset
dataset = pd.read_csv('Melbourne_housing_FULL.csv')
cols_to_use =['Suburb','Rooms','Type','Method','SellerG','Regionname','Propertycount','Distance','CouncilArea','Bedroom2','Bathroom','Car','Landsize','BuildingArea','Price']
dataset = dataset[cols_to_use]

# Handling missing values
cols_to_fill_zero = ['Propertycount','Distance','Bedroom2','Bathroom','Car']
dataset[cols_to_fill_zero] = dataset[cols_to_fill_zero].fillna(0)

cols_to_fill_mean = ['Landsize','BuildingArea']
dataset[cols_to_fill_mean] = dataset[cols_to_fill_mean].fillna(dataset[cols_to_fill_mean].mean())

dataset.dropna(inplace=True)

# Encoding categorical data
dataset = pd.get_dummies(dataset, drop_first=True)

X=dataset.drop('Price',axis=1)
Y=dataset['Price']

# Splitting the dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 2)

# Fitting Linear Regression to the dataset
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

print(regressor.score(X_test, Y_test))
print(regressor.score(X_train, Y_train))

# Fitting Lasso Regression to the dataset (L1 Regularization)
lasso_reg = Lasso(alpha=50, max_iter=100, tol=0.1)
lasso_reg.fit(X_train, Y_train)

print(lasso_reg.score(X_test, Y_test))
print(lasso_reg.score(X_train, Y_train))

# Fitting Ridge Regression to the dataset (L2 Regularization)
ridge_reg = Ridge(alpha=50, max_iter=100, tol=0.1)
ridge_reg.fit(X_train, Y_train)

print(ridge_reg.score(X_test, Y_test))
print(ridge_reg.score(X_train, Y_train))

