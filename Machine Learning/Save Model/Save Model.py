import pandas as pd
import numpy as np
import pickle
from sklearn import linear_model
import joblib

df = pd.read_csv("homeprices.csv")

model = linear_model.LinearRegression()
model.fit(df[['area']],df.price)

print(model.coef_)
print(model.intercept_)

print(model.predict([[3300]]))

with open('model_pickle','wb') as f:
    pickle.dump(model,f)

joblib.dump(model,'model_joblib')