import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

df = pd.read_csv('customer_churn.csv')

df.drop('customerID', axis=1, inplace=True)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df = df[df.TotalCharges!=' ']
df.TotalCharges = pd.to_numeric(df.TotalCharges)

tenure_churn_no = df[df.Churn == 'No'].tenure
tenure_churn_yes = df[df.Churn == 'Yes'].tenure

plt.xlabel("Tenure")
plt.ylabel("Number of Customers")
plt.title("Customer Churn Prediction Visualization")
plt.hist([tenure_churn_yes, tenure_churn_no], color = ['green','red'], label = ['Churn=Yes','Churn=No'])
plt.legend()
plt.show()

def print_unique_col_values(df):
    for column in df:
        if df[column].dtypes == 'object':
            print(f'{column}: {df[column].unique()}')

print_unique_col_values(df)

df.replace('No internet service', 'No', inplace=True)
df.replace('No phone service', 'No', inplace=True)

yes_no_columns = ["Partner", "Dependents", "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling", "Churn"]

for col in yes_no_columns:
    df[col].replace({'Yes':1, "No":0}, inplace=True)

df['gender'].replace({"Female":1, "Male":0}, inplace=True)

df = pd.get_dummies(data=df, columns=['InternetService', 'Contract', 'PaymentMethod'])

cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']

scaler = MinMaxScaler()

df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

X = df.drop('Churn', axis='columns')
Y = df['Churn']

print(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

model = keras.Sequential([
    keras.layers.Dense(26, input_shape=(26,), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# X_train = np.asarray(X_train).astype(float)
# Y_train = np.asarray(Y_train).astype(float)
# X_test = np.asarray(X_test).astype(float)
# Y_test = np.asarray(Y_test).astype(float)

model.fit(X_train, Y_train, epochs=100)

print(model.evaluate(X_test, Y_test))



