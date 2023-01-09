import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

df = pd.read_csv('Churn_Modelling.csv')

df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

df = pd.get_dummies(df, columns=['Geography'])

df['Gender'].replace({"Female":1, "Male":0}, inplace=True)

df = df.dropna()

X = df.drop('Exited', axis='columns')
Y = df['Exited']

def model_builder(X_train, Y_train, X_test, Y_test):
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    print(model.score(X_test, Y_test))
    y_preds = model.predict(X_test)
    y_preds = np.round(y_preds)
    print("Classification Report: \n", classification_report(Y_test, y_preds))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=15, stratify=Y)

print("Normal Sampling")
model_builder(X_train, Y_train, X_test, Y_test)

count_class_0, count_class_1 = df.Exited.value_counts()

df_class_0 = df[df['Exited'] == 0]
df_class_1 = df[df['Exited'] == 1]

# Under Sampling
df_class_0_under = df_class_0.sample(count_class_1)
df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)

X = df_test_under.drop('Exited',axis='columns')
Y = df_test_under['Exited']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=15, stratify=Y)

print("Under Sampling")
model_builder(X_train, Y_train, X_test, Y_test)

# Over Sampling
df_class_1_over = df_class_1.sample(count_class_0, replace=True)
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

X = df_test_over.drop('Exited',axis='columns')
Y = df_test_over['Exited']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=15, stratify=Y)

print("Over Sampling")
model_builder(X_train, Y_train, X_test, Y_test)

# SMOTE
smote = SMOTE(sampling_strategy='minority')

X_sm, Y_sm = smote.fit_resample(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X_sm, Y_sm, test_size=0.2, random_state=15, stratify=Y_sm)

print("SMOTE")
model_builder(X_train, Y_train, X_test, Y_test)

# Ensemble
X = df.drop('Exited', axis='columns')
Y = df['Exited']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=15, stratify=Y)

df1 = X_train.copy()
df1['Exited'] = Y_train

df1_class_0 = df1[df1['Exited'] == 0]
df1_class_1 = df1[df1['Exited'] == 1]

def get_train_batch(df_majority,df_minority,start,end):
    df_train = pd.concat([df_majority[start:end],df_minority],axis=0)
    X_train = df_train.drop('Exited',axis='columns')
    Y_train = df_train['Exited']
    return X_train, Y_train

print("Ensemble")

X_train , Y_train = get_train_batch(df1_class_0,df1_class_1,0,2037)
model_builder(X_train, Y_train, X_test, Y_test)

X_train , Y_train = get_train_batch(df1_class_0,df1_class_1,2037,4074)
model_builder(X_train, Y_train, X_test, Y_test)

X_train , Y_train = get_train_batch(df1_class_0,df1_class_1,4074,6111)
model_builder(X_train, Y_train, X_test, Y_test)

X_train , Y_train = get_train_batch(df1_class_0,df1_class_1,6111,8148)
model_builder(X_train, Y_train, X_test, Y_test)






