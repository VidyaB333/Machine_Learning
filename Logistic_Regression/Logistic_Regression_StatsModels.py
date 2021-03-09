import math
import numpy as np
import pandas as pd
import statsmodels
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

df = pd.read_csv(
    'C:\\Users\\vidya\\OneDrive\\Desktop\\Python_coding_practice_Datasets\\PythonDataSets\\Logistic_Regression\\User_Data_purchase.csv')
print(df.shape)
print(df.columns)
X = df[df.columns[0:-1]]

X.drop(['Gender'], axis=1, inplace=True)
y = np.array(df['Purchased']).reshape(-1, 1)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=23)
X_train = sm.add_constant(X_train)

logistic = sm.Logit(y_train, X_train)
model = logistic.fit()
print(model.summary())

# Iteration2
X.drop(['User ID'], axis=1, inplace=True)
y = np.array(df['Purchased']).reshape(-1, 1)
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=23)
X_train = sm.add_constant(X_train)

X_test = sm.add_constant(X_test)
logistic = sm.Logit(y_train, X_train)
model = logistic.fit()
print(model.summary())
y_pred = model.predict(X_test)
print(y_pred)
"""
for i in range(len(y_pred)):
    a = y_pred[i]
    a = 1 if y_pred[i] >= 0.5 else 0
    y_pred[i] = a
"""
#Using round function as well, we can use below functionality

y_pred = y_pred.apply(lambda x: 1 if x >= 0.5 else 0)
print(y_pred)
from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_pred, y_test))

print()
print(accuracy_score(y_pred,  y_pred))
print(model.c)