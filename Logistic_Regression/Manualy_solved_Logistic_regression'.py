import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import cmath

#x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

#For finding perfrct parameters, need to find cost function


#Normalization of features
def standardization(X):
    sd = np.std(X)
    mu = np.mean(X)
    print(mu, sd)
    for i in range(len(X)):
        X[i]  = np.round((X[i] -mu)/sd , 4)
    return X

def logistic_func(beta, X):
    print('Logistic Regression Activation function')
    #logistic(sigmoid) function
    return 1.0/(1 + np.exp(-np.dot(X, beta.T)))

def cost_function(X, y, beta):
    print('Cost function Of model')
    step1 = np.log(logistic_func(beta, X))
    step2 = np.log(1 - logistic_func(beta, X))

    J = -y* step1 - (1-y)*step2
    print(J)
    return np.mean(J)

def gradient_descent(X, y, beta, leaning_rate = 0.1, converge_change=.001):
    print('Gradient descent')
    iter= 1
    change_cost =1
    while change_cost >converge_change:
        old_cost = cost_function(X, y, beta)
        step1 = leaning_rate*((logistic_func(beta, X) -y)*X)
        print(step1)
        beta = beta-(leaning_rate*step1)
        print(beta)
        new_cost = cost_function(X, y, beta)
        change_cost = old_cost-new_cost
    return beta, iter

if __name__ == "__main__":
    df = pd.read_csv(
        'C:\\Users\\vidya\\OneDrive\\Desktop\\Python_coding_practice_Datasets\\PythonDataSets\\Logistic_Regression\\User_Data_purchase.csv')
    print(df.shape)
    x = df['Age']
    y = df['Purchased']
    n = len(x)
    x = np.array(x, dtype='float').reshape(-1, 1)

    y = np.array(y).reshape(-1, 1)
    # print('datatype :', y.dtype)

    X = standardization(x)
    print('AFTER :', X)
    #print(X.shape)
    #input_features = np.hstack((np.ones(X.shape[0]), X))
    #print(input_features.shape)
    X = np.hstack((np.matrix(np.ones(X.shape[0])).T, X))
    #print(input_features[0:10, :])
    beta = np.zeros(X.shape[1])
    print(beta)
    print(X)

    print(cost_function(X, y, beta))
    beta, iter = gradient_descent(X, y, beta, leaning_rate = 0.1)
    print('coefficients: ', beta)
    print('Iterations: ', iter)
