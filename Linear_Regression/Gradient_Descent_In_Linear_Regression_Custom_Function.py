import pandas as pd
import numpy as np
df = pd.read_csv(
    'C:\\Users\\vidya\OneDrive\\Desktop\\Python_coding_practice_Datasets\\PythonDataSets\\Logistic_Regression\\archive\\wineQualityReds.csv',
    usecols=['alcohol', 'quality'])
print(df.columns)
n = df.shape[0]
print(n)
X = df['alcohol']
print(type(X[0]))
y = df['quality']
# Gradient Descent ON SIMPLE LINEAR Regression
print(np.random.random(1))
print(np.random.random(1))
print(X.shape)
#alpha_s = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]


def Gradient_descent( X, y,max_iter, alpha, batch_size, coverserged_th):
    print(max_iter)
    print(alpha)
    print(batch_size)
    print(coverserged_th)
    Q0 = 0.03085956
    Q1 = 0.75802637
    iter_ =0
    converged = False

    while not converged:
        MSE = (sum(  ((Q1*X[i] + Q0 - y[i]) ** 2 for i in range(batch_size)))/batch_size)
        delta_Q1 = (-2 / batch_size) * (sum((X[i]) * (y[i] - (X[i] * Q1 + Q0)) for i in range(batch_size)))
        delta_Q0 = (2 / batch_size) * sum(2 * (y[i] - (X[i] * Q1 + Q0)) for i in range(batch_size))

        Q1 = Q1 - alpha * delta_Q1
        Q0 = Q0 - alpha * delta_Q0

        MSE_new = (sum(  ((Q1*X[i] + Q0 - y[i]) ** 2 for i in range(batch_size))) / batch_size)

        if abs((MSE_new - MSE)) <= coverserged_th:
            print('Cost Function Converged, ', iter_)
            converged= True
        iter_ +=1

        if iter_>= max_iter:
            print('Max iteration exhausted')
            converged = True
    print('MSEs: ', MSE, MSE_new)

    print(round(Q1,4), round(Q0,4))
    return round(Q1,4), round(Q0,4)

slope, intercept = Gradient_descent(X, y,max_iter=1500000, alpha=0.00003,  batch_size= 32, coverserged_th=1e-8)
print(slope, intercept)