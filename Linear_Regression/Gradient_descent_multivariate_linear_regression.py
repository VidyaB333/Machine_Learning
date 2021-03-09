import pandas as pd
import numpy as np




#RuntimeWarning: overflow encountered in double_scalars---> Getting failed with this error need to check the dtyaes in numpy

df = pd.read_csv(
    'C:\\Users\\vidya\OneDrive\\Desktop\\Python_coding_practice_Datasets\\PythonDataSets\\Logistic_Regression\\archive\\wineQualityReds.csv', )

col = df.columns
col = [i.replace('.', '_') for i in col]
df.columns = col
X = df[df.columns[1: -1]]
print(X.shape)
y = df[df.columns[-1]]


def Gradient_descent(X, y, max_iter, alpha, batch_size, coverserged_th):
    print(max_iter)
    print(alpha)
    print(batch_size)
    print(coverserged_th)
    Q0 = Q1 = Q2 = Q3 = Q4 = Q5 = Q6 = Q7 = Q8 = Q9 = Q10 = Q11 = round(float(np.random.random(1)), 5)
    print(Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, Q10, Q11)
    iter_ = 0
    converged = False

    while not converged:
        MSE = (sum(((
                            (Q1 * X.iloc[i, 0] + Q2 * X.iloc[i, 1] + Q3 * X.iloc[i, 2] + Q4 * X.iloc[i, 3] + Q5 *
                             X.iloc[i, 4] + Q6 * X.iloc[i, 5] + Q7 * X.iloc[i, 6] + Q8 * X.iloc[i, 7]
                             + Q8 * X.iloc[i, 7] + Q9 * X.iloc[i, 8] + Q10 * X.iloc[i, 9] + Q11 * X.iloc[i, 10])
                            - y[i]) ** 2 for i in range(batch_size))) / batch_size)

        delta_Q1 = (-2 / batch_size) * (
            sum((X.iloc[i, 0]) * (y[i] - (X.iloc[i, 0] * Q1 + Q0)) for i in range(batch_size)))
        delta_Q2 = (-2 / batch_size) * (
            sum((X.iloc[i, 1]) * (y[i] - (X.iloc[i, 1] * Q2 + Q0)) for i in range(batch_size)))
        delta_Q3 = (-2 / batch_size) * (
            sum((X.iloc[i, 2]) * (y[i] - (X.iloc[i, 2] * Q3 + Q0)) for i in range(batch_size)))
        delta_Q4 = (-2 / batch_size) * (
            sum((X.iloc[i, 3]) * (y[i] - (X.iloc[i, 3] * Q4 + Q0)) for i in range(batch_size)))
        delta_Q5 = (-2 / batch_size) * (
            sum((X.iloc[i, 4]) * (y[i] - (X.iloc[i, 4] * Q5 + Q0)) for i in range(batch_size)))
        delta_Q6 = (-2 / batch_size) * (
            sum((X.iloc[i, 5]) * (y[i] - (X.iloc[i, 5] * Q6 + Q0)) for i in range(batch_size)))
        delta_Q7 = (-2 / batch_size) * (
            sum((X.iloc[i, 6]) * (y[i] - (X.iloc[i, 6] * Q7 + Q0)) for i in range(batch_size)))
        delta_Q8 = (-2 / batch_size) * (
            sum((X.iloc[i, 7]) * (y[i] - (X.iloc[i, 7] * Q8 + Q0)) for i in range(batch_size)))
        delta_Q9 = (-2 / batch_size) * (
            sum((X.iloc[i, 8]) * (y[i] - (X.iloc[i, 8] * Q9 + Q0)) for i in range(batch_size)))
        delta_Q10 = (-2 / batch_size) * (
            sum((X.iloc[i, 9]) * (y[i] - (X.iloc[i, 9] * Q10 + Q0)) for i in range(batch_size)))
        delta_Q11 = (-2 / batch_size) * (
            sum((X.iloc[i, 10]) * (y[i] - (X.iloc[i, 10] * Q11 + Q0)) for i in range(batch_size)))
        # delta_Q0 = (2 / batch_size) * sum(2 * (y[i] - (X[i] * Q1 + Q0)) for i in range(batch_size))

        Q1 = Q1 - alpha * delta_Q1
        Q2 = Q1 - alpha * delta_Q2
        Q3 = Q1 - alpha * delta_Q3
        Q4 = Q1 - alpha * delta_Q4
        Q5 = Q1 - alpha * delta_Q5
        Q6 = Q1 - alpha * delta_Q6
        Q7 = Q1 - alpha * delta_Q7
        Q8 = Q1 - alpha * delta_Q8
        Q9 = Q1 - alpha * delta_Q9
        Q10 = Q1 - alpha * delta_Q10
        Q11 = Q1 - alpha * delta_Q11
        # Q0 = Q0 - alpha * delta_Q0

        MSE_new = (sum(((
                                (Q1 * X.iloc[i, 0] + Q2 * X.iloc[i, 1] + Q3 * X.iloc[i, 2] + Q4 * X.iloc[i, 3] + Q5 *
                                 X.iloc[i, 4] + Q6 * X.iloc[i, 5] + Q7 * X.iloc[i, 6] + Q8 * X.iloc[i, 7]
                                 + Q8 * X.iloc[i, 7] + Q9 * X.iloc[i, 8] + Q10 * X.iloc[i, 9] + Q11 * X.iloc[i, 10])
                                - y[i]) ** 2 for i in range(batch_size))) / batch_size)

        if abs((MSE_new - MSE)) <= coverserged_th:
            print('Cost Function Converged, ', iter_)
            converged = True
        iter_ += 1

        if iter_ >= max_iter:
            print('Max iteration exhausted')
            converged = True
    print('MSEs: ', MSE, MSE_new)

    print(round(Q1, 4), round(Q2, 4), round(Q3, 4), round(Q4, 4), round(Q5, 4), round(Q6, 4),
          round(Q7, 4),
          round(Q8, 4), round(Q9, 4), round(Q10, 4), round(Q11, 4))
    return [ round(Q1, 4), round(Q2, 4), round(Q3, 4), round(Q4, 4), round(Q5, 4), round(Q6, 4),
            round(Q7, 4),
            round(Q8, 4), round(Q9, 4), round(Q10, 4), round(Q11, 4)]


slope = Gradient_descent(X, y, max_iter=1500000, alpha=0.0003, batch_size=32, coverserged_th=1e-8)
print(slope)
