import pandas as pd
import numpy as np
df = pd.read_csv(
    'C:\\Users\\vidya\OneDrive\\Desktop\\Python_coding_practice_Datasets\\PythonDataSets\\Logistic_Regression\\archive\\wineQualityReds.csv')

y = np.array(df['quality'])
y = y.reshape(-1,1)
print(y.shape, y.ndim)
x= df[df.columns[1:-1]]
print(x.shape, x.ndim)
x = np.array(x)

ones = np.ones((1599, 12))

print('printing one')
print(ones)

print(ones.shape)
print()
for i in range(11):
    ones[:, i+1]= x[: , i]
print(ones)

coefficients = np.linalg.inv(ones.T.dot(ones)).dot(ones.T).dot(y)
print(coefficients)
for i , j in zip(df.columns, coefficients):
    print(i, j)