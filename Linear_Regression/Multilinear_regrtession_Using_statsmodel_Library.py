import col as col
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

df = pd.read_csv(
    'C:\\Users\\vidya\OneDrive\\Desktop\\Python_coding_practice_Datasets\\PythonDataSets\\Logistic_Regression\\archive\\wineQualityReds.csv', )

col = df.columns
col = [i.replace('.', '_') for i in col]
df.columns = col
x = df[df.columns[1: -1]]
print(x.shape)
y = df[df.columns[-1]]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=44)
print()

x_train_new = sm.add_constant(x_train)
x_test_new = sm.add_constant(x_test)

full_mod = sm.OLS(y_train, x_train_new)
full_res = full_mod.fit()

print(full_res.summary())

print('Test r2', r2_score(y_test, full_res.predict(x_test_new)))

print('Variable Infaltion Factor')

col = list(x.columns)
print(col)
"""
for i in (col):
    x_var = col
    y_var = x.pop(i)
    
    print(y_var)
    print(x_var)
    
    mod = sm.OLS(x_train[y_var], sm.add_constant(x_train[x_var]))
    mod = mod.fit()
    
    vif = 1/(1-mod.rsquared)
    print(y_var, round(vif, 3))
"""
print(x.shape)
x.drop(labels=['density', 'residual_sugar', 'fixed_acidity', 'citric_acid' ], inplace = True, axis=1)
print(x.shape)

y = df['quality']
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=44)
print()

x_train_new = sm.add_constant(x_train)
x_test_new = sm.add_constant(x_test)

full_mod = sm.OLS(y_train, x_train_new)
full_res = full_mod.fit()

print(full_res.summary())

print('Variable Infaltion Factor')
col = list(x.columns)
print(col)