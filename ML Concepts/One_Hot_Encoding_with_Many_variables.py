import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv(
    r'C:\Users\vidya\OneDrive\Desktop\Python_coding_practice_Datasets\PythonDataSets\Feature_selection_Gs\mercedesebenz\train.csv',
    usecols=['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8'])
print(df.columns)
print(df.shape)

# print(df['X0'].value_counts())
for col in df.columns:
    print(col, ':', len(df[col].unique()))
    pass

# print(df['X0'].value_counts().sort_values(ascending=False))
top_15_X0 = df['X0'].value_counts().sort_values(ascending=False).head(15).index
top_15_X0 = list(top_15_X0)
for label in top_15_X0:
    a = 'X0_' + label
    df[a] = np.where(df['X0'] == label, 1, 0)
    top_15_X0[top_15_X0.index(label)] = a
print(df[['X0'] + top_15_X0].head(2))

top_15_X1 = df['X1'].value_counts().sort_values(ascending=False).head(15).index
top_15_X1 = list(top_15_X1)
for label in top_15_X1:
    a = 'X1_' + label
    df[a] = np.where(df['X1'] == label, 1, 0)
    top_15_X1[top_15_X1.index(label)] = a
print(top_15_X1)
print(df[['X1'] + top_15_X1].head(2))



top_15_X2 = df['X2'].value_counts().sort_values(ascending=False).head(15).index
top_15_X2 = list(top_15_X2)
print(top_15_X2)
for label in top_15_X2:
    a = 'X2_' + label
    df[a] = np.where(df['X2'] == label, 1, 0)
    top_15_X2[top_15_X2.index(label)] = a
print(df[['X2'] + top_15_X2].head(2))

top_7_X3 = df['X3'].value_counts().sort_values(ascending=False).head(7).index
top_7_X3 = list(top_7_X3)
for label in top_7_X3:
    a = 'X3_'+label
    df[a] = np.where(df['X3'] == label, 1, 0)
    top_7_X3[top_7_X3.index(label)] = a
print(df[['X3'] + top_7_X3].head(2))


top_4_X4 = df['X4'].value_counts().sort_values(ascending=False).head(4).index
top_4_X4 = list(top_4_X4)
# print(top_4_X4)
for label in top_4_X4:
    a = 'X4_' + label
    df[a] = np.where(df['X4'] == label, 1, 0)
    top_4_X4[top_4_X4.index(label)] = a
print(df[['X4'] + top_4_X4].head(2))




top_20_X5 = df['X5'].value_counts().sort_values(ascending=False).head(20).index
top_20_X5 = list(top_20_X5)
for label in top_20_X5:
    a = 'X5_' + label
    df[a] = np.where(df['X5'] == label, 1, 0)
    top_20_X5[top_20_X5.index(label)] = a
print(df[['X5'] + top_20_X5].head(2))


top_7_X6 = df['X6'].value_counts().sort_values(ascending=False).head(7).index
top_7_X6 = list(top_7_X6)
# print(top_7_X6)
for label in top_7_X6:
    a = 'X6_' + label
    df[a] = np.where(df['X6'] == label, 1, 0)
    top_7_X6[top_7_X6.index(label)] = a
print(df[['X6'] + top_7_X6].head(2))

top_15_X8 = df['X8'].value_counts().sort_values(ascending=False).head(15).index
top_15_X8 = list(top_15_X8)
# print(top_15_X8)
for label in top_15_X8:
    a = 'X8_' + label
    df[a] = np.where(df['X8'] == label, 1, 0)
    top_15_X8[top_15_X8.index(label)] = a
print(df[['X8'] + top_15_X8].head(2))


print(df.columns)
print(len(df.columns))
df.drop(['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8'],axis=1, inplace=True)

print(len(df.columns))
print(df.columns)
y= pd.read_csv(
    r'C:\Users\vidya\OneDrive\Desktop\Python_coding_practice_Datasets\PythonDataSets\Feature_selection_Gs\mercedesebenz\train.csv',
    usecols=['y'])
print(df.shape, y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=12)
print(x_train.shape, x_test.shape)
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
LR= LinearRegression().fit(x_train, y_train)
y_train_predict = LR.predict(x_train)
print('r2 score: ', r2_score(y_train, y_train_predict))
print('RMSE: ', math.sqrt(mean_squared_error(y_train, y_train_predict)))

y_test_predict = LR.predict(x_test)
print('r2 score: ', r2_score(y_test, y_test_predict))
print('RMSE: ', math.sqrt(mean_squared_error(y_test, y_test_predict)))
coef = LR.coef_
#print('Coefficients for linear regression: ', coef)
print('Intercept for linear regression: ', LR.intercept_)
print('Coefficient of LR model')
l = list(coef)[0]
l = [round(i, 3) for i in l]
print(len(l), l)
l1 =[i for i in range(len(l))]
print(len(l1), l1)
plt.bar(l1, l)
plt.title('Variable importance for categorical variables')
plt.xticks(range(len(l1)), list(df.columns))
plt.ylim(-40,40)
plt.show()

print('Vraibles with sorted values')

l_sort_value = sorted(l)
print(l_sort_value)
l_sort_arg =  np.argsort(l)
print(l_sort_arg)
print(l[l_sort_arg[0]])
print(len(df.columns))

for i in range(len(df.columns)):
    print('Feature ',(df.columns[l_sort_arg[i]]), ': ', l[l_sort_arg[i]])

d1 = df[['X3_c', 'X3_f', 'X3_a', 'X3_d', 'X3_g',
       'X3_e', 'X3_b', 'X4_d', 'X4_a', 'X4_b', 'X4_c']]
data = pd.concat([d1, y])
print(data.shape)
corre_m = data.corr()
sns.heatmap(corre_m, annot=True)
plt.show()

