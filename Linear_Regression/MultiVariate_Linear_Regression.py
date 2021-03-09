import math
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics, linear_model
from sklearn.metrics import accuracy_score, mean_squared_error
from matplotlib import pyplot as plt


df = pd.read_csv(
    'C:\\Users\\vidya\OneDrive\\Desktop\\Python_coding_practice_Datasets\\PythonDataSets\\Logistic_Regression\\archive\\wineQualityReds.csv')
print(df.shape)
print(df.index)  # Type of index is RangeIndex

# Renaming the columns of dataframe
col = df.columns
col = [i.replace('.', '_') for i in col]
# col = [i.upper() for i in col]
print(col)
df.columns = col

# Dropping irrelevant columns from dataset
df = df.drop(['Unnamed: 0'], axis=1)
print(df['free_sulfur_dioxide'].head(10))

# Checking the statistical summery, Information of dataset
print(df.describe(include='all'))
print(df.info())
# All features are numeric in nature
# No need to do type conversion , one hot encodeing, label encoder

# Checking for missing value
print(df.isna().any())
# print(df.isna().sum())
# print(df.isnull().values.sum())

# Filling na values in pandas
# print(df.fillna(np.mean, inplace=True))
# print(df.isna().any())

# Spliting of data set
X = df[df.columns[0:-1]]
print(X.shape)
y = df[df.columns[-1]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)
print(X_train.shape)

print(' Feature Scaling Using standardization')
# Feature Scaling Using standardization
Scaler = StandardScaler()
print(Scaler.fit(X_train))
print('means of features')
print(Scaler.mean_)
# print(Scaler.n_samples_seen_)
# print(Scaler.scale_)
X_train_scale = Scaler.transform(X_train)

print(type(X_train), type(X_train_scale))
# print(X_train[0:2, :])
# print(X_train_scale[0])

X_test_scale = Scaler.transform(X_test)

# Training the linear Regression
model = LinearRegression()
model = model.fit(X_train_scale, y_train)
print(y_train.shape, X_train_scale.shape)

print('Model Coefficients: {} and Intercept {}'.format(model.coef_, model.intercept_))
print('Varaiance of training dataset :,', model.score(X_train_scale, y_train))

# print('Training r2', r2_score(y_train, model.predict(np.array(X_train_scale[2]).reshape((1, -1)))))

print()
print(X_test_scale[2], y_test[2])

op = model.predict(np.array(X_test_scale[2]).reshape((1, -1)))
print('actual : {} and predicted {}'.format(y_test[2], op))

print(X_test_scale.shape, y_test.shape)
print('Varaiance of testing dataset : ', model.score(X_test_scale, y_test))
# print('Testing R2: ', metrics.r2_score(y_test, op))


# Different performance matrices of linear regression
print('Different performance metices of linear regression on Testing dataset')
print('MSE: ', metrics.mean_squared_error(y_test, model.predict(X_test_scale)))
print('RMSE: ', math.sqrt(metrics.mean_squared_error(y_test, model.predict(X_test_scale))))
print('R2 score: ', metrics.r2_score(y_test, model.predict(X_test_scale)))

print()
print('Ridge Regression')

# Ridge Regression
Ridge_regression = linear_model.Ridge(alpha=0.0001)
ridge_model = Ridge_regression.fit(X_train_scale, y_train)
print('Coefficients for ridge regression :', ridge_model.coef_)
print('Different performance metices of Ridge regression on Testing dataset')
print('MSE: ', metrics.mean_squared_error(y_test, ridge_model.predict(X_test_scale)))
print('RMSE: ', math.sqrt(metrics.mean_squared_error(y_test, ridge_model.predict(X_test_scale))))
print('R2 score: ', metrics.r2_score(y_test, ridge_model.predict(X_test_scale)))

# Lasso Regression
Lasso_regression = linear_model.Lasso(alpha=0.0001)
lasso_model = Lasso_regression.fit(X_train_scale, y_train)
print('Coefficients for Lasso regression :', lasso_model.coef_)
print('Different performance metices of Lasso regression on Testing dataset')
print('MSE: ', metrics.mean_squared_error(y_test, lasso_model.predict(X_test_scale)))
print('RMSE: ', math.sqrt(metrics.mean_squared_error(y_test, lasso_model.predict(X_test_scale))))
print('R2 score: ', metrics.r2_score(y_test, lasso_model.predict(X_test_scale)))
print()

columns = X_train.columns
print('%20s %s %s %s ' % ('Coefficient', 'Linear_Regre', 'Ridge_Regr', 'Lasso_Regre'))
for i in range(len(columns)):
    print('%20s %10.4f %10.4f %10.4f' % (
    columns[i], round(model.coef_[i], 3), round(ridge_model.coef_[i], 3), round(lasso_model.coef_[i], 3)))

print()
correlation_matrix = df.corr()


sns.heatmap(data=correlation_matrix, annot=True)
plt.title('Correlation Coefficient/ Multi-collinearty graph')
plt.show()

plt.bar(X_train.columns, model.coef_)
plt.title('Variable Importance Graph')
plt.show()