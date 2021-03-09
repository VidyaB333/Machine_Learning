import  pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
seed=12

data = pd.read_csv(r'C:\Users\vidya\OneDrive\Desktop\Python_coding_practice_Datasets\PythonDataSets\python\PythonDataSets\Housing.csv')
print(data.shape)
col = data.columns
print(col)
x = data[['lotsize', 'bedrooms', 'bathrms', 'stories', 'garagepl']]
y = data['price']
print(x.shape)
correlation_m = data.corr()
sns.heatmap(correlation_m, annot=True)
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=seed)
best_features = SelectKBest(score_func=f_regression, k=5) #Learn relationship fromtraining data
x_new = best_features.fit(x_train, y_train)
x_train_trans = best_features.transform(x_train)
X_test_trans = best_features.transform(x_test)
print('Scores : ',best_features.scores_)

#print('New k best features :', x_new.shape)
#print(x_new) #-->lotsize=0.54, bathrooms=0.53, srories=0.42 bedrooms =0.37, garagepl=0.38


LR= LinearRegression()


LR =LR.fit(x_train_trans, y_train)

y_train_pred  = LR.predict(x_train_trans)
r2_train = r2_score(y_train, y_train_pred)
print('Training r2 score: ', r2_train)

y_test_pred  = LR.predict(X_test_trans)
r2_test = r2_score(y_test, y_test_pred)
print('Training r2 score: ', r2_test)
print('coefficients: ', LR.coef_)


"""
from sklearn.preprocessing import LabelEncoder
lb =LabelEncoder()


categorical_variables = ['driveway', 'recroom', 'fullbase', 'gashw', 'airco', 'prefarea']
x_cat = data[categorical_variables]
x_train, x_test, y_train, y_test = train_test_split(x_cat, y, train_size=0.75, random_state=seed)

for i in categorical_variables:
    data[i] = lb.fit_transform(data[i])
col = data.columns
print(col)
print(data.head(10))
x = data[col[1:]]
print(x.columns)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=seed)

LR =LR.fit(x_train, y_train)

y_train_pred  = LR.predict(x_train)
r2_train = r2_score(y_train, y_train_pred)
print('Training r2 score: ', r2_train)

y_test_pred  = LR.predict(x_test)
r2_test = r2_score(y_test, y_test_pred)
print('Training r2 score: ', r2_test)

best_features = SelectKBest(score_func=f_classif, k=5)

x_new_1 = best_features.fit_transform(x_cat, y)
print(x_new_1)
"""