import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, mutual_info_regression
from sklearn.datasets import make_regression

seed = 12

# Creating data for regression
x, y = make_regression(n_samples=1000, n_features=100, n_informative=20, n_targets=1, random_state=seed)
y = y.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=seed)
print(x_train.shape, y_train.shape)

LR = LinearRegression()

#Using all thr input variables
LR = LR.fit(x_train, y_train)
print('Training r2: ', r2_score(y_train, LR.predict(x_train)))
print('Testing r2: ', r2_score(y_test, LR.predict(x_test)))
print('MAE : ', mean_absolute_error(y_test, LR.predict(x_test)))

# Using 10 important input variables
print('# Using 10 important input variables')
SK = SelectKBest(score_func=f_regression, k=10)
x_new = SK.fit_transform(x, y)
x_train, x_test, y_train, y_test = train_test_split(x_new, y, train_size=0.75, random_state=seed)
print(x_train.shape, y_train.shape)

LR = LinearRegression()

#Using all thr input variables
LR = LR.fit(x_train, y_train)
print('Training r2: ', r2_score(y_train, LR.predict(x_train)))
print('Testing r2: ', r2_score(y_test, LR.predict(x_test)))
print('MAE : ', mean_absolute_error(y_test, LR.predict(x_test)))
for i in range(100):
    #print('Feature {} : {}'.format(i, round(SK.scores_[i],3)))
    pass
m = [i  for i in range(len(SK.scores_))]

#plt.bar( m, SK.scores_)
#plt.show()


#Using grid search method
print('\n\n\n')
print('Using grid search method')
y = y.ravel()
print(y.shape, y.ndim)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=seed)
pipeline = Pipeline([('sk', SelectKBest()),('lr', LinearRegression())])
params = {
    'sk__k':[10,20,30,40,50,60,70,80,90,100],
    'sk__score_func' :[f_regression, mutual_info_regression]
}
cv =RepeatedKFold(n_repeats=4, n_splits=5, random_state=seed)
Grid_s = GridSearchCV(pipeline,params,scoring='r2', cv=cv, n_jobs=-1 )
print('Grid search details: ', Grid_s)
results = Grid_s.fit(x_train, y_train)
print(results)

print('best score : ', results.best_score_)
print( results.best_estimator_)
d = results.cv_results_
d = pd.DataFrame(d)
print(d.to_csv(r'C:\Users\vidya\OneDrive\Desktop\Python_coding_practice_Datasets\PythonDataSets\python\PythonDataSets\new.csv'))
print()
print(d)
mean = results.cv_results_['mean_test_score']
print('All parameters')
params = results.cv_results_['params']
print(mean)
print(params)