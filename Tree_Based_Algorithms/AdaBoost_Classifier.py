import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv(
    "C:\\Users\\vidya\OneDrive\\Desktop\\Python_coding_practice_Datasets\\PythonDataSets\\DecisionTress\\HR-Employee-Attrition.csv")
print(df.shape)
col = df.columns

# Making categorical label to numerical
print('Making categorical label to numerical')
num = LabelEncoder()
df['Attrition'] = num.fit_transform(df['Attrition'])
print(df['Attrition'].count())
print(df[df['Attrition'] == 0]['Attrition'].count())

# Removed variables with uniforn distribution as those variables provide no information
print('Removed variables with uniforn distribution as those variables provide no information')
print(len(df.columns))
df.drop(['EmployeeCount', 'Over18', 'StandardHours'], axis=1, inplace=True)
print(len(df.columns))
col = df.columns
print(col)

print('\n\n\n')

# NUMERICAL AND CATEGORICAL VARIABLE DISTINGUHION
print('NUMERICAL AND CATEGORICAL VARIABLE DISTINGUHION')
numerical_variables = []
Categorical_variables = []
for i in col:
    if df[i].dtypes != 'object':
        numerical_variables.append(i)
    if df[i].dtypes == 'object':
        Categorical_variables.append(i)

print('\n\n\n')

# FEATURE NORMALIZATION OF NUMERICAL FEATURES USING
print('Feature Scaling of Numerical data')
print(df['HourlyRate'].head(10))
scaler = MinMaxScaler()
scaler = scaler.fit_transform(df[numerical_variables])
print(scaler.shape)
df_num = pd.DataFrame(scaler)
df_num.columns = numerical_variables
print(df_num['HourlyRate'].head(10))

print('\n\n\n\n')

# Converting Categorical variables into Numrical types using one hot encoding through pd.get_dummies
print('Converting Categorical variables into Numrical types using one hot encoding through pd.get_dummies')
df_cat = pd.get_dummies(df[Categorical_variables], prefix='C')
print('before applying one hot encoding on categorical variable len of columns {} and after {} '.format(
    len(Categorical_variables), len(df_cat.columns)))
print('Before --> ', Categorical_variables)
print('After --> ', list(df_cat.columns))

print('\n\n\n')

# Combine all variables
print('Combine all variables ')

print(df_cat.head(10))
print(df_num.head(10))
df_new = pd.concat([df_cat, df_num], axis=1)
print(list(df_new.columns))
print(df_new.shape)

print(df_new.head(10))
df_new.to_csv(
    "C:\\Users\\vidya\OneDrive\\Desktop\\Python_coding_practice_Datasets\\PythonDataSets\\DecisionTress\\HR-Employee-Attrition_scaler.csv")
print('\n\n\n\n')
print('Splitting data into input and output')
y = df_new['Attrition']
df_new.drop('Attrition', axis=1, inplace=True)
print(len(df_new.columns))
x = df_new
print(x.shape)

# Checking for na values in dattaframe
print('Checking for na values in dattaframe')
# print(x.isna().count())
print(np.where(np.isnan(x)))
print(x.isna().any(axis=0).count())
print(x.isna().any(axis=0)['Age'])

print('\n\n')
seed = 12

# Created Base estimator for bagging classifier
print('Created Base estimator for bagging classifier')
DT = DecisionTreeClassifier(criterion='entropy', max_depth=1,
                             class_weight={0: 0.3, 1: 0.7}, random_state=seed)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=seed)

print(DT)

# Creating Boosting classifer
print(' Adaptive Boost classifier')

from sklearn.ensemble import AdaBoostClassifier

##Using GridSearch on ADABOOST classifier
"""
AdaBoost_estimator = AdaBoostClassifier(base_estimator=DT
                                      )

from sklearn.pipeline import Pipeline
pipeline = Pipeline([('ABC', AdaBoostClassifier(base_estimator=DT))])

parameters = {
    'ABC__n_estimators': [50,100,200,500,1000,2000]
    }

Grid_search = GridSearchCV(pipeline, parameters,scoring='accuracy', cv=5)
print(AdaBoost_estimator)
print(Grid_search)
Grid_search = Grid_search.fit(x_train, y_train)

print('Best Score of grid search : ', Grid_search.best_score_)
print('Best estimator of grid search')
print(Grid_search.best_estimator_)
best_parameters = Grid_search.best_estimator_.get_params()
for i in parameters.keys():
    print('{} : {}'.format(i, best_parameters[i]))

print('Best value of n_estimators in Adaboost classifier : ', Grid_search.best_estimator_)
"""





AdaBoost_estimator = AdaBoostClassifier(base_estimator=DT, n_estimators=200, learning_rate=0.01
                                      )
AdaBoost_estimator.fit(x_train, y_train)
y_train_predict = AdaBoost_estimator.predict(x_train)
print('Training accuracy: ', accuracy_score(y_train, y_train_predict))
print(confusion_matrix(y_train, y_train_predict))
print(classification_report(y_train, y_train_predict))


y_test_predict = AdaBoost_estimator.predict(x_test)
print('Testing accuracy: ', accuracy_score(y_test, y_test_predict))
print(confusion_matrix(y_test, y_test_predict))
print(classification_report(y_test, y_test_predict))
