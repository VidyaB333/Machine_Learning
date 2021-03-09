import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer.csv', header=None)

df.columns = ['age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps',
              'deg-malig', 'breast', 'breast-quad', 'irradiat', 'class']
print(df.head(1))

print('\n\n\n')
print('Working on missing data in df')
##Working on missing data in df
# print('Total missing values: ', df.isna().sum())
df['node-caps'].fillna('no', inplace=True)
df['breast-quad'].fillna('central', inplace=True)
# print(df.isna().sum())

# Doing Feature Engineering on categorical variables
print('\n\n\n')
print('Doing Feature Engineering on categorical variables')

##1: Converting datatype of column values
print('1: Converting datatype of column values')
df['deg-malig'] = df['deg-malig'].apply(lambda x:int(x.split('\'')[1]))


"""
arr = np.array(df['deg-malig'])
# print(arr)
for i in range(len(arr)):
    a = int(arr[i].split('\'')[1])
    arr[i] = a
# print(arr)
df['deg-malig'] = pd.DataFrame(arr)
df['deg-malig'] = df['deg-malig'].astype(int)
"""

print('2: Converting categorical to numerical using Label encoder')
varaibles_with_2op = ['class', 'irradiat', 'breast', 'node-caps']
LE = LabelEncoder()
for i in varaibles_with_2op:
    df[i] = LE.fit_transform(df[i])

print(df[varaibles_with_2op].head(10))

print('3: Converting categorical to numerical using ONEHOTENCODING')
varaibles_with_morethan_2op = ['menopause', 'breast-quad']

df_menopause = pd.get_dummies(df['menopause'], drop_first=True)
df_quad = pd.get_dummies(df['breast-quad'], drop_first=True)

df_new = pd.concat([df, df_menopause, df_quad], axis=1)
df_new.drop('central', axis=1, inplace=True)
df_new.drop('menopause', axis=1, inplace=True)

print('New columns : ', df_new.columns)
print('lenght: ', len(df_new.columns))
print(df.head(3))



print(df[df_new['breast-quad'] == 'central'])
df_new.loc[240, 'breast-quad'] = "\'central\'"
print(df_new['breast-quad'].loc[240])


df_new.drop('breast-quad', axis=1, inplace=True)
# Using dictionary
print('Using Dictionary')
age = {"'20-29'": 0, "'30-39'": 1, "'40-49'": 2, "'50-59'": 3, "'60-69'": 4, "'70-79'": 5}
inv_nodes = {"'0-2'": 0, "'3-5'": 1, "'15-17'": 5, "'6-8'": 2, "'9-11'": 3, "'24-26'": 10, "'12-14'": 4}
tumor_size = {"'0-4'": 0, "'5-9'": 1, "'10-14'": 2, "'15-19'": 3, "'20-24'": 4, "'25-29'": 5,
              "'30-34'": 6, "'35-39'": 7, "'40-44'": 8, "'45-49'": 9, "'50-54'": 10}
# df_new.replace({"age": age, 'tumor-size':tumor_size}, inplace=True)
df_new['age'].replace(to_replace=age, inplace=True)
df_new['tumor-size'].replace(tumor_size, inplace=True)
df_new['inv-nodes'].replace(inv_nodes, inplace=True)

print(df_new.head(5))

df_new.rename(columns={"'lt40'": 'lt40',
                       "'premeno'": 'premeno',
                       "'left_low'": 'left_low',
                       "'left_up'": 'left_up',
                       "'right_low'": 'right_low',
                       "'right_up'": 'right_up'}, inplace=True)

print(df_new.columns)



df_new.to_csv(
    r'C:\Users\vidya\OneDrive\Desktop\Python_coding_practice_Datasets\PythonDataSets\Feature_selection_Gs\breast_cancer_after3.csv')

print(df_new.columns)
x = df_new[['age', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast',
       'irradiat', 'lt40', 'premeno', 'left_low', 'left_up',
       'right_low', 'right_up']]
y =df_new['class']
print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=12)

from sklearn.pipeline import  Pipeline
from sklearn.model_selection import GridSearchCV, RepeatedKFold
pipeline =Pipeline([('RF', RandomForestClassifier(criterion='gini', class_weight={0:0.3, 1:0.7}))])
#RF = RandomForestClassifier(=100, criterion='gini', max_depth=2)
params = {
    'RF__n_estimators':[100,200,500,1000],
    'RF__max_depth':[2,3,4]
}
cv = RepeatedKFold(n_repeats=2, n_splits=4)

GS= GridSearchCV(pipeline, params, scoring='recall', n_jobs=-1,cv=cv)
GS.fit(x_train, y_train)
print(GS.best_estimator_)
print(GS.best_score_)
d = pd.DataFrame(GS.cv_results_)
d.to_csv(r'C:\Users\vidya\OneDrive\Desktop\Python_coding_practice_Datasets\PythonDataSets\Feature_selection_Gs\GS.csv')


print(accuracy_score(y_train, GS.predict(x_train)))
print(confusion_matrix(y_train, GS.predict(x_train)))
print(classification_report(y_train, GS.predict(x_train)))

print(accuracy_score(y_test, GS.predict(x_test)))
print(confusion_matrix(y_test, GS.predict(x_test)))
print(classification_report(y_test, GS.predict(x_test)))