import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
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
sns.countplot(x=df['Attrition'], data=df)
plt.show()




"""
print('Distribution of all features')
#Distribution of features
for i in col:
    sns.displot(df[i])
    plt.title(i)
    plt.show()
"""



#Removed variables with uniforn distribution as those variables provide no information
print('Removed variables with uniforn distribution as those variables provide no information')
print(len(df.columns))
df.drop(['EmployeeCount','Over18', 'StandardHours'], axis=1, inplace=True)
print(len(df.columns))
col = df.columns
print(col)


print('\n\n\n\n')



#NUMERICAL AND CATEGORICAL VARIABLE DISTINGUHION
print('NUMERICAL AND CATEGORICAL VARIABLE DISTINGUHION')
numerical_variables = []
Categorical_variables = []
for i in col:
    if df[i].dtypes !='object':
        numerical_variables.append(i)
    if df[i].dtypes =='object':
        Categorical_variables.append(i)
print('Numberical variables : \n', numerical_variables)
print(len(numerical_variables))
print('Categorical variables : \n', Categorical_variables)
print(len(Categorical_variables))



print('\n\n\n\n')



#FEATURE NORMALIZATION OF NUMERICAL FEATURES USING
print('Feature Scaling of Numerical data')
print(df['HourlyRate'].head(10))
scaler = MinMaxScaler()
scaler = scaler.fit_transform(df[numerical_variables])
print(scaler.shape)
df_num = pd.DataFrame(scaler)
df_num.columns = numerical_variables
print(df_num['HourlyRate'].head(10))


print('\n\n\n\n\n')





#Converting Categorical variables into Numrical types using one hot encoding through pd.get_dummies
print('Converting Categorical variables into Numrical types using one hot encoding through pd.get_dummies')
df_cat = pd.get_dummies(df[Categorical_variables], prefix='C', drop_first=True)
print('before applying one hot encoding on categorical variable len of columns {} and after {} '.format(len(Categorical_variables), len(df_cat.columns)))
print('Before --> ', Categorical_variables)
print('After --> ',list(df_cat.columns))


print('\n\n\n\n')


#Combine all variables
print('Combine all variables ')

print(df_cat.head(10))
print(df_num.head(10))
df_new = pd.concat([df_cat, df_num], axis=1)
print(list(df_new.columns))
print(df_new.shape)


print(df_new.head(10))
df_new.to_csv("C:\\Users\\vidya\OneDrive\\Desktop\\Python_coding_practice_Datasets\\PythonDataSets\\DecisionTress\\HR-Employee-Attrition_scaler.csv")
print('\n\n\n\n')
print('Splitting data into input and output')
y = df_new['Attrition']
df_new.drop('Attrition', axis =1, inplace =True)
print(len(df_new.columns))
x = df_new
print(x.shape)


#Checking for na values in dattaframe
print('Checking for na values in dattaframe')
#print(x.isna().count())
print(np.where(np.isnan(x)))
print(x.isna().any(axis = 0).count())
print(x.isna().any(axis = 0)['Age'])

print('\n\n')


dummy_zeros =np.zeros((6,10))
data = pd.DataFrame(dummy_zeros)
data.columns = ['Training_Accuracy', 'Testing_Accuracy', 'O class wt', '1 class wt', '0precision', '1precision', 'totalPrecision', '0recall','1recall', 'totalrecall' ]
zero_category_wt =[0.01,0.1,0.2,0.3,0.5]
for i in range(len(zero_category_wt)):
    print()
    class_wt = {0: zero_category_wt[i], 1: (1 - zero_category_wt[i])}
    print(class_wt)
    DT = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=2,
                                min_samples_leaf=1, random_state=42, class_weight=class_wt)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=12)
    #print(DT)
    DT_model = DT.fit(x_train, y_train)
    #print(DT_model)

    # Checking metrices on training dataset
    print('Checking metrices on training dataset')
    y_train_predict = DT_model.predict(x_train)
    #print(classification_report(y_train, y_train_predict))
    cm = confusion_matrix(y_train, y_train_predict)
    #print('Confusion Matrix :\n')
    #print(cm)
    print('Accuracy Score : ', accuracy_score(y_train, y_train_predict))
    data.loc[i, 'Training_Accuracy'] = accuracy_score(y_train, y_train_predict)


    # Checking metrices on testing dataset
    print('Checking metrices on testing dataset')


    y_test_predict = DT_model.predict(x_test)
    print(classification_report(y_test, y_test_predict))

    cm = confusion_matrix(y_test, y_test_predict)
    print('Confusion Matrix :\n')
    print(cm)
    print('Accuracy Score : ', accuracy_score(y_test, y_test_predict))


##

print('Balancuing')

DT1 = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=2,
                                min_samples_leaf=1, random_state=42, class_weight='balancing')
DT1 = DT1 = DT.fit(x_train, y_train)
print(classification_report(y_train, DT1.predict(x_train)))


