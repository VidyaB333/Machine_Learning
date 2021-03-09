import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score, classification_report
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt


"""
Dataset has downloaded from UCI & then converted to df & then to CSV
columns = ['Status of existing checking account', 'Duration in month',
              'Credit history', 'Purpose', 'Credit amount', 'Savings account/bonds',
              'Present employment since', 'Installment rate in percentage of disposable income',
              'Personal status and sex', 'Other debtors / guarantors', 'Present residence since',
              'Property', 'Age in years','Other installment plans', 'Housing', 'Number of existing credits at this bank',
              'Job', 'Number of people being liable to provide maintenance for','Telephone', 'foreign worker', 'Customer type']
df = pd.read_csv('C:\\Users\\vidya\\OneDrive\\Desktop\Python_coding_practice_Datasets\\PythonDataSets\\Logistic_Regression\\german.data', sep=' ', names= columns)
col = df.columns
new_col = [i.replace(' ', '_') for i in col]
df.columns = new_col
print(df.columns)
print(df.shape)
df.to_csv('C:\\Users\\vidya\\OneDrive\\Desktop\Python_coding_practice_Datasets\\PythonDataSets\\Logistic_Regression\\german_credit_data.csv')
#print(df.head(1))
"""

df = pd.read_csv('C:\\Users\\vidya\\OneDrive\\Desktop\Python_coding_practice_Datasets\\PythonDataSets\\Logistic_Regression\\german_credit_data.csv')
#print(df.shape)
#print(df.columns)
df.drop('Unnamed: 0', axis=1, inplace=True) #Extra columns has created
print(df.shape)
print(df.head(5))
df['Customer_type'] = df['Customer_type']-1


"""
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=12)
#print(X_train.head())

model = LogisticRegression()
model.fit(X_train, y_train)
print()
y_pred = model.predict(X_test)
conf_m = confusion_matrix(y_test, y_pred)
print(conf_m)
"""

#Calculation of Information Value of features
print('Information value')
def information_value(feature):
    if df[feature].dtypes == 'object':
        dataf = df.groupby(feature)['Customer_type'].agg(['count', 'sum'])
        dataf.columns = ['Total', 'Bad']
        dataf['Good'] = dataf['Total'] - dataf['Bad']
        dataf['%ev'] = dataf['Bad'] / dataf['Bad'].sum()
        dataf['%non_ev'] = dataf['Good'] / dataf['Good'].sum()
        dataf['WOE'] = np.log(dataf['%non_ev'] / dataf['%ev'])
        dataf['IV'] = (dataf['%non_ev'] - dataf['%ev']) * dataf['WOE']
        # print(dataf)
        print('%20s is %.5f:  %10s' % (feature, dataf['IV'].sum(), df[feature].dtypes))

    else:
        df['bin_val'] = pd.qcut(df[feature].rank(method='first'), 10)
        dataf = df.groupby(['bin_val'])['Customer_type'].agg(['count', 'sum'])
        dataf.columns = ['Total', 'Bad']
        dataf['Good'] = dataf['Total'] - dataf['Bad']
        dataf['%ev'] = dataf['Bad'] / dataf['Bad'].sum()
        dataf['%non_ev'] = dataf['Good'] / dataf['Good'].sum()
        dataf['WOE'] = np.log(dataf['%non_ev'] / dataf['%ev'])
        dataf['IV'] = (dataf['%non_ev'] - dataf['%ev']) * dataf['WOE']
        # print(dataf)
        print('%20s is %.5f:   %10s' % (feature, dataf['IV'].sum(), df[feature].dtypes))
        # print(dataf)



print(df.info())

cateorical_var = ['Status_of_existing_checking_account', 'Credit_history', 'Purpose', 'Savings_account/bonds', 'Present_employment_since',
                  'Personal_status_and_sex','Other_debtors_/_guarantors', 'Property', 'Other_installment_plans',
                  'Housing', 'Job', 'Telephone', 'foreign_worker']

numerical_var = ['Duration_in_month', 'Credit_amount', 'Installment_rate_in_percentage_of_disposable_income','Present_residence_since',
                 'Age_in_years', 'Number_of_existing_credits_at_this_bank','Number_of_people_being_liable_to_provide_maintenance_for',
                 ]

variables= cateorical_var + numerical_var

for i in variables:
    information_value(i)


#Creating Dummy variables for categorical features
dummy_Credit_history = pd.get_dummies(df['Credit_history'], prefix='Cre_history', )
dummy_Purpose = pd.get_dummies(df['Purpose'], prefix='Purpose')
dummy_Savings_account = pd.get_dummies(df['Savings_account/bonds'], prefix='Savings_acct')
dummy_Property = pd.get_dummies(df['Property'], prefix='Property')

df_continous = df[numerical_var]
df = pd.concat([dummy_Credit_history, dummy_Purpose, dummy_Savings_account,
                dummy_Property, df_continous, df['Customer_type']], axis=1)
print(df.shape)
print(df.head())

remove_dummy_variable = ['Purpose_A49', 'Cre_history_A34', 'Savings_acct_A65', 'Property_A124']
remove_insignificant_features =[]
remove_multicollinear_variable = []

remove_features = list(set(remove_dummy_variable + remove_insignificant_features + remove_multicollinear_variable))
print(remove_features)

df.drop(remove_features, axis=1, inplace=True)
print(df.shape)

X = df[df.columns[0:-1]]
y = df[df.columns[-1]]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=12)
#print(type(X_train))
#print(type(y_train))
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

print('Craeting Logistic Regression Model')
## Creating statsmodel
X_train = sm.add_constant(X_train)
#print(X_train)
print((X_train.dtypes))
print(type(y_train))
logistic_regression = sm.Logit(y_train, X_train).fit()
print(logistic_regression.summary())





print('Iteration 2')
########
remove_insignificant_features = ['Purpose_A410', 'Purpose_A48', 'Savings_acct_A64','Cre_history_A32',
                                'Number_of_existing_credits_at_this_bank', 'Number_of_people_being_liable_to_provide_maintenance_for']
remove_multicollinear_variable = []


remove_features = list(set(remove_insignificant_features + remove_multicollinear_variable))
print('Remove features')
print(remove_features)

df.drop(remove_features, axis=1, inplace=True)
print(df.shape)

X = df[df.columns[0:-1]]
y = df[df.columns[-1]]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=12)
#print(type(X_train))
#print(type(y_train))
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

print('Craeting Logistic Regression Model')
## Creating statsmodel
X_train = sm.add_constant(X_train)
#print(X_train)
print((X_train.dtypes))
print(type(y_train))
logistic_regression = sm.Logit(y_train, X_train).fit()
print(logistic_regression.summary())

print('Predictiong the labels for test dataset')
y_pred = logistic_regression.predict(sm.add_constant(X_test))
#print(round(y_pred))
print(r2_score(y_test, y_pred))

col = list(X_train.columns[1:])
print(len(col))
#print(col)
"""
print('Variance Inflation Factor')
for i in col:
    y_var = col.pop()
    x_var = col

    mod = sm.Logit(X_train[y_var], sm.add_constant(X_train[x_var]))
    Logit_m = mod.fit()
    y_pred = Logit_m.predict(sm.add_constant(X_train[x_var]))
    vif = 1/(1 - r2_score(X_train[y_var], y_pred))
    print(y_var, vif)
"""
#Plotting of ROC Curve and calculating AUC

print('ROC Curve')
#Created data dataframe to store details of test dataset
data = pd.DataFrame(y_test )
data.columns = ['Actual']
data['Predicted'] = y_pred

fpr, tpr, threhold = roc_curve(data['Actual'], data['Predicted'], pos_label=1)

print(fpr.shape, tpr.shape, threhold.shape)

print('Values of FPR for given frequencies, TPR, Frequencies ')
for i in range(len(fpr)):
    print(round(fpr[i],3), round(tpr[i],3), round(threhold[i],3))

#Finding Area Under ROC curve
roc_auc = auc(fpr, tpr)
print('Area under ROC Curve :', roc_auc)

#Plotting ROC curve using FTR, TPR
plt.plot(fpr, tpr, color= 'darkorange', label = 'ROC Curve(area=%.4f'%roc_auc)
plt.plot([0,1],[0,1], color = 'navy')
plt.xlim([-0.2, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity/FPR')
plt.ylabel('Sensitivity/TPR')
plt.title('ROC curve')
plt.legend(loc = 'lower right')
plt.show()

#Different thresholds used in plotting ROC Curve
plt.scatter((np.linspace(0,1,len(threhold))), threhold)
plt.ylabel('Frequencies')
plt.show()


#Finnding the best threshold on training dataset

#Created data_train dataset to store the details of train set
data_train = pd.DataFrame(y_train)
y_train_predict = logistic_regression.predict(sm.add_constant(X_train))
data_train['y_train_predict'] = y_train_predict
#print(data_train)

for i in np.arange(0.1,1,0.1):
    data_train.loc[data_train['y_train_predict'] >=i, 'new'] = 1
    data_train.loc[data_train['y_train_predict'] <= i, 'new'] =0
    data_train['new']= data_train['new'].astype('int64') #Changing the datatype to int

    #print(data_train)
    #print(data_train.dtypes)
    acc = accuracy_score(data_train['Customer_type'], data_train['new'])
    print('Threshold : {} and Training Accuracy: {}'.format(round(i, 4), round(acc, 4)))

data.loc[data['Predicted'] >= 0.5 ,'NEW_PREDICTED']= 1
data.loc[data['Predicted'] < 0.5 ,'NEW_PREDICTED']= 0
print(data.columns)
print(data)
data['NEW_PREDICTED'] = data['NEW_PREDICTED'].astype('int')
print(data.dtypes)

print('Classification report on testing dataset :\n',classification_report(data['Actual'], data['NEW_PREDICTED']))

print('Confusion matrix on testing dataset :\n',confusion_matrix(data['Actual'], data['NEW_PREDICTED']))
print('Accurancy Score: ', accuracy_score(data['Actual'], data['NEW_PREDICTED']))

