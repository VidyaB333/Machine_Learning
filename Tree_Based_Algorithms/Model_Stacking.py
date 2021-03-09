import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier

def Preproceesing():
    df = pd.read_csv(
        "C:\\Users\\vidya\OneDrive\\Desktop\\Python_coding_practice_Datasets\\PythonDataSets\\DecisionTress\\HR-Employee-Attrition.csv")
    # print(df.shape)
    col = df.columns

    # Making categorical label to numerical
    print('Making categorical label to numerical')
    num = LabelEncoder()
    df['Attrition'] = num.fit_transform(df['Attrition'])
    # print(df['Attrition'].count())
    # print(df[df['Attrition'] == 0]['Attrition'].count())

    # Removed variables with uniform distribution as those variables provide no information
    print('Removed variables with uniform distribution as those variables provide no information')
    # print(len(df.columns))
    df.drop(['EmployeeCount', 'Over18', 'StandardHours'], axis=1, inplace=True)
    # print(len(df.columns))
    col = df.columns
    # print(col)

    # print('\n\n\n')

    # NUMERICAL AND CATEGORICAL VARIABLE DISTINGUHION
    print('NUMERICAL AND CATEGORICAL VARIABLE DISTINGUHION')
    numerical_variables = []
    Categorical_variables = []
    for i in col:
        if df[i].dtypes != 'object':
            numerical_variables.append(i)
        if df[i].dtypes == 'object':
            Categorical_variables.append(i)

    # print('\n\n\n')

    # FEATURE NORMALIZATION OF NUMERICAL FEATURES USING
    print('Feature Scaling of Numerical data')
    # print(df['HourlyRate'].head(10))
    scaler = MinMaxScaler()
    scaler = scaler.fit_transform(df[numerical_variables])
    # print(scaler.shape)
    df_num = pd.DataFrame(scaler)
    df_num.columns = numerical_variables
    # print(df_num['HourlyRate'].head(10))

    # print('\n\n\n\n')

    # Converting Categorical variables into Numrical types using one hot encoding through pd.get_dummies
    print('Converting Categorical variables into Numrical types using one hot encoding through pd.get_dummies')
    df_cat = pd.get_dummies(df[Categorical_variables], prefix='C')
    print('before applying one hot encoding on categorical variable len of columns {} and after {} '.format(
        len(Categorical_variables), len(df_cat.columns)))
    print('Before --> ', Categorical_variables)
    print('After --> ', list(df_cat.columns))

    # print('\n\n\n')

    # Combine all variables
    print('Combine all variables ')

    # print(df_cat.head(10))
    # print(df_num.head(10))
    df_new = pd.concat([df_cat, df_num], axis=1)
    # print(list(df_new.columns))
    # print(df_new.shape)

    # print(df_new.head(10))
    df_new.to_csv(
        "C:\\Users\\vidya\OneDrive\\Desktop\\Python_coding_practice_Datasets\\PythonDataSets\\DecisionTress\\HR-Employee-Attrition_scaler.csv")
    # print('\n\n\n\n')
    print('Splitting data into input and output')
    y = df_new['Attrition']
    df_new.drop('Attrition', axis=1, inplace=True)
    # print(len(df_new.columns))
    x = df_new
    # print(x.shape)

    # Checking for na values in dattaframe
    print('Checking for na values in dattaframe')
    # print(x.isna().count())
    # print(np.where(np.isnan(x)))
    # print(x.isna().any(axis=0).count())
    # print(x.isna().any(axis=0)['Age'])
    return x, y

#Base model creation
def Base_model_creation():
    # Created Base estimators for bagging classifier
    print('Created Base models for Stacking')

    models = dict()
    # Created different models woth default hyperparameter setting
    models['lr'] = LogisticRegression()
    models['dt'] = DecisionTreeClassifier()
    models['svm'] = SVC()
    models['nb'] = GaussianNB()
    models['knn'] = KNeighborsClassifier()
    return models

#Calculationg scores of models
def Model_scores(models, x, y):

    from sklearn.model_selection import cross_val_score
    print('In model calculation method')
    accuracy = []
    names = []
    for i in models.keys():
        print(models[i])
        scores = cross_val_score(models[i], x, y, cv=5, scoring='accuracy')
        #print(scores)
        accuracy.append(scores)
        names.append(i)
    print('In accuracy calculation: ', names)
    return accuracy, names

def stacking(models):

    estimators = []
    for i in models.keys():
        estimators.append((i, models[i]))
    print('Base models for stacking algo: ', estimators)

    meta_class_model = StackingClassifier(estimators= estimators, final_estimator=models['lr'], cv=5)
    models['stack'] = meta_class_model
    return models, meta_class_model


if __name__ == '__main__':

    print('Start point')
    x, y = Preproceesing()
    print(x.shape, y.shape)

    x_train, x_test,y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=12)

    #Base model creation
    models = Base_model_creation()
    print('All algorithms in stacking:', models)


    #stacking of base model
    models, meta_class_model = stacking(models) #New list of models for stacking

    accuracies, names = Model_scores(models, x_train,y_train, )


    for accuracy, name in zip(accuracies, names):
        print(accuracy, name)
        #print('Accuracy of %s : %.3f with standard deviation 5.3f'%(name, np.mean(accuracy), np.std(accuracy)))
        print('{0:10s} : {1:.4f} +/- {2:.3f}'.format(name, np.mean(accuracy), np.std(accuracy)))
        print()

    #Box plot for accuries
    plt.boxplot(accuracies, labels=names, showmeans=True)
    plt.show()


    #Need to create Stacking algo
    print('Prediction of new data')
    print(meta_class_model)
    meta_class_model=meta_class_model.fit(x_train, y_train)
    y_train_p = meta_class_model.predict(x_train)
    print('Training performance metrices')
    print(classification_report(y_train, y_train_p))
    print(accuracy_score(y_train, y_train_p))
    print(confusion_matrix(y_train, y_train_p))

    print('Testing performance metrices')
    y_test_p = meta_class_model.predict(x_test)
    print(classification_report(y_test, y_test_p))
    print(accuracy_score(y_test, y_test_p))
    print(confusion_matrix(y_test, y_test_p))


    print()
    print('No of classes in model: ', meta_class_model.classes_)
    print('List of esrtimator :', meta_class_model.estimators_)
    print(meta_class_model.named_estimators_)
    print(meta_class_model.final_estimator_)
    print(meta_class_model.stack_method_)

    #print(meta_class_model.coef)

