import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
import math
import seaborn as sns

##Loading data from csv file using pandas read_csv function
dataset = pd.read_csv('C:\\Users\\vidya\OneDrive\\Desktop\\Python_coding_practice_Datasets\\PythonDataSets\\Logistic_Regression\\archive\\wineQualityReds.csv')
print(dataset.shape)
columns = dataset.columns


DS = dataset.drop([dataset.columns[0]], axis='columns')
print(DS.shape)

DS.dropna(inplace=True)


##REnaming of columns
def rename(x):
    return x.replace('.', '_')

DS.rename(columns=lambda x: rename(x), inplace=True)
print(DS.columns)


##Visuation of relationship between input and output

# plt.scatter(DS['fixed_acidity'], DS['quality'])
# plt.show()

def plotting(column):
    plt.scatter(DS[column], DS['quality'])
    plt.title(column)
    plt.xlabel(column)
    plt.ylabel('quality')
    plt.show()

# for i in DS.columns:
#    plotting(i)

#Using above scatter plot, we have found that alcohol and quality has linear relationship

#TRaining the module
X_train, X_test, y_train, y_test = train_test_split(DS['alcohol'], DS['quality'], train_size=0.7, random_state=1)
print(X_train.shape, X_test.shape)
print(type(X_train))
print(X_train.shape)
print(X_train.ndim)

#Sklearn library works on numpya array

X_train = np.array(X_train)
print(type(X_train))
print(X_train.shape)
X_train = np.reshape(X_train,(-1, 1))
print(X_train.shape)
print(X_train.ndim)
###Simple LInear REgression

simple_r = LinearRegression()
simple_r = simple_r.fit(X_train, y_train)

print('Coefficient : ', simple_r.coef_)
print('INtercept: ', simple_r.intercept_)
print('Score:')
#print('Intercept: ', simple_r.intercept )

print('Prection of linear regression')
X_test = np.array(X_test)
X_test = np.reshape(X_test,(-1,1))
print(X_test.shape)
y_pred = simple_r.predict(X_test)

####Performance metrices of Linear regression

Score = r2_score(y_test, y_pred)
print('Score : ', Score)

plt.scatter(X_test, y_test)

plt.scatter(X_test, y_pred)
#plt.show()

print('With all variables LInear regression')
print(DS.shape)
y = DS['quality']
print(y.shape)
X = DS.drop([DS.columns[-1]], axis='columns')
print(X.shape)
print(type(X))
print(X.ndim)
y = np.array(y)
y = np.reshape(y, (-1, 1))
print(y.shape)
print(type(y))
print(y.ndim)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)
model = LinearRegression()
print('Linear regression model')
model = model.fit(X_train, y_train)
y_predict = model.predict(X_test)
r2 = r2_score(y_test, y_predict)
print(r2)
print('Coef for multiple input:', model.coef_ )
print('intercept: ', model.intercept_)
# Correlation Coefficient for each feature
y_mean = np.mean(y_train)
n = 1199
print('mean of y: ', y_mean)
print()

def variance(input):
    sum = 0
    mean = np.mean(input)
    #print('Mean of input array :', mean)
    for i in range(n):
        sum += (input[i] - mean) ** 2
        #print(i, input[i], sum)
    var = sum / (n - 1)
    return var


y_variance = math.sqrt(variance(y_train))
print('Varaiance of y :', y_variance)

print('for input features')
X_fixed_acidity_variance = math.sqrt(variance(np.array(X_train['fixed_acidity'])))
print('Varaiance of X_fixed_acidity :', X_fixed_acidity_variance)
X_quality_variance = math.sqrt(variance(np.array(X_train['alcohol'])))
print('Varaiance of X_quality :', X_quality_variance)
X_volatile_acidity_variance = math.sqrt(variance(np.array(X_train['volatile_acidity'])))
print('Varaiance of volatile_acidity :', X_volatile_acidity_variance)
X_citric_acid_variance = math.sqrt(variance(np.array(X_train['citric_acid'])))
print('Varaiance of citric_acid :', X_citric_acid_variance)
X_residual_sugar_variance = math.sqrt(variance(np.array(X_train['residual_sugar'])))
print('Varaiance of residual_sugar :', X_residual_sugar_variance)
X_chlorides_variance = math.sqrt(variance(np.array(X_train['citric_acid'])))
print('Varaiance of chlorides :', X_chlorides_variance)
X_free_sulfur_dioxide_variance = math.sqrt(variance(np.array(X_train['free_sulfur_dioxide'])))
print('Varaiance of free_sulfur_dioxide :', X_free_sulfur_dioxide_variance)
X_total_sulfur_dioxide_variance =  math.sqrt(variance(np.array(X_train['total_sulfur_dioxide'])))
print('Varaiance of total_sulfur_dioxide :', X_total_sulfur_dioxide_variance)
X_density_variance = math.sqrt(variance(np.array(X_train['density'])))
print('Varaiance of density :', X_density_variance)
X_free_sulfur_dioxide_variance = math.sqrt(variance(np.array(X_train['free_sulfur_dioxide'])))
print('Varaiance of free_sulfur_dioxide :', X_free_sulfur_dioxide_variance)
X_pH_variance = math.sqrt(variance(np.array(X_train['pH'])))
print('Varaiance of pH :', X_pH_variance)
X_sulphates_variance = math.sqrt(variance(np.array(X_train['sulphates'])))
print('Varaiance of sulphates :', X_sulphates_variance)

def correlation(x, y):
    sum = 0
    x_mean = np.mean(x)
    #print('Mean of x and y :', x_mean, y_mean)

    for i in range(len(x)):
        a = (x[i] - x_mean)
        b = (y[i] - y_mean)
        sum += a * b
    return sum/(n-1)


print('Correlation coeffiecient of fixed acidity with label:', (correlation(np.array(X_train['fixed_acidity']), y_train) / (X_fixed_acidity_variance * y_variance)))
print('Correlation coeffiecient of quality with label: ',(correlation(np.array(X_train['alcohol']), y_train) / (X_quality_variance * y_variance)))
print('Correlation coeffiecient of volatile_acidity with label: ',(correlation(np.array(X_train['volatile_acidity']), y_train) / (X_volatile_acidity_variance * y_variance)))
print('Correlation coeffiecient of citric_acid with label: ',(correlation(np.array(X_train['citric_acid']), y_train) / (X_citric_acid_variance * y_variance)))
print('Correlation coeffiecient of residual_sugar with label: ',(correlation(np.array(X_train['residual_sugar']), y_train) / (X_residual_sugar_variance * y_variance)))
print('Correlation coeffiecient of chlorides with label: ',(correlation(np.array(X_train['chlorides']), y_train) / (X_chlorides_variance * y_variance)))
print('Correlation coeffiecient of free_sulfur_dioxide with label: ',(correlation(np.array(X_train['free_sulfur_dioxide']), y_train) / (X_free_sulfur_dioxide_variance * y_variance)))
print('Correlation coeffiecient of density with label: ',(correlation(np.array(X_train['density']), y_train) / (X_density_variance * y_variance)))
print('Correlation coeffiecient of pH with label: ',(correlation(np.array(X_train['pH']), y_train) / (X_pH_variance * y_variance)))
print('Correlation coeffiecient of sulphates with label: ',(correlation(np.array(X_train['sulphates']), y_train) / (X_sulphates_variance * y_variance)))
print('Correlation coeffiecient of total_sulfur_dioxide with label: ',(correlation(np.array(X_train['total_sulfur_dioxide']), y_train) / (X_total_sulfur_dioxide_variance * y_variance)))

print('Correlation coeffiecient of sulphates with label: ',(correlation(np.array(y_train), y_train) / (y_variance * y_variance)))

x = DS[['alcohol', 'volatile_acidity']]
x_train, x_test, Y_train, Y_test = train_test_split(x, y, train_size=0.75, random_state=2)
m = LinearRegression()
m = m.fit(x_train, y_train)
Y_pred = m.predict(x_test)
print('r2: ', r2_score(Y_test,Y_pred))
columns = DS.columns


import scipy
print(scipy.stats.pearsonr(DS['volatile_acidity'], DS['quality'])[0])
sns.heatmap(DS, annot=True )
correlation_coe_matrix = DS.corr().round()
#print(correlation_coe_matrix)
sns.heatmap(correlation_coe_matrix, annot=True)
#plt.show()
#Need to check the heatmap in sns