import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, validation_curve, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv(
    'C:\\Users\\vidya\\OneDrive\\Desktop\\Python_coding_practice_Datasets\\PythonDataSets\\Learning_Curves\\electrial_power.csv')
print(data.shape)
print(data.head())
# sns.displot(data['PE'])
# plt.plot()
print(data.isna().count())  # No nA values
col = data.columns

# For standardization of datapoints
for i in range(4):
    x = data[data.columns[i]]
    # sns.displot(x)
    # plt.plot()
    # print(x.shape)
    x = np.array(x)
    # print(np.min(x), np.max(x))
    x = x - np.min(x)
    # print(x[0:10])
    x = (x / np.max(x) - np.min(x))
    # print(x[0:10])
    data[col[i]] = pd.DataFrame(x)
    # print(data[col[i]].head(5))
print(data.head(5))

#sns.pairplot(data)
#plt.title('Scatter plot and distribution plot')
#plt.show()

correlation_cofficients = data.corr()
#sns.heatmap(correlation_cofficients, annot=True)
#plt.title('Correlation coefficients')
#plt.show()

x = data[col[0:-1]]
print(x.shape)
y = data[col[-1]]
print(y.shape)

# Training and validation set
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=12)
print(x_train.shape)
print(x_test.shape)
"""
train_size = [1, 5, 10, 50, 100, 500, 2000, 5000, 7654]
print('Linear Regression')
train_sizes, train_scores, validation_scores = learning_curve(estimator=LinearRegression(),
                                                              X=x,
                                                              y=y,
                                                              train_sizes=[0.1, 0.33, 0.55, 0.78, 1.0],
                                                              cv=5, scoring='neg_mean_squared_error'
                                                              )
print(train_sizes)
print(train_scores)
print('Validation error')
print(validation_scores)
print(type(train_scores))
print(train_scores.shape)
trains = []
validation = []
for i in range(len(train_sizes)):
    mean = np.mean(train_scores[i])
    trains.append(abs(mean))
    mean_val = np.mean(validation_scores[i])
    validation.append(abs(mean_val))
    print(trains[i], validation[i])
    print()

print('Mean errors of training set: ', trains)
print('Mean errors on validation set: ', validation)

print('Mean training scores\n\n', pd.Series(trains, index=train_sizes))
print('\n', '-' * 20)  # separator
print('\nMean validation scores\n\n', pd.Series(validation, index=train_sizes))
sns.lineplot(x=train_sizes, y=trains, color='g', label='Traininf_error')
sns.lineplot(x=train_sizes, y=validation, color='b', label='Validation_Error')
plt.title('Learning curves', fontsize=18, y=1.03)
plt.legend()
plt.ylabel('r2', fontsize=14)
plt.xlabel('Training set size', fontsize=14)

# plt.xlim(0,1000)
plt.grid(True, color='y')
plt.show()

print('\n\n\n')

print('Random Foest Regression')
train_sizes, train_scores, validation_scores = learning_curve(estimator=RandomForestRegressor(max_leaf_nodes=200),
                                                              X=x,
                                                              y=y,
                                                              train_sizes=[0.1, 0.33, 0.55, 0.78, 1.0],
                                                              cv=5, scoring='neg_mean_squared_error'
                                                              )
print(train_sizes)
print(train_scores)
print('Validation error')
print(validation_scores)
print(type(train_scores))
print(train_scores.shape)
trains = []
validation = []
for i in range(len(train_sizes)):
    mean = np.mean(train_scores[i])
    trains.append(abs(mean))
    mean_val = np.mean(validation_scores[i])
    validation.append(abs(mean_val))
    print(trains[i], validation[i])
    print()

print('Mean errors of training set: ', trains)
print('Mean errors on validation set: ', validation)

print('Mean training scores\n\n', pd.Series(trains, index=train_sizes))
print('\n', '-' * 20)  # separator
print('\nMean validation scores\n\n', pd.Series(validation, index=train_sizes))
sns.lineplot(x=train_sizes, y=trains, color='g', label='Traininf_error')
sns.lineplot(x=train_sizes, y=validation, color='b', label='Validation_Error')
plt.title('Learning curves', fontsize=18, y=1.03)
plt.legend()
plt.ylabel('r2', fontsize=14)
plt.xlabel('Training set size', fontsize=14)

# plt.xlim(0,1000)
plt.grid(True, color='y')
plt.show()

"""
print('Checking the Validation curves')
train_scores_v, test_scores_v = validation_curve(estimator=RandomForestRegressor(),
                                             X=x,
                                             y=y,
                                             param_name='n_estimators',
                                             param_range=[50,100,200,300,400,500,600,1000],
                                             cv=5,scoring='neg_mean_squared_error',n_jobs=-1)

print(train_scores_v)
print(test_scores_v)
val_train =[]
val_test =[]
for i in range(8):
    mean = np.mean(train_scores_v[i])
    val_train.append(abs(mean))
    mean_val = np.mean(test_scores_v[i])
    val_test.append(abs(mean_val))
    print(i)
    print(val_train, val_test)
    print(val_train[i], val_test[i])

    print()


tree_size = [50,100,200,300,400,500,600,1000]
print('Mean errors of training set: ', val_train)
print('Mean errors on validation set: ', val_test)

train_d = pd.Series(val_train, index=tree_size)
test_d = pd.Series(val_test, index=tree_size)

print('Mean training scores\n\n', train_d)
print('\n', '-' * 20)  # separator
print('\nMean validation scores\n\n', test_d)
plt.plot(train_d,color='g', label='Training_error')
plt.plot(test_d, color='b', label='Validation_Error')
#sns.lineplot(x=8, y=val_train, color='g', label='Training_error')
#sns.lineplot(x=8, y=val_test, color='b', label='Validation_Error')

plt.title('Validation curves', fontsize=18, y=1.03)
plt.legend()
plt.ylabel('neg_mean_squared_error', fontsize=14)
plt.xlabel('Tree size', fontsize=14)
plt.show()

print()