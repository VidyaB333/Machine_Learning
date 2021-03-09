from sklearn.datasets import make_regression
from sklearn.feature_selection import f_regression
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot

#X, y=make_classification(n_samples=100, n_features=20, n_informative=2,
#                                    n_classes=2, n_clusters_per_class=2, weights=None)


##Generate Dataset
X, y, coef= dataset = make_regression(n_samples=50, n_features=5, n_informative=3, bias=0.2, coef=True)
print(type(dataset))
print(len(dataset))
print(X.shape)
print(type(X))
print(coef)
print('Before selecting the best features')
linear_r = LinearRegression()
linear_r.fit(X, y)
print(linear_r.score(X,y))
print(linear_r.intercept_)
print('Coeff:', linear_r.coef_)
#pyplot is applicable when x and y have same size
#pyplot.scatter(X, y)
#pyplot.show()
print('X before feature selection:', X[0:5, :])


print('Fature selection method from sklearn library')
Best_features = SelectKBest(score_func=f_regression, k=2)
print(type(Best_features))
X_new = Best_features.fit_transform(X, y)
print(type(X_new))
print(X_new.shape)
print('X after feature SElection method ')
print(X_new[0:5, :])
indices = [i for i in range(X.shape[0])]
print(indices)
print('after ')
X_new = pd.DataFrame(X_new,index=indices, columns=['X1', 'X2'])
print(type(X_new))
print(X_new.head(5))

print(X[:,0])
print('ffffffffffffffffffff')
X1 = pd.DataFrame(X[:,0],index=indices, columns=['X1'])
print(type(X1))

X_D = pd.DataFrame(X, index=indices, columns=['XA','XB','XC','XD','XE'])

print(X_D.shape)
print(type(X_D))
                                     
y_D = pd.DataFrame(y, index= indices, columns=['YA'])
print(y_D.shape)

DATA = pd.concat([X_D, y_D], axis=1)
print(DATA.shape)
print(type(DATA))
print(DATA.head(5))

linear_r.fit(X_D, y_D)
print(linear_r.score(X_D, y_D))
print(linear_r.intercept_)
print('Coeff:', linear_r.coef_)