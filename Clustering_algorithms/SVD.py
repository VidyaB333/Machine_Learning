import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from sklearn.datasets import load_digits
digits =  load_digits()
x =  digits.data
y = digits.target

print(x.shape)
U, Sigma, VT = randomized_svd(x, n_components=15, n_iter=300, random_state=12)
print('Left Singular matrix shape: ', U.shape)
print('Sigma matrix: ', Sigma.shape)
print('Right Singular matrix: ', VT.shape)


#Total variance explained by mentioned components
svd =TruncatedSVD(n_components = 15, n_iter=300, random_state =12)
x_reduced = svd.fit_transform(x)
print(x_reduced.shape)

print('Total variance explained by 15 components : ', svd.explained_variance_ratio_.sum())

print(Sigma)
variance =[]
componets =30
for i in range(componets):
    svd = TruncatedSVD(n_components=i+1, n_iter=300, random_state=12)
    x_reduced = svd.fit_transform(x)
    variance.append(svd.explained_variance_ratio_.sum())
from matplotlib import pyplot as plt
plt.plot(range(1,31), variance, 'r')
plt.plot(range(1,31), variance, 'bs')
plt.grid()
plt.title('Singular value Decomposition')
plt.xlabel('No of principle components')
plt.show()
