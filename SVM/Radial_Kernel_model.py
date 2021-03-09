import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs, make_circles
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


#Dataset for creating circles
n_samples = 500
x, y = make_circles(n_samples=n_samples, noise=0.1)
print(x.shape)
print(y.shape)


l = list(x[:, 0])
l1 = list(x[:, 1])
#plt.scatter(l, l1, color='g', s=1)
# plt.show()
#plt.scatter(l, l1, color='b', s=2)
# plt.show()


print('Using Ravel')
print('Before: ', y.shape, y.ndim)
y = y.ravel()
print('After : ',y.shape, y.ndim)
y1 = y.reshape(-1, 1)
print('After reshape: ', y1.shape, y1.ndim)


m = []
for i, j in zip(l, l1):
    m.append((i, j))
print(m[0:3])

x_train, x_test, y_train, y_test = train_test_split(m, y1, train_size=0.7)

print(type(x))



from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
svm = SVC(C=1.0, kernel='rbf')

svm = svm.fit(x_train, y_train)
y_train_pred = svm.predict(y_train)
print('training accuracy: ', accuracy_score(y_train, y_train_pred))
print('training confusionm: \n', confusion_matrix(y_train, y_train_pred))
print('training classification: \n', classification_report(y_train, y_train_pred))

y_test_pred = svm.predict(y_test)
print('testing accuracy: ', accuracy_score(y_test, y_test_pred))
print('testing confusionm: \n', confusion_matrix(y_test, y_test_pred))
print('testing classification: \n', classification_report(y_test, y_test_pred))
