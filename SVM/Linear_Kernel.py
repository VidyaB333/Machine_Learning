import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs, make_classification
import seaborn as sns
from matplotlib import pyplot as plt

n_samples = 500

x, y = make_classification(n_samples=n_samples, n_features=2,n_informative=2,n_redundant=0,n_clusters_per_class=1)
print(x.shape)
print(y.shape)

sns.scatterplot(x = x[:,0],y=y, color ='g', size=5)
sns.scatterplot(x=x[:,1],y=y, color = 'r',size=1)
plt.show()
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
"""
accuries = cross_val_score(SVC(kernel='linear'), x, y, cv=5)
print(type(accuries))
print(accuries)
print(accuries.mean())
print()
"""
SVM =SVC(kernel='linear')
SVM = SVM.fit(x,y)
print('suppoert vectors')
sv =SVM.support_vectors_
print(sv)
print(len(sv))

sv = list(sv)
print(sv)
l = []
l1 =[]
for i in range(len(sv)):
    print(sv[i][0],sv[i][1])
    l.append(sv[i][0])
    l1.append(sv[i][1])

print(l)
print(l1)

plt.scatter(l,l1 )