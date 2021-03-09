import pandas as pd
from sklearn.datasets import make_blobs
from matplotlib import  pyplot as plt
import  seaborn as sns

x, y= make_blobs(n_features=2, n_samples=10, centers=2,cluster_std=1 )
print(x)
print(y)

plt.scatter(x[:, 0], x[:, 1])
plt.show()

#distance metrix
from scipy.spatial.distance import  pdist
xm = pdist(x, metric='euclidean')
#print(xm)

xm = [round(i, 3) for i in xm]
print(xm)

from scipy.cluster.hierarchy import linkage
xl = linkage(xm, method='ward')
print(xl)

from scipy.cluster.hierarchy import dendrogram
xd = dendrogram(xl)
print(xd)
plt.show()

"""
{'icoord': [[35.0, 35.0, 45.0, 45.0], [25.0, 25.0, 40.0, 40.0], [15.0, 15.0, 32.5, 32.5],
            [5.0, 5.0, 23.75, 23.75], [55.0, 55.0, 65.0, 65.0], [85.0, 85.0, 95.0, 95.0],
            [75.0, 75.0, 90.0, 90.0], [60.0, 60.0, 82.5, 82.5], [14.375, 14.375, 71.25, 71.25]],
 'dcoord': [[0.0, 0.237, 0.237, 0.0], [0.0, 1.3519259101987309, 1.3519259101987309, 0.237],
            [0.0, 1.7971312788255989, 1.7971312788255989, 1.3519259101987309],
            [0.0, 3.1732306093317577, 3.1732306093317577, 1.7971312788255989],
            [0.0, 1.885, 1.885, 0.0], [0.0, 1.325, 1.325, 0.0],
            [0.0, 2.6957057826600193, 2.6957057826600193, 1.325],
            [1.885, 4.027473591885281, 4.027473591885281, 2.6957057826600193],
            [3.1732306093317577, 15.731759806200959, 15.731759806200959, 4.027473591885281]],
 'ivl': ['0', '1', '9', '3', '4', '2', '8', '7', '5', '6'],
 'leaves': [0, 1, 9, 3, 4, 2, 8, 7, 5, 6],
 'color_list': ['C1', 'C1', 'C1', 'C1', 'C2', 'C2', 'C2', 'C2', 'C0']}

"""