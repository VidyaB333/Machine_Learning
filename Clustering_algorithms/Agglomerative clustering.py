import matplotlib
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs

from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier

colors = ['red', 'green', 'yellow', 'blue', 'purple', 'black', 'pink', 'magenta']
x, y = make_blobs(n_samples=1000, n_features=2, centers=8, cluster_std=1, random_state=1)
print(type(x))
df = pd.DataFrame(x)


##Dendogram
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram

distance_matix = pdist(df[[0,1]])
xl = linkage(distance_matix, method='ward')
dend = dendrogram(xl)
plt.title('Dendrogram')
plt.show()

ac = AgglomerativeClustering(n_clusters=8, linkage='average')
y_pred = ac.fit_predict(x)
df['Agglomerative_labels'] = y_pred
print(type(y_pred))
y_p = pd.Series(y_pred)
print('Silhouette score for Agglomerative clustering (complete linkage) : %.3f'%(silhouette_score(x, y_pred)))
print('Adjusted rand score: %.3f'%adjusted_rand_score(y, df['Agglomerative_labels']))

plt.figure(figsize=(15,10))
plt.subplot(1,2,1)
plt.scatter(x[:,0], x[:,1], c='green', s=8)
plt.title('Dataset')
plt.subplot(1,2,2)
plt.scatter(df[0], df[1], c=df['Agglomerative_labels'],cmap=matplotlib.colors.ListedColormap(colors),s=8)
plt.title('Clusters')
plt.show()



# K distance graph
KNN=KNeighborsClassifier(n_neighbors=2, metric='euclidean')
KNN = KNN.fit(df[[0,1]], y)
distance, indices = KNN.kneighbors(df[[0,1]])

distance = distance[:, 1]
distance= np.sort(distance, axis=0)

plt.plot(range(3000), distance)
plt.grid()
plt.title('K distance graph', fontsize=20)
plt.xlabel('Datapoints', fontsize=14)
plt.ylabel('Distance between nearest neightbor')
plt.show()

#DBSCAN Clustering algorithm
dc = DBSCAN(eps=0.4, min_samples=5, metric='euclidean')
y_dbscan = dc.fit_predict(df[[0,1]])
df['DBSCAN_Labels'] = y_dbscan
print('Silhouette score for DBSCAN: %.3f'%(silhouette_score(x, y_dbscan)))
print('Adjusted rand score: %.3f'%adjusted_rand_score(y, df['DBSCAN_Labels']))

plt.figure(figsize=(14,10))
plt.title('DBSCAN')
plt.subplot(1,2,1)
plt.scatter(x[:,0], x[:,1], c='green', s=8)
plt.title('Dataset')
plt.subplot(1,2,2)
plt.scatter(df[0], df[1], c=df['DBSCAN_Labels'],cmap=matplotlib.colors.ListedColormap(colors),s=8)
plt.title('Clusters')
plt.show()


#KMean clustering
from sklearn.cluster import KMeans
KM = KMeans(n_clusters=8,max_iter=300)
y_KMEAN = KM.fit_predict(df[[0,1]])
df['KMEAN_labels'] =  y_KMEAN
print('Silhouette score of KMean clustering: ', silhouette_score(df[[0,1]], y_KMEAN))
print('Adjusted rand score: %.3f'%adjusted_rand_score(y, df['KMEAN_labels']))

plt.figure(figsize=(14,10))
plt.title('KMEAN clusteing')
plt.subplot(1,2,1)
plt.scatter(x[:,0], x[:,1], c='green', s=8)
plt.title('Dataset')
plt.subplot(1,2,2)
plt.scatter(df[0], df[1], c=df['KMEAN_labels'],cmap=matplotlib.colors.ListedColormap(colors),s=8)
plt.title('KMEAN Clusters')
plt.show()