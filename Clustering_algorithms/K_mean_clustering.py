import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

df = load_iris()
y = df.target
#print(y)
print(df.target_names)
x = df.data
print(type(x))
print(x.shape)
km = KMeans(n_clusters=3, max_iter=100)
km.fit(x)
from sklearn.metrics import confusion_matrix, silhouette_score, homogeneity_score, completeness_score
cm = confusion_matrix(y, km.labels_)
print('Confusion Matrix')
print(cm)

print('Silhoutte score: ')
print(round(silhouette_score(x, km.labels_, metric='euclidean'),3))
print(km.predict(x))
print('homogeneity score: ', homogeneity_score(y, km.predict(x)))
print('completeness score: ', completeness_score(y, km.predict(x)))
print('****')
print(km.labels_)
print(km.n_clusters)
print(km.n_iter_)
print(km.inertia_)
print(km.cluster_centers_)
print()

inertia = []
sil_score = []
for k in range(2, 10):
    km = KMeans(n_clusters=k, max_iter=300)
    km.fit(x)
    print('For k {}, Silhouette-score {}, Inertia: {}'.format(k , round(silhouette_score(x, km.labels_),3), km.inertia_))
    inertia.append(km.inertia_)
    print('homogeneity score: ', homogeneity_score(y, km.predict(x)))
    print('completeness score: ', completeness_score(y, km.predict(x)))
    sil_score.append( round(silhouette_score(x, km.labels_),3))
print(inertia)
print(sil_score)

plt.plot(range(2,10),inertia)
plt.xlabel('No of Clusters')
plt.title('Optimal value of K')
plt.show()

plt.plot(range(2,10), sil_score)
plt.show()