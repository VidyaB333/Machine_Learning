

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist


x, y= make_blobs(n_samples=1000, n_features=2, centers=4, cluster_std =2)
df = pd.DataFrame(x)
print(df.head(2))



plt.scatter(df[0], df[1], c='red', s=10)
plt.title('Dataset')
plt.show()





distance_matrix = pdist(df[[0,1]])




plt.figure(figsize=(10,10))
xl = linkage(distance_matrix, method='complete')
x_complete = dendrogram(xl)
plt.title('Using Complete method')
plt.show()


# In[26]:


plt.figure(figsize=(10,10))
xl = linkage(distance_matrix, method='single')
x_complete = dendrogram(xl)
plt.title('Using single method')
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
xl = linkage(distance_matrix, method='average')
x_complete = dendrogram(xl)
plt.title('Using average method')
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
xl = linkage(distance_matrix, method='ward')
x_complete = dendrogram(xl)
plt.title('Using ward method')
plt.show()

