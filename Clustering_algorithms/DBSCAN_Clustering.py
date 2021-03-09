import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import DBSCAN

np.random.seed(42)


def PointsInCircum(r, n=100):
    return [(math.cos(2 * math.pi / n * x) * r + np.random.normal(-30, 30),
             math.sin(2 * math.pi / n * x) * r + np.random.normal(-30, 30)) for x in range(1, n + 1)]


# Creating data points in the form of a circle
df = pd.DataFrame(PointsInCircum(500, 1000))
df = df.append(PointsInCircum(300, 700))
df = df.append(PointsInCircum(100, 300))

# Adding noise to the dataset
df = df.append([(np.random.randint(-600, 600), np.random.randint(-600, 600)) for i in range(300)])
#print(df.shape)
#print(df.head(2))
"""
plt.figure(figsize=(10,10))
plt.scatter(df[0], df[1], s=7)
plt.title('Dataset', fontsize = 20)
plt.xlabel('Feature 1', fontsize = 14)
plt.ylabel('Feature 2', fontsize = 14)
plt.show()

from sklearn.cluster import KMeans, DBSCAN
k_means=KMeans(n_clusters=4,random_state=42)
k_means.fit(df[[0,1]])
df['KMeans_labels']= k_means.labels_

#Plotting resulting clusters
colors = ['red', 'green', 'yellow', 'purple']
plt.figure(figsize=(10,10))
plt.scatter(df[0],df[1],c=df['KMeans_labels'],cmap=matplotlib.colors.ListedColormap(colors),s=15)
plt.title('K-Means Clustering',fontsize=20)
plt.xlabel('Feature 1',fontsize=14)
plt.ylabel('Feature 2',fontsize=14)
plt.show()


#Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=4, affinity='euclidean')
model.fit(df[[0,1]])
df['HR_labels']=model.labels_

# Plotting resulting clusters
plt.figure(figsize=(10,10))
plt.scatter(df[0],df[1],c=df['HR_labels'],cmap=matplotlib.colors.ListedColormap(colors),s=15)
plt.title('Hierarchical Clustering',fontsize=20)
plt.xlabel('Feature 1',fontsize=14)
plt.ylabel('Feature 2',fontsize=14)
plt.show()



#DBSCAN Clustering
dbscan = DBSCAN()
dbscan.fit(df[[0,1]])
print(dbscan)

df['DBSCAN_labels']=dbscan.labels_

# Plotting resulting clusters
plt.figure(figsize=(10,10))
plt.scatter(df[0],df[1],c=df['DBSCAN_labels'],cmap=matplotlib.colors.ListedColormap(colors),s=15)
plt.title('DBSCAN Clustering',fontsize=20)
plt.xlabel('Feature 1',fontsize=14)
plt.ylabel('Feature 2',fontsize=14)
plt.show()

"""
#K distance Graph
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(df[[0,1]])
distances, indices = nbrs.kneighbors(df[[0,1]])
#print(distances)
#print(indices)

distances = np.sort(distances,  axis=0)
distances = distances[:,1]
#plt.figure(figsize=(10,10))
#plt.plot(distances)
#plt.title('K-distance Graph',fontsize=20)
#plt.xlabel('Data Points sorted by distance',fontsize=14)
#plt.ylabel('Epsilon',fontsize=14)
#plt.show()

colors = ['red', 'green', 'yellow', 'purple']
dbscan_opt= DBSCAN(eps=30, n_jobs=-1, min_samples=6)
dbscan_opt.fit(df[[0,1]])
df['DBSCAN_opt_labels'] = dbscan_opt.labels_
print(df['DBSCAN_opt_labels'].value_counts())

# Plotting the resulting clusters
plt.figure(figsize=(10,10))
plt.scatter(df[0],df[1],c=df['DBSCAN_opt_labels'],cmap=matplotlib.colors.ListedColormap(colors),s=15)
plt.title('DBSCAN Clustering',fontsize=20)
plt.xlabel('Feature 1',fontsize=14)
plt.ylabel('Feature 2',fontsize=14)
plt.show()

from sklearn.metrics import silhouette_score, completeness_score
print('Performance metices of DBSCAN clustering')
print('Number of clusters: ',len(df['DBSCAN_opt_labels'].value_counts()))
print('Number of of outliers: ',df[df['DBSCAN_opt_labels']==-1].shape[0])
outliers = df[df['DBSCAN_opt_labels']==-1]
print(outliers.shape)
plt.figure(figsize=(17,15))
plt.subplot(1,2,1)
plt.scatter(outliers[0], outliers[1], s=10, c='purple')
plt.title('OUTLIERS scatter plot')

info = df[df['DBSCAN_opt_labels']!=-1]
print(info.shape)
plt.subplot(1,2,2)
plt.scatter(info[0], info[1], s=10, c='red')
plt.title('Info scatter plot')
plt.show()


print()
print('Silhoutte score: ', silhouette_score(df[[0,1]], dbscan_opt.labels_))
"""
dbscan_opt= DBSCAN(eps=10, n_jobs=-1, min_samples=6) #More noise will be there
dbscan_opt.fit(df[[0,1]])
df['DBSCAN_opt_labels'] = dbscan_opt.labels_
print(df['DBSCAN_opt_labels'].value_counts())

# Plotting the resulting clusters
plt.figure(figsize=(10,10))
plt.scatter(df[0],df[1],c=df['DBSCAN_opt_labels'],cmap=matplotlib.colors.ListedColormap(colors),s=15)
plt.title('DBSCAN Clustering_ less eps',fontsize=20)
plt.xlabel('Feature 1',fontsize=14)
plt.ylabel('Feature 2',fontsize=14)
plt.show()

dbscan_opt= DBSCAN(eps=60, n_jobs=-1, min_samples=6)# Smller clusters grouped into a bigger clusers
dbscan_opt.fit(df[[0,1]])
df['DBSCAN_opt_labels'] = dbscan_opt.labels_
print(df['DBSCAN_opt_labels'].value_counts())

# Plotting the resulting clusters
plt.figure(figsize=(10,10))
plt.scatter(df[0],df[1],c=df['DBSCAN_opt_labels'],cmap=matplotlib.colors.ListedColormap(colors),s=15)
plt.title('DBSCAN Clustering more esplon',fontsize=20)
plt.xlabel('Feature 1',fontsize=14)
plt.ylabel('Feature 2',fontsize=14)
plt.show()




dbscan_opt= DBSCAN(eps=30, n_jobs=-1, min_samples=60)#more noise
dbscan_opt.fit(df[[0,1]])
df['DBSCAN_opt_labels'] = dbscan_opt.labels_
print(df['DBSCAN_opt_labels'].value_counts())

# Plotting the resulting clusters
plt.figure(figsize=(10,10))
plt.scatter(df[0],df[1],c=df['DBSCAN_opt_labels'],cmap=matplotlib.colors.ListedColormap(colors),s=15)
plt.title('DBSCAN Clustering more minpoints',fontsize=20)
plt.xlabel('Feature 1',fontsize=14)
plt.ylabel('Feature 2',fontsize=14)
plt.show()
dbscan_opt= DBSCAN(eps=30, n_jobs=-1, min_samples=3)# Smller clusters grouped into a bigger clusers
dbscan_opt.fit(df[[0,1]])
df['DBSCAN_opt_labels'] = dbscan_opt.labels_
print(df['DBSCAN_opt_labels'].value_counts())

# Plotting the resulting clusters
plt.figure(figsize=(10,10))
plt.scatter(df[0],df[1],c=df['DBSCAN_opt_labels'],cmap=matplotlib.colors.ListedColormap(colors),s=15)
plt.title('DBSCAN Clustering less minpoints',fontsize=20)
plt.xlabel('Feature 1',fontsize=14)
plt.ylabel('Feature 2',fontsize=14)
plt.show()

"""