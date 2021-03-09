import math

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, completeness_score
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv(r'C:\Users\vidya\OneDrive\Desktop\Python_coding_practice_Datasets\PythonDataSets\KNN\clustering.csv')
print(df.shape)
x = df[['ApplicantIncome', 'LoanAmount']]
km = KMeans(n_clusters=3)
km.fit(x)
print(km)
#print(km.predict(x))
print(km.labels_)
print(km.cluster_centers_)
inertia = []
silhouette = []
for k in range(2, 10):
    km = KMeans(n_clusters=k)
    km.fit(x)
    inertia.append(km.inertia_)
    silhouette.append(silhouette_score(x, km.labels_))
print(inertia)
print(silhouette)
plt.plot(range(2, 10), inertia)
plt.show()

plt.plot(range(2, 10), silhouette)
plt.show()