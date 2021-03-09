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
print(x.head(5))
k = 3
Centroid = x.sample(n=k)
print(Centroid)
c = np.array(Centroid)
print('Initial Centroids: \n', c)
plt.scatter(df['ApplicantIncome'], df['LoanAmount'])
plt.scatter(Centroid['ApplicantIncome'], Centroid['LoanAmount'], c='red')
plt.xlabel('Annual Income')
plt.ylabel('Loan amount')
plt.title('scatter plot')
plt.show()
x = np.array(x)
No_change_flag = False


def centroid_display_plotting(c, cluster0, cluster1, cluster2):
    print(c)
    x = [c[i][0] for i in range(k)]
    y = [c[i][1] for i in range(k)]
    # plt.scatter(df['ApplicantIncome'], df['LoanAmount'])
    x_0 = [cluster0[i][0] for i in range(len(cluster0))]
    y_0 = [cluster0[i][1] for i in range(len(cluster0))]
    plt.scatter(x_0, y_0, c='black')
    plt.scatter([cluster1[i][0] for i in range(len(cluster1))], [cluster1[i][1] for i in range(len(cluster1))],
                c='yellow')
    plt.scatter([cluster2[i][0] for i in range(len(cluster2))], [cluster2[i][1] for i in range(len(cluster2))],
                c='blue')
    plt.scatter(x, y, c='red')
    plt.xlabel('Annual Income')
    plt.ylabel('Loan amount')
    plt.show()


# Function for calculating distance between centroid and data points
def distance(x1, x2):
    # print('points are : ', x1, x2 )
    dis = math.sqrt((x2[0] - x1[0]) ** 2 + (x2[1] - x1[1]) ** 2)
    return dis


def distance_between_centroid_and_point(c):
    cluster_0 = []
    cluster_1 = []
    cluster_2 = []
    for i in range(x.shape[0]):

        d1 = distance(x[i], c[0])
        d2 = distance(x[i], c[1])
        d3 = distance(x[i], c[2])
        # print('Distances for {} sample {}, {}, {}'.format(i, d1, d2,d3))

        if min(d1, d2, d3) == d1:
            # print('in clustor 0')
            cluster_0.append(x[i])
        elif min(d1, d2, d3) == d2:
            # print('in clustor 1')
            cluster_1.append(x[i])
        else:
            # print('In cluster 2')
            cluster_2.append(x[i])
        # print(cluster_0)
        # print(cluster_1)
        # print(cluster_2)

    print('Elements in each Clusters : ', len(cluster_0), len(cluster_1), len(cluster_2))

    return cluster_0, cluster_1, cluster_2


def method_for_new_centroid(cluster_0, cluster_1, cluster_2, c):
    print('In method for calculating new centroid')
    # print(len(cluster_0), len(cluster_1), len(cluster_2))
    # c0
    x0 = y0 = x1 = y1 = x2 = y2 = 0
    for i in range(len(cluster_0)):
        x0 += (cluster_0[i][0])
        y0 += (cluster_0[i][1])
    x0 = x0 / len(cluster_0)
    y0 = y0 / len(cluster_0)
    new_c0 = np.array([x0, y0])

    # c1
    for i in range(len(cluster_1)):
        x1 += cluster_1[i][0]
        y1 += cluster_1[i][1]
    x1 = x1 / len(cluster_1)
    y1 = y1 / len(cluster_1)
    new_c1 = np.array([x1, y1])

    # c2
    for i in range(len(cluster_2)):
        x2 += cluster_2[i][0]
        y2 += cluster_2[i][1]
    x2 = x2 / len(cluster_2)
    y2 = y2 / len(cluster_2)
    new_c2 = np.array([x2, y2])

    print('New Centroids: ', new_c0, new_c1, new_c2)
    print('Old Centroids: ', c[0], c[1], c[2])
    centroid_l = list([new_c0, new_c1, new_c2])
    centroid_display_plotting(centroid_l, cluster_0, cluster_1, cluster_2)

    if ((c[0][0] == new_c0[0]) & (c[0][1] == new_c0[1])) & ((c[1][0] == new_c1[0]) & (c[1][1] == new_c1[1])) & (
            (c[2][0] == new_c2[0]) & (c[2][1] == new_c2[1])):
        print('No change in  centroids')
        No_change_flag = True
        return new_c0, new_c1, new_c2, No_change_flag

    return new_c0, new_c1, new_c2


max_iter = 100
for i in range(max_iter):
    print()
    # print('{} th iteration'.format(i))
    clusters = distance_between_centroid_and_point(c)
    c = method_for_new_centroid(*clusters, c)
    if len(c) > 3:
        print('No of Iteration used for stable centroid: ', i)
        break

