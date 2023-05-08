# -*- coding: utf-8 -*-
"""
Created on Wed May 16 14:45:35 2018

@author: Leo
"""

# clustering

# K-means
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

os.chdir("J:\DSDegree\PennState\DAAN_862\Week 11\Python Files")
iris = pd.read_csv('iris.csv')
iris.species = iris.species.astype("category").cat.codes

# Rescale variables
scaler = MinMaxScaler()
X = scaler.fit_transform(iris.iloc[:, 0:4])
y = iris.species

kmeans = KMeans(n_clusters = 3, random_state= 130)
y_pred = kmeans.fit_predict(X)

metrics.homogeneity_score(y, y_pred)
metrics.completeness_score(y, y_pred)
metrics.adjusted_rand_score(y, y_pred)
metrics.silhouette_score(X, y_pred, metric = 'euclidean')

centers = kmeans.cluster_centers_
centers_pl = centers[:, 2]
centers_pw = centers[:, 3]



plt.figure(figsize = (12, 6))
 # create a plot with two subplots and plot the first subplot
plt.subplot(121)
# Scatter plot of petal_length and petal_width,  color represents
# the original categories (c = y).
plt.scatter(X[:, 2], X[:, 3], c = y)
plt.title('iris data')
plt.xlabel('Petal length')
plt.ylabel('Petal width')
# Plot the second subplot.
plt.subplot(122)
# Scatter plot of petal_length and petal_width, the color represents the
# predicted  categories (c = y_pred).
plt.scatter(X[:, 2], X[:, 3], c = y_pred, label = 'Predicted')
# PLot the cluster centroids with 'X' and color is red
plt.scatter(centers_pl, centers_pw, s= 100, c = 'r', 
            marker = 'x', label = 'Cluster Centers')
plt.title('Clustering results')
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.legend()
# adjust the space between two subplots
plt.subplots_adjust(wspace = 0.2)

# try different n_clusters

result = []
nclusters = range(2, 7)
for n in nclusters:
    y_pred_temp = KMeans(n_clusters = n).fit_predict(X)  
    score = metrics.silhouette_score(X, y_pred_temp,metric = 'euclidean')
    result.append((n, score))

result



###############################################################################
# Hierarchical Clustering
from sklearn.cluster import AgglomerativeClustering

# average linkage
Hier_y_pred1 = AgglomerativeClustering(
        n_clusters = 3, 
        affinity = 'euclidean', 
        linkage = 'average').fit_predict(X)

metrics.homogeneity_score(y, Hier_y_pred1)
metrics.completeness_score(y, Hier_y_pred1)
metrics.adjusted_rand_score(y, Hier_y_pred1)
metrics.silhouette_score(X, Hier_y_pred1, metric = 'euclidean')

#Visualize the Dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage
Z_avg = linkage(X, method = 'average')
plt.figure()
den = dendrogram(Z_avg, leaf_font_size = 8)

# complete linkage
Hier_y_pred2 = AgglomerativeClustering(
        n_clusters = 3, 
        affinity = 'euclidean', 
        linkage = 'complete').fit_predict(X)

metrics.homogeneity_score(y, Hier_y_pred2)
metrics.completeness_score(y, Hier_y_pred2)
metrics.adjusted_rand_score(y, Hier_y_pred2)
metrics.silhouette_score(X, Hier_y_pred2, metric = 'euclidean')

# Use ward
Hier_y_pred3 = AgglomerativeClustering(
        n_clusters = 3, 
        affinity = 'euclidean', 
        linkage = 'ward').fit_predict(X)

metrics.homogeneity_score(y, Hier_y_pred3)
metrics.completeness_score(y, Hier_y_pred3)
metrics.adjusted_rand_score(y, Hier_y_pred3)
metrics.silhouette_score(X, Hier_y_pred2, metric = 'euclidean')

# Plot the iris data colored by original categories.
plt.figure(figsize = (12, 12))
plt.subplot(221)
plt.scatter(X[:, 2], X[:, 3], c = y)
plt.title('iris data')
plt.xlabel('Petal length')
plt.ylabel('Petal width')

# Plot the iris data colored by the average linkage prediction.
plt.subplot(222)
plt.scatter(X[:, 2], X[:, 3], c = Hier_y_pred1)
plt.title('Average linkage')
plt.xlabel('Petal length')
plt.ylabel('Petal width')

# Plot the iris data colored by the complete linkage prediction.
plt.subplot(223)
plt.scatter(X[:, 2], X[:, 3], c = Hier_y_pred2)
plt.title('Complete linkage')
plt.xlabel('Petal length')
plt.ylabel('Petal width')

# Plot the iris data colored by the ward linkage prediction.
plt.subplot(224)
plt.scatter(X[:, 2], X[:, 3], c = Hier_y_pred3)
plt.title('Ward linkage')
plt.xlabel('Petal length')
plt.ylabel('Petal width')

plt.subplots_adjust(wspace = 0.2)



#############################################################################
# DB-scan
from sklearn.cluster import DBSCAN
db_y_pred = DBSCAN(eps = 0.2, min_samples =5).fit_predict(X)

metrics.homogeneity_score(y, db_y_pred)
metrics.completeness_score(y, db_y_pred)
metrics.adjusted_rand_score(y, db_y_pred)
metrics.silhouette_score(X, db_y_pred, metric = 'euclidean')

# manually grid search for eps and min_samples
result = []
epses = [ 0.1, 0.2, 0.3, 0.4]
min_samples = [5, 10, 15]
for v in epses:
    for n in min_samples:
        model = DBSCAN(eps = v, min_samples = n)  
        y_pred_temp = model.fit_predict(X)
        # find the number of clusters
        n_clusters = np.unique(model.labels_).size
        score = metrics.silhouette_score(X, y_pred_temp,metric = 'euclidean')
        result.append((v, n, score, n_clusters))

result


