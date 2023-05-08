# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 10:42:10 2022

@author:Brandon Botzer btb5103


Data Set Information:

The examined group comprised kernels belonging to three different varieties 
of wheat: Kama, Rosa and Canadian, 70 elements each, randomly selected for 
the experiment. High quality visualization of the internal kernel structure 
was detected using a soft X-ray technique. It is non-destructive and 
considerably cheaper than other more sophisticated imaging techniques like 
scanning microscopy or laser technology. The images were recorded on 
13x18 cm X-ray KODAK plates. Studies were conducted using combine harvested 
wheat grain originating from experimental fields, explored at the Institute 
of Agrophysics of the Polish Academy of Sciences in Lublin. 


Attribute Information:

To construct the data, seven geometric parameters of wheat kernels were 
measured: 
    1. area A, 
    2. perimeter P, 
    3. compactness C = 4*pi*A/P^2, 
    4. length of kernel, 
    5. width of kernel, 
    6. asymmetry coefficient 
    7. length of kernel groove. 


    
Relevant Papers:

M. Charytanowicz, J. Niewczas, P. Kulczycki, P.A. Kowalski, S. Lukasik, S. Zak,
 'A Complete Gradient Clustering Algorithm for Features Analysis of 
 X-ray Images', in: Information Technologies in Biomedicine, Ewa Pietka, 
 Jacek Kawa (eds.), Springer-Verlag, Berlin-Heidelberg, 2010, pp. 15-24.
  




Please use this data to finish the following tasks.

    1. Explore the data set (10 points)
    2. Use K-means clustering to group the seed data. (30 points)
    3. Use different linkage type for Hierarchical clustering to the seed data, 
    which linkage type give the best result? (30 points)
    4. Use DBscan clustering group the seed data and find the best epses 
    and min_samples value. (30 points)
    
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics

#0. Read the data in

#set the read path
readpath = "J:\DSDegree\PennState\DAAN_862\Week 11\Homework"
#change the directory
os.chdir(readpath)
#This data set does not have a header for the column names
#Set the column names for the data frame
column_names = ["area", "perimeter", "compactness", "length_kernel", 
                "width_kernel", "asymmetry", "length_grove", "type"]
#read in the csv data, set header=0 and names=column_names to
#have proper naming for the column information
df = pd.read_csv("seeds_dataset.csv", header=0, names=column_names)

#in case I want to split the class into dummy variables
dummy_class = pd.get_dummies(df["type"])
#drop class so I can use the dummy class instead
df_dummy = df.drop("type", axis = 1)
#Right Join the dummy classes to the seed dataframe
df_dummy = df_dummy.join(dummy_class, how = 'right')

#Bring the seed variables by themselves as a data frame
seeds = df.iloc[:, :7]






#1. Explore the data set (10 points)

corr_val = seeds.corr()

#plot a correlation matrix
plt.matshow(corr_val)
plt.title("Correlation Matrix")
plt.colorbar()
plt.xticks(range(7), list(seeds.columns))
plt.yticks(range(7), list(seeds.columns))


#get some info about the seeds data frame
print("\nData frame information:\n")
print(seeds.info())
print()
print(seeds.describe())


#the data should be clean but I'll still check for duplicates
dups = seeds.duplicated()
print("\nAre there any duplicates?\n" + str(dups.max()))





#2. Use K-means clustering to group the seed data. (30 points)

#I will start this using k = 3
#I will later show using the elbow method that k = 3 is correct

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

#Normalize independent variables for the seed data

#make the scaler object
scaler = MinMaxScaler()
seed_X = scaler.fit_transform(seeds)
y = df.type


#build the k-Means object
kmeans = KMeans(n_clusters = 3, random_state=226)
#fit and predict the clusters
y_seed_pred = kmeans.fit_predict(seed_X)

#Get metrics
print("\nMetrics for K-Means at k = 3:")
print("  Homogentiy: " + str(metrics.homogeneity_score(y, y_seed_pred)))
print("  Completeness: " + str(metrics.completeness_score(y, y_seed_pred)))
print("  Adjusted Rand Score: " + 
      str(metrics.adjusted_rand_score(y, y_seed_pred)))
print("  Silhouette Score: " + 
      str(metrics.silhouette_score(seed_X, y_seed_pred, metric = 'euclidean')))


#try to create a plt that shows the clustering worked
#Try this for all combinations of the variables to see good
#separating distinguishers 

#I'll comment the loop out later and ony show one plot of this
"""
for i in range (0,7):
    for j in range(0,7):
        if i != j: 

            centers = kmeans.cluster_centers_
            centers_a = centers[:, i]
            centers_b = centers[:, j]
            
            plt.figure()
            # predicted  categories (c = y_pred).
            plt.scatter(seed_X[:, i], seed_X[:, j], c = y_seed_pred, 
                        label = 'Predicted')
            # PLot the cluster centroids with 'X' and color is red
            plt.scatter(centers_a, centers_b, s= 100, c = 'r', 
                        marker = 'x', label = 'Cluster Centers')
            plt.title('Clustering results')
            plt.xlabel(seeds.columns[i])
            plt.ylabel(seeds.columns[j])
            plt.legend()
"""            


#This is a good plot to show the clustering
#I'll use it throughout this homework        
centers = kmeans.cluster_centers_
centers_a = centers[:, 6]
centers_b = centers[:, 4]

plt.figure()
# predicted  categories (c = y_pred).
plt.scatter(seed_X[:, 6], seed_X[:, 4], c = y_seed_pred, 
            label = 'Predicted')
# PLot the cluster centroids with 'X' and color is red
plt.scatter(centers_a, centers_b, s= 100, c = 'r', 
            marker = 'x', label = 'Cluster Centers')
plt.title('Clustering results')
plt.ylabel("Width_kernal")
plt.xlabel("Length_Grove")
plt.legend()
            
            
#Use the elbow method to show that k = 3 is the proper choice

#computing distances from two inputs
from scipy.spatial.distance import cdist

#set up a range of k values
k_vals = range(1,10)

#set an empty list for the mean dispersions
meanDisp = []

for k in k_vals:
    
    #build the K-Means object with the variable n_clusters
    kmeans = KMeans(n_clusters=k)
    #fit the data via K-means
    kmeans.fit(seed_X)
    
    #from the sklearn book (Hackeling 2nd ed., pg. 208)
    #Find the average minimum distance to the cluster centers
    #shows if you've done a good job with clustering
    #if K = number of points then this would be zero    
    meanDisp.append(sum(np.min(cdist(seed_X, kmeans.cluster_centers_,
                                     'euclidean'),
                               axis = 1)) / seed_X.shape[0])
    
#plot the elbow plot
plt.figure()
plt.plot(k_vals, meanDisp, 'bx-')
plt.xlabel("K")
plt.ylabel("Average Dispersion")
plt.title("Elbow plot to show best K value")




#3. Use different linkage type for Hierarchical clustering to the seed data, 
#   which linkage type give the best result? (30 points)
    

from sklearn.cluster import AgglomerativeClustering


#Using linkage = 'average'
hier1_seed_pred = AgglomerativeClustering(n_clusters=3,
                                         affinity='euclidean',
                                         linkage='average').fit_predict(seed_X)

#Get metrics
print("\nMetrics for Hierarchical using Average Linkage:")
print("  Homogentiy: " + str(metrics.homogeneity_score(y, hier1_seed_pred)))
print("  Completeness: " + str(metrics.completeness_score(y, hier1_seed_pred)))
print("  Adjusted Rand Score: " + 
      str(metrics.adjusted_rand_score(y, hier1_seed_pred)))
print("  Silhouette Score: " + 
      str(metrics.silhouette_score(seed_X, hier1_seed_pred, 
                                   metric = 'euclidean')))

#Create the dendrogram of the clusters
from scipy.cluster.hierarchy import dendrogram, linkage

Z_avg = linkage(seed_X, method = 'average')

plt.figure()
den = dendrogram(Z_avg, leaf_font_size = 8)



#Using linkage = 'complete'
hier2_seed_pred = AgglomerativeClustering(n_clusters=3,
                                         affinity='euclidean',
                                         linkage='complete').fit_predict(seed_X)
#Get metrics
print("\nMetrics for Hierarchical using Complete Linkage:")
print("  Homogentiy: " + str(metrics.homogeneity_score(y, hier2_seed_pred)))
print("  Completeness: " + str(metrics.completeness_score(y, hier2_seed_pred)))
print("  Adjusted Rand Score: " + 
      str(metrics.adjusted_rand_score(y, hier2_seed_pred)))
print("  Silhouette Score: " + 
      str(metrics.silhouette_score(seed_X, hier2_seed_pred, 
                                   metric = 'euclidean')))



#Using linkage = 'ward'
hier3_seed_pred = AgglomerativeClustering(n_clusters=3,
                                         affinity='euclidean',
                                         linkage='ward').fit_predict(seed_X)
#Get metrics
print("\nMetrics for Hierarchical using Ward Linkage:")
print("  Homogentiy: " + str(metrics.homogeneity_score(y, hier3_seed_pred)))
print("  Completeness: " + str(metrics.completeness_score(y, hier3_seed_pred)))
print("  Adjusted Rand Score: " + 
      str(metrics.adjusted_rand_score(y, hier3_seed_pred)))
print("  Silhouette Score: " + 
      str(metrics.silhouette_score(seed_X, hier3_seed_pred, 
                                   metric = 'euclidean')))


#Based on the metrics run, the Ward Linkage performs the best for clustering
#this data set having the highest silohette score (amoung other values)


#Plot the different Hierarchical Clusterings linkage types
plt.figure(figsize = (12,12))
plt.subplot(221)
plt.scatter(seed_X[:, 6], seed_X[:, 4], c = y)

plt.title('Clustering results')
plt.ylabel("Width_kernal")
plt.xlabel("Length_Grove")


plt.subplot(222)
plt.scatter(seed_X[:, 6], seed_X[:, 4], c = hier1_seed_pred)
plt.title('Average Linkage')
plt.ylabel("Width_kernal")
plt.xlabel("Length_Grove")

plt.subplot(223)
plt.scatter(seed_X[:, 6], seed_X[:, 4], c = hier2_seed_pred)
plt.title('Complete Linkage')
plt.ylabel("Width_kernal")
plt.xlabel("Length_Grove")

plt.subplot(224)
plt.scatter(seed_X[:, 6], seed_X[:, 4], c = hier3_seed_pred)
plt.title('Ward Linkage')
plt.ylabel("Width_kernal")
plt.xlabel("Length_Grove")

plt.subplots_adjust(wspace = 0.2)






#4. Use DBscan clustering group the seed data and find the best epses 
#   and min_samples value. (30 points)

from sklearn.cluster import DBSCAN


#build the model and fit predict the clusters
db_y_seed_pred = DBSCAN(eps = 0.2, min_samples=5,).fit_predict(seed_X)

#Get metrics
print("\nMetrics for DBSCAN with eps = 0.2 and min_samples = 5:")
print("  Homogentiy: " + str(metrics.homogeneity_score(y, db_y_seed_pred)))
print("  Completeness: " + str(metrics.completeness_score(y, db_y_seed_pred)))
print("  Adjusted Rand Score: " + 
      str(metrics.adjusted_rand_score(y, db_y_seed_pred)))
print("  Silhouette Score: " + 
      str(metrics.silhouette_score(seed_X, db_y_seed_pred, 
                                   metric = 'euclidean')))


#Run a search for the best 'epses' and 'min_samples' for the DBSCAN
print("\n\nSearching for the best parameters for DBSCAN... \n\n")
#make and empty list to store results
results = []

#make the 'epses' and 'min_samples' grid
epses = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
min_sampels = [2, 5, 10, 15]

#run over the search grid manually
for e in epses:
    for n in min_sampels:
        
        #set up the DBSCAN model
        model = DBSCAN(eps = e, min_samples = n)
        
        #Get the predicted clusters
        db_y_seed_pred_temp = model.fit_predict(seed_X)
        
        #find the number of clusters
        n_clusters = np.unique(model.labels_).size
        
        #get the sihouellette metrics
        #Check if there is just one cluster, this will error the Silhouette
        #so if there is only one cluster this is a bad score and will 
        #equal 0 (assuming there is more than one cluster...)
        if n_clusters == 1:
            results.append((e,n,np.nan, n_clusters))
        else:
            score = metrics.silhouette_score(seed_X, db_y_seed_pred_temp,
                                         metric = 'euclidean')
            #store the score results
            results.append((e, n, score, n_clusters))
        
#put results in to a data frame
res_hold = pd.DataFrame(results, columns = ["epses", 
                                       "min_samples", 
                                       "sil_score", 
                                       "n_clusters"])

print("\nThe Silhouette scores from the first DBSCAN search:\n")
print(res_hold)

#Get the epses, min_samps, and sil_score as reference values for the loop
r_epses = res_hold.epses[res_hold.sil_score.idxmax()]
r_min_samp = res_hold.min_samples[res_hold.sil_score.idxmax()]
r_sil_score = res_hold.sil_score.max()



#We've found the epses and min_sampels around 0.3 and 15 to be best

#will refine search to obtain better sil_score
#DO this in the while loop
#Find espes and min_num from trial 1 above, then do + / - around the params
#while loop goes until sil_score results don't increase (could be local max)
#   I have not accounted for local maximums (could provide inertia for this)

check = False

epses_inc = 0.05
min_samp_inc = 1

while check == False:

    #make and empty list to store results
    results = []
    
    #make the 'epses' and 'min_samples' grid
    epses_hold = [r_epses-(2*epses_inc),
             r_epses-epses_inc,
             r_epses,
             r_epses+epses_inc,
             r_epses+(2*epses_inc)]
    min_sampels_hold = [r_min_samp-(2*min_samp_inc),
                   r_min_samp-min_samp_inc,
                   r_min_samp,
                   r_min_samp+min_samp_inc,
                   r_min_samp+(2*min_samp_inc)]
        
    
    #run over the search grid manually
    for e in epses_hold:
        for n in min_sampels_hold:
            
            #set up the DBSCAN model
            model = DBSCAN(eps = e, min_samples = n)
            
            #Get the predicted clusters
            db_y_seed_pred_temp = model.fit_predict(seed_X)
            
            #find the number of clusters
            n_clusters = np.unique(model.labels_).size
            
            #get the sihouellette metrics
            #Check if there is just one cluster, this will error the Silhouette
            #so if there is only one cluster this is a bad score and will 
            #equal 0 (assuming there is more than one cluster...)
            if n_clusters == 1:
                results.append((e,n,np.nan, n_clusters))
            else:
                score = metrics.silhouette_score(seed_X, db_y_seed_pred_temp,
                                             metric = 'euclidean')
                #store the score results
                results.append((e, n, score, n_clusters))
            
    #put results in to a temp data frame
    res_temp = pd.DataFrame(results, columns = ["epses", 
                                           "min_samples", 
                                           "sil_score", 
                                           "n_clusters"])
    
    r_epses_temp = res_temp.epses[res_temp.sil_score.idxmax()]
    r_min_samp_temp = res_temp.min_samples[res_temp.sil_score.idxmax()]
    r_sil_score_temp = res_temp.sil_score.max()
    
    #check the reference sil_score for changes
    if r_sil_score_temp > r_sil_score:
        #a better score has been found.  Reassign the reference vals and go again
        r_sil_score = r_sil_score_temp
        r_epses = r_epses_temp
        r_min_samp = r_min_samp_temp
    elif r_sil_score_temp <= r_sil_score:
        #No improvements have been made.  Hit the flag and break the loop
        check = True
    



#reset the reference scores
res = res_temp
best_clusters = res.n_clusters[res.sil_score.idxmax()]

print("\nThe Silhouette scores from the last DBSCAN search:\n")
print(res)
print("\nThe best DBSCAN values are:")
print("  epses:            " + str(r_epses))
print("  min_samples:      " + str(r_min_samp))
print("  Silhouette Score: " + str(r_sil_score))
print("  N Clusters:       " + str(best_clusters))














































