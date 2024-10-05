# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 08:14:33 2024

@author: om
"""

import pandas as pd
import matplotlib.pylab as plt
#now import file from data set and create a dataframe
#Load and preprocess data
Univ1=pd.read_csv("C:/7-clustering/University_Clustering.csv")
Univ1.head()
#we have one column "State" which really not usefull we will dorp it
a=Univ1.describe()
a
Univ=Univ1.drop(["State"],axis=1)
Univ
#we know that there is scale difference among the columns,
#which we have to remove
#either by using normalization or standardization
#whenever there is mixed data apply normalization
#normalize the data
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
#now apply this normalization function to Univ dataframe
#for all the rows and column from 1 untill end
#since 0 th column has University name hence skipped
    
df_norm=norm_func(Univ.iloc[:,1:])
df_norm
#you can check the df_norm dataframe which is scaled
#between values from 0 to 1
#you can apply describe function to new dataframe

b=df_norm.describe()
b
#before you apply clustering, you need to plot dendogram first
#now to create dendogram, we need to measured distance,
#we have to import linkage
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
#linkage function gives us hierarchical or aglomerative clustering
#ref the help of linkage
# Perform hierarchical clustering 
z=linkage(df_norm,method="complete",metric="euclidean")
# Plot the dendrogram
plt.figure(figsize=(15,8));
plt.title("Hierachical Clustering dendrogram");
plt.xlabel("Index");
plt.ylabel("Distance")
#ref help of dendogram
#sch.dendogram(z)
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()
#dendogram()
#applying agglomerative clustering choosing 5 as clusters from dendogram
#whatever has been displayed in dendogram is not clustering
#it is just showing number of possible clusters
# Perform Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=3,linkage='complete',metric="euclidean")
#######affinity has been depricated,use metric
#apply labels to the clusters
labels=h_complete.fit_predict(df_norm)
# Convert labels to a pandas Series
cluster_labels = pd.Series(labels)
print(cluster_labels)
#assign this series to Univ DataFrame as columns and namethe columns 
Univ['clust']=cluster_labels
#we want to relocate the column 7 to 0th position
Univ1=Univ.iloc[:,[7,1,2,3,4,5,6]]
#now check the Univ1 dataframe




Univ1.iloc[:,2:].groupby(Univ1.clust).mean()
#from the output cluster 2 has got highest Top10






