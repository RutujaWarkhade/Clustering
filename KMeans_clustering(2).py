# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 08:33:12 2024

@author: om
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#let us try to understand first how k means works for two
#dimentional data
#for that, generate random numbers in the range 0 to 1
#and with uniform probability of 1/50
X=np.random.uniform(0,1,50)
X
Y=np.random.uniform(0,1,50)
#create a empty dataframe with 0 rows and 2 columns
df_xy=pd.DataFrame(columns=['X','Y'])
#assign the values of x and y to these columns
df_xy.X=X
df_xy.Y=Y
df_xy.plot(x='X',y='Y',kind='scatter')
model1=KMeans(n_clusters=3).fit(df_xy)
"""
with data x and y , apply kmeans model,
generate scatter plot
with scale/font=10
cmap=plt.cm.coolwarm:cool color combination
"""
model1.labels_
df_xy.plot(x='X',y='Y',c=model1.labels_,
           kind='scatter',s=10,cmap=plt.cm.coolwarm)
##############################################

Univ1=pd.read_csv("C:/7-clustering/University_Clustering.csv")
Univ1.describe()
Univ=Univ1.drop(["State"],axis=1)
#we know that there is scale difference among the columns , 
#which we
#either by using normalization or standardization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
#not apply this normalization function to Univ dataframe for allthe

df_norm=norm_func(Univ.iloc[:,1:])
"""what will be ideal cluster number,will it be 1,2 or 3"""

TWSS=[]
k=list(range(2,8))
k
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    #total within sum of square
    
"""KMeans inertia ,also known as SUm 
of Square Errors(or SSE),
calculate the sum of the distances
of all the points within a cluster from
the centroid of the point.
It is the differnece 
between the observed value and
 the predicted value"""
TWSS

##As k value increases the TWSS value decreases
plt.plot(k,TWSS,'ro-')
plt.xlabel("No_of_clusters");
plt.ylabel("Total_within_SS")
'''
How to select value of k from elbow curve
when k changes from 2 to 3,then decrease
in twss is higher than when k changes from 3 to 4.
when k values changes from 5 to 6 dectease
'''
model=KMeans(n_clusters=3)
model.fit(df_norm)




