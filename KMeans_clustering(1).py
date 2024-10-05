# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 09:08:27 2024

@author: om
"""

from sklearn.cluster import KMeans
#library is used for performing K-means clustering in machine learning
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#library is used to scale features to a specific range, 
#typically [0, 1]. It performs min-max normalization, 
#which transforms the data to a common scale without 
#distorting differences in the range of values.
from matplotlib import pyplot as plt
df=pd.read_csv("C:/7-clustering/income (1).csv")
df.head()
plt.scatter(df.Age,df['Income($)'])
plt.xlabel('Age')
plt.ylabel('Income($)')
km=KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age','Income($)']])
y_predicted
df['cluster']=y_predicted
df.head()
km.cluster_centers_

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='red')
plt.scatter(df3.Age,df3['Income($)'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
#here purple color is for centroid
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.legend()
#observation:clusters here formed do not show the same properties
#as 1 cluster should have same color which defines same properties
#for that we need to do scaling then will get uniform clusters having same properties
    
#preprocessing using min max scaler
scaler=MinMaxScaler()
scaler.fit(df[['Income($)']])
df['Income($)']=scaler.transform(df[['Income($)']])

scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])
df.head()

plt.scatter(df.Age,df['Income($)'])
km=KMeans(n_clusters=3)
y_predicted =km.fit_predict(df[['Age','Income($)']])

y_predicted
df['cluster']=y_predicted
df.head()
km.cluster_centers_

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='red')
plt.scatter(df3.Age,df3['Income($)'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
#here purple color is for centroid
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.legend()
#here clusters will be formed having same properties 
#as show by the colurs
#and purple color is centroid of each clusters




