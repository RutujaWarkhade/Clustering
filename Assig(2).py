# -*- coding: utf-8 -*-
"""
Created on Sat  Aug  31 05:05:20 2024

@author: om
"""

#2.	Perform clustering for the crime data and 
#identify the number of clusters formed and draw inferences.
# Refer to crime_data.csv dataset.
'''
business objective -
business objective is to perform clustering on states that have 
similar charateristics

'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv("C:/Assignments(DS)/crime_data.csv")
df.head()
df.dtypes
"""
Unnamed: 0     object
Murder        float64
Assault         int64
UrbanPop        int64
Rape          float64
dtype: object

"""
df.shape
#(50, 5)
df.columns
#Index(['Unnamed: 0', 'Murder', 'Assault', 'UrbanPop', 'Rape'], dtype='object')
df.describe()
'''
Murder     Assault   UrbanPop       Rape
count  50.00000   50.000000  50.000000  50.000000
mean    7.78800  170.760000  65.540000  21.232000
std     4.35551   83.337661  14.474763   9.366385
min     0.80000   45.000000  32.000000   7.300000
25%     4.07500  109.000000  54.500000  15.075000
50%     7.25000  159.000000  66.000000  20.100000
75%    11.25000  249.000000  77.750000  26.175000
max    17.40000  337.000000  91.000000  46.000000'''


#column unnamed which contains the states does not have much use
#so we can remove it from dataset
df.drop(['Unnamed: 0'],inplace=True,axis=1)

df.columns
#Index(['Murder', 'Assault', 'UrbanPop', 'Rape'], dtype='object')

#EDA

#first we can analyze the data through pairplot
import seaborn as sns
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df,height=5);
plt.show()

#by pairplot we can analyze each variable to another

#first we analyze the data by Univariate analysis.
#Univariate analysis means here we analyze the single variable
#in this we analyze throug box plot

import seaborn as sns
sns.boxplot(df['Murder'])
#their is no any outlier in Murder column
sns.boxplot(df['Assault'])
#their is no any outlier in Assaukt column
sns.boxplot(df['UrbanPop'])
#their is no any outlier in UrbanPoP column
sns.boxplot(df['Rape'])
#their is outlier in Rape column

#Hence we have to remove it by using iqr technique


iqr = df['Rape'].quantile(0.75)-df['Rape'].quantile(0.25)
iqr

q1 = df['Rape'].quantile(0.25)
q3=df['Rape'].quantile(0.75)

lower_bound = q1-(1.5*iqr)
upper_bound = q3+(1.5*iqr)

df['Rape'] = np.where(df.Rape >upper_bound,upper_bound,np.where(df.Rape<lower_bound,lower_bound,df.Rape))
sns.boxplot(df['Rape'])


#now outlier is removed in rape column


#now we have to normalized the data set
def norm_fun(i):
    x = (i-i.min())/(i.max()-i.min())
    return x

df_norm = df.apply(norm_fun)
df_norm

#hence now all values are in range 0 to 1

#Dendogram
#to calculate number of cluster we have to plot dendogram
#dendrogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z = linkage(df_norm,method='complete',metric='euclidean')
plt.figure(figsize=(15,8))
plt.title('Hierarchical clustering dendrogram')
plt.xlabel('index')
plt.ylabel('distance')
#dendrogram
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()

#now apply clustering 
from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters=3,linkage='complete',metric='euclidean').fit(df_norm)
#apply labels to clusters
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
#assign this series to autoIns dataframe as column
df['cluster'] = cluster_labels
df.columns
df1 = df.iloc[:,[-1,0,1,2,3]]
df1.columns
df1.iloc[:,2:].groupby(df1.cluster).mean()
df1.to_csv("CrimeDataNew.csv",encoding='utf-8')
df1.cluster.value_counts()
import os
os.getcwd()
