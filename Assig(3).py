# -*- coding: utf-8 -*-
"""
Created on Sun Aug  30 06:03:12 2024

@author: om
"""

#3.	Analyze the information given in the following ‘Insurance Policy dataset’
# to create clusters of persons falling in the same type. 
#Refer to Insurance Dataset.csv


'''
#business objective:
business objective is to perform clustering on data that have 
similar charateristics

'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
df = pd.read_csv("C:/Assignments(DS)/Insurance Dataset.csv")
df.shape
#(100, 5)

df.columns
#Index(['Premiums Paid', 'Age', 'Days to Renew', 'Claims made', 'Income'], 
#dtype='object')

df.dtypes
'''
Premiums Paid      int64
Age                int64
Days to Renew      int64
Claims made      float64
Income             int64
'''
df.describe()
#pairplot
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df, height=4);
plt.show()
#boxplot and outlier treatments

sns.boxplot(df['Premiums Paid'])
#their is outlier
sns.boxplot(df['Age'])
#no outlier
sns.boxplot(df['Days to Renew'])
#no outlier
sns.boxplot(df['Claims made'])
#outlier
sns.boxplot(df['Income'])
#no outlier

# only columns premium paid, claims made have outliers

#1
iqr = df['Premiums Paid'].quantile(0.75)-df['Premiums Paid'].quantile(0.25)
iqr
q1=df['Premiums Paid'].quantile(0.25)
q3=df['Premiums Paid'].quantile(0.75)

lower_bound = q1-1.5*(iqr)
upper_bound = q3+1.5*iqr
df['Premiums Paid']=  np.where(df['Premiums Paid']>upper_bound,upper_bound,np.where(df['Premiums Paid']<lower_bound,lower_bound,df['Premiums Paid']))
sns.boxplot(df['Premiums Paid'])

#2
iqr = df['Claims made'].quantile(0.75)-df['Claims made'].quantile(0.25)
iqr
q1=df['Claims made'].quantile(0.25)
q3=df['Claims made'].quantile(0.75)

lower_bound = q1-1.5*(iqr)
upper_bound = q3+1.5*iqr
df['Claims made']=  np.where(df['Claims made']>upper_bound,upper_bound,np.where(df['Claims made']<lower_bound,lower_bound,df['Claims made']))
sns.boxplot(df['Claims made'])
df.describe()
#we can see that there is huge difference between min,max and mean
# values for all the columns so we need to normalize the dataset

def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return x

df_norm = norm_func(df)
df_norm

#dendogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z = linkage(df_norm,method='complete',metric='euclidean')
plt.figure(figsize=(15,8))
plt.title('Hierarchical clustering dendrogram')
plt.xlabel('index')
plt.ylabel('Distance')
#ref of dendrogram

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
df.shape
df1 = df.iloc[:,[-1,0,1,2,3,4,5,6,7,8,9,10]]
df1.columns

df1.iloc[:,2:].groupby(df1.cluster).mean()
df1.to_csv("Insurance DatasetNew.csv",encoding='utf-8')
df1.cluster.value_counts()
import os
os.getcwd()

#kmeans clustering on insurance data
#for this we will use normalized dataset i.e df_normal

from sklearn.cluster import KMeans
#total sum of squares
TWSS = []

#initially we will find the ideal cluster number using elbow curve

k = list(range(2,8))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
  
TWSS
'''
[21.37095544074799,
 15.758410404424241,
 12.06613037075154,
 9.729441304368878,
 8.101284801872284,
 7.230049855975996]
'''

'''
k selected by calculating the difference or decrease in
twss value 
'''
def find_cluster_number(TWSS):
    diff =[]
    for i in range(0,len(TWSS)-1):
        d = TWSS[i]-TWSS[i+1]
        diff.append(d)
    max = 0
    k =0
    for i in range(0,len(diff)):
        if max<diff[i]:
            max = diff[i]
            k = i+3
    return k

k = find_cluster_number(TWSS)
print("Cluster number is = ",k)
plt.plot(k,TWSS,'ro-')
plt.xlabel('No of clusters')
plt.ylabel('Total_within_SS')

model = KMeans(n_clusters=k)
model.fit(df_norm)
model.labels_
mb = pd.Series(model.labels_)
df_norm['clusters'] = mb
df_norm.head()
df_norm.shape
df_norm.columns
df_norm = df_norm.iloc[:,[-1,0,1,2,3,4]]
df_norm
#df_normal.drop(['clusters'],axis=1,inplace=True)
df_norm.iloc[:,2:5].groupby(df_norm.clusters).mean()
df_norm.to_csv('k_means_Insurance_data.csv')
import os
os.getcwd()




