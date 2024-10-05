# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 21:36:14 2024

@author: om
"""

"""
Problem Statement:
    Perform clustering for the crime data and
    identify the number of clusters formed and 
    draw inferences. Refer to crime_data.csv dataset.
"""

#######################################################
"""
Business Problem:
    
Perform clustering for the crime data and identify
the number of clusters formed and draw inferences.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
crime1 = pd.read_csv("C:/Assignments(DS)/crime_data.csv")
crime1.shape
#(50, 5)
crime1.columns
"""
Index(['Unnamed: 0', 'Murder', 'Assault', 'UrbanPop', 'Rape'], dtype='object')
"""
crime1.head()
""" 
   Unnamed: 0  Murder  Assault  UrbanPop  Rape
0     Alabama    13.2      236        58  21.2
1      Alaska    10.0      263        48  44.5
2     Arizona     8.1      294        80  31.0
3    Arkansas     8.8      190        50  19.5
4  California     9.0      276        91  40.6
"""
"""
Here 'Unnamed: 0' column is not usefull 
so we have to drop it
"""
crime=crime1.drop(["Unnamed: 0"],axis=1)
crime.head()
"""
Out[9]: 
   Murder  Assault  UrbanPop  Rape
0    13.2      236        58  21.2
1    10.0      263        48  44.5
2     8.1      294        80  31.0
3     8.8      190        50  19.5
4     9.0      276        91  40.6
"""
crime.isnull().sum()
"""
Murder      0
Assault     0
UrbanPop    0
Rape        0
dtype: int64
"""
"""
Their is no any null
value in this dataset

"""
crime.describe()
"""
Out[8]: 
         Murder     Assault   UrbanPop       Rape
count  50.00000   50.000000  50.000000  50.000000
mean    7.78800  170.760000  65.540000  21.232000
std     4.35551   83.337661  14.474763   9.366385
min     0.80000   45.000000  32.000000   7.300000
25%     4.07500  109.000000  54.500000  15.075000
50%     7.25000  159.000000  66.000000  20.100000
75%    11.25000  249.000000  77.750000  26.175000
max    17.40000  337.000000  91.000000  46.000000
"""
"""
There is large difference in between min, max, & mean
Hence we have to convert it in same range.
so we have to apply normalization to convert 
all values in between 0 to 1
"""
def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df_norm = norm_fun(crime)
df_norm.head()
"""
Out[14]: 
     Murder   Assault  UrbanPop      Rape
0  0.746988  0.654110  0.440678  0.359173
1  0.554217  0.746575  0.271186  0.961240
2  0.439759  0.852740  0.813559  0.612403
3  0.481928  0.496575  0.305085  0.315245
4  0.493976  0.791096  1.000000  0.860465
"""
#total within sum of square (TWSS)
TWSS=[]
k=list(range(2,8))
for i in k:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
TWSS
"""
[6.596893867946197,
 5.019054546630204,
 3.6908204103921114,
 3.2282685857460947,
 2.9817801030435462,
 2.8823888444318477]
"""
#now we plot elbow curve
plt.plot(k, TWSS, 'ro-')
plt.title("Elbow curve")
plt.xlabel("Number of cluster")
plt.ylabel("TWSS")
plt.show()

"""
as the number of cluster increases 
TWSS dcreases
"""
model = KMeans(n_clusters=3)
model.fit(df_norm)
model.labels_
#now we conver this into dataframe Series
mb = pd.Series(model.labels_)

"""
now we add this clustering column in
 our orignal data set
"""
crime['clust']=mb
crime.head()
"""
Out[27]: 
   Murder  Assault  UrbanPop  Rape  clust
0    13.2      236        58  21.2      1
1    10.0      263        48  44.5      1
2     8.1      294        80  31.0      2
3     8.8      190        50  19.5      2
4     9.0      276        91  40.6      2
"""
crime.groupby(crime.clust).mean()
"""
Out[28]: 
          Murder     Assault   UrbanPop       Rape
clust                                             
0       4.300000   88.000000  57.894737  13.636842
1      13.469231  265.307692  61.538462  28.400000
2       7.366667  189.833333  76.500000  24.072222
"""
"""
It is observed that clust 1 has highest 
Murder & Rape cases
"""


crime.to_csv("C:/Assignments(DS)/Kmeans(Crime).csv")
import os
os.getcwd()

"""
Benifits:
    K-means clustering analysis enhances customer 
    safety by enabling targeted crime prevention 
    efforts in high-risk areas. 
"""