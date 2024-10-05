# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 19:18:08 2024

@author: om
"""
"""
Business Problem:
Perform K means clustering on the airlines dataset 
to obtain optimum number of clusters. 
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

airline = pd.read_excel("C:/Assignments(DS)/EastWestAirlines.xlsx")

#Perform Exploratory Analysis

airline.shape
#(3999, 12)
#there are 3999 rows and 12 columns
airline.columns
"""
Index(['ID#', 'Balance', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles',
       'Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12',
       'Days_since_enroll', 'Award?'],
      dtype='object')
"""
airline.head()
"""
     ID#  Balance  Qual_miles  ...  Flight_trans_12  Days_since_enroll  Award?
0    1    28143           0  ...                0               7000       0
1    2    19244           0  ...                0               6968       0
2    3    41354           0  ...                0               7034       0
3    4    14776           0  ...                0               6952       0
4    5    97752           0  ...                4               6935       1

[5 rows x 12 columns]

"""
airline.isnull()
"""
ID#  Balance  Qual_miles  ...  Flight_trans_12  Days_since_enroll  Award?
0     False    False       False  ...            False              False   False
1     False    False       False  ...            False              False   False
2     False    False       False  ...            False              False   False
3     False    False       False  ...            False              False   False
4     False    False       False  ...            False              False   False
    ...      ...         ...  ...              ...                ...     ...
3994  False    False       False  ...            False              False   False
3995  False    False       False  ...            False              False   False
3996  False    False       False  ...            False              False   False
3997  False    False       False  ...            False              False   False
3998  False    False       False  ...            False              False   False

[3999 rows x 12 columns]

There is no any null value in this dataset

"""
airline.describe()
"""
               ID#       Balance  ...  Days_since_enroll       Award?
count  3999.000000  3.999000e+03  ...         3999.00000  3999.000000
mean   2014.819455  7.360133e+04  ...         4118.55939     0.370343
std    1160.764358  1.007757e+05  ...         2065.13454     0.482957
min       1.000000  0.000000e+00  ...            2.00000     0.000000
25%    1010.500000  1.852750e+04  ...         2330.00000     0.000000
50%    2016.000000  4.309700e+04  ...         4096.00000     0.000000
75%    3020.500000  9.240400e+04  ...         5790.50000     1.000000
max    4021.000000  1.704838e+06  ...         8296.00000     1.000000

[8 rows x 12 columns]
"""
"""
There is large difference between 
min , max , mean hence we have convert 
it in same range.
so he apply normalization
"""

def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df_norm = norm_fun(airline)
df_norm.head()
"""
now there is all value is between 0 to 1
"""
"""
Out[14]: 
        ID#   Balance  Qual_miles  ...  Flight_trans_12  Days_since_enroll  Award?
0  0.000000  0.016508         0.0  ...         0.000000           0.843742     0.0
1  0.000249  0.011288         0.0  ...         0.000000           0.839884     0.0
2  0.000498  0.024257         0.0  ...         0.000000           0.847842     0.0
3  0.000746  0.008667         0.0  ...         0.000000           0.837955     0.0
4  0.000995  0.057338         0.0  ...         0.075472           0.835905     1.0

[5 rows x 12 columns]

"""
#total within sum of square
TWSS=[]
k=list(range(2,8))
for i in k:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
"""
[1146.4595170935338,
 865.2661583136116,
 674.8718323189204,
 589.5030007211359,
 497.3526390969363,
 397.95515863240183]
"""

# Scree Plot : (elbow curve)

plt.plot(k, TWSS, 'ro-')
plt.title("Elbow curve")
plt.xlabel("Number of clusters")
plt.ylabel("Total within sum of square")

"""
Obervation from elbow cueve:
    when number of cluster increases then TWSS
    will be  increses
"""
model = KMeans(n_clusters=3)
model.fit(df_norm)
model.labels_

#now we conver this into dataframe series

mb=pd.Series(model.labels_)
mb
"""
Out[22]: 
0       2
1       2
2       2
3       2
4       1
       ..
3994    1
3995    1
3996    1
3997    0
3998    0
Length: 3999, dtype: int32

"""
"""
now we add this clustering column in
 our orignal data set
"""
airline['Clust']=mb
airline.head()
"""
Out[24]: 
   ID#  Balance  Qual_miles  ...  Days_since_enroll  Award?  Clust
0    1    28143           0  ...               7000       0      2
1    2    19244           0  ...               6968       0      2
2    3    41354           0  ...               7034       0      2
3    4    14776           0  ...               6952       0      2
4    5    97752           0  ...               6935       1      1

[5 rows x 13 columns]
     
"""
airline.groupby(airline.Clust).mean()
"""
Out[25]: 
               ID#       Balance  ...  Days_since_enroll  Award?
Clust                             ...                           
0      3149.768634  42416.722826  ...        2110.052019     0.0
1      1745.592843  97053.051317  ...        4625.062120     1.0
2      1150.518699  78019.025203  ...        5611.914634     0.0

[3 rows x 12 columns]

"""
"""
From above it is observed that 1 clust has highest balance
"""
airline.to_csv("C:/Assignments(DS)/KMeans(EastWestAirlines).csv")
import os
os.getcwd()

"""
Benifits:
    By analyzing the clusters, the airline can optimize 
    resources such as staff and flight schedules 
    according to customer demand patterns, 
    improving efficiency and service delivery.
"""