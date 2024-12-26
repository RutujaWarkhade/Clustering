# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 21:31:34 2024

@author: om
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#first we import the file
df = pd.read_excel("C:/Assignments(DS)/EastWestAirlines.xlsx")

"""
Business objective:   
business objective is to perform Perform K means 
clustering and obtain optimum number of clusters
"""
#Data Dictionary

df.columns
""" 
By this we get all columns present in our dataset
i.e
Index(['ID#', 'Balance', 'Qual_miles', 
'cc1_miles', 'cc2_miles', 'cc3_miles',
'Bonus_miles', 'Bonus_trans', 
'Flight_miles_12mo', 'Flight_trans_12',
'Days_since_enroll', 'Award?'],dtype='object')

"""
#now we check the data type of each column by using set function
set(df['ID#'])
#it is continuous data type and it does not give any info
set(df['Balance'])
#continuous data type and it give account balance of the customer
set(df['Qual_miles'])
#continuous data type and it provide Qualifying miles earned
set(df['cc1_miles'])
#it is discrete data type and it is The number of miles earned from the first credit card
set(df['cc2_miles'])
#it is discrete data type and it is The number of miles earned from the second credit card
set(df['cc3_miles'])
#it is discrete data type and it is The number of miles earned from the third credit card
set(df['Bonus_miles'])
#it is continous data type and it is The number of bonus miles earned by the customer through promotions or offers.
set(df['Bonus_trans'])
#it is continous data type and it is The number of transactions that resulted in bonus miles being awarded.
set(df['Flight_miles_12mo'])
#it is continous data type and it is The number of miles the customer has flown in the last 12 months.
set(df['Flight_trans_12'])
#it is continous data type and it is The number of flights taken by  the customer has flown in the last 12 months.
set(df['Days_since_enroll'])
#it is continous data type and it is number of days since the customer enrolled in the airline's frequent flyer
set(df['Award?'])
#it is descrete data type and it is Indicates whether the customer has received an award (such as a free flight or upgrade).

#Exploratory Data Analysis (EDA)

df.shape
"""
(3999, 12)
i.e there are 3999 rows and 12 columns
"""

df.head
"""
from this we get 5 rows data
"""
nan_summary = df.isna().sum()
print(nan_summary)

"""
ID#                  0
Balance              0
Qual_miles           0
cc1_miles            0
cc2_miles            0
cc3_miles            0
Bonus_miles          0
Bonus_trans          0
Flight_miles_12mo    0
Flight_trans_12      0
Days_since_enroll    0
Award?               0
dtype: int64

This shows that their is no NaN value in given data set
"""

#5 no. summary
df.describe()
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
from above we can observed that that is many diffrence between 
mean, max, and min values hence we have to normalized the data
"""

#first we analyze the data by Univariate analysis.
#Univariate analysis means here we analyze the single variable
#in this we analyze throug box plot

import seaborn as sns
sns.boxplot(df['ID#'])
#their is no any outlier in ID column
sns.boxplot(df['Balance'])
#their is outlier in Balance column
sns.boxplot(df['Qual_miles'])
#their is outlier in Qual_miles column
sns.boxplot(df['cc1_miles'])
#their is no outlier in cc1_miles column
sns.boxplot(df['cc2_miles'])
#their is outlier in cc2_miles column
sns.boxplot(df['cc3_miles'])
#their is outlier in cc3_miles column
sns.boxplot(df['Bonus_miles'])
#their is outlier in Bonus_miles column
sns.boxplot(df['Bonus_trans'])
#their is outlier in Bonus_trans column
sns.boxplot(df['Flight_miles_12mo'])
#their is outlier in Flight_miles_12mo column
sns.boxplot(df['Flight_trans_12'])
#there is outlier in Flight_trans_12 column
sns.boxplot(df['Days_since_enroll'])
#there is no outlier in Days_since_enroll column
sns.boxplot(df['Award?'])
#there is no outlier in Award? column
"""
Hence from above boxplot it is observerd that except
ID#,cc1_miles,Days_since_enroll,Award? columns all
other columns have outlier hence we have to remove it
"""

#Data Pre-processing
#for removing outlier we use IQR
#1.Balance
# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df['Balance'].quantile(0.25)
Q3 = df['Balance'].quantile(0.75)
IQR = Q3 - Q1
# Determine the bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# Remove outliers
df['Balance'] = df.loc[(df['Balance'] >= lower_bound) & (df['Balance'] <= upper_bound), 'Balance'] 
sns.boxplot(df['Balance'])
#from again from box plot we can see that outlier has removed


#2.Qual_miles
# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df['Qual_miles'].quantile(0.25)
Q3 = df['Qual_miles'].quantile(0.75)
IQR = Q3 - Q1
# Determine the bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# Remove outliers
df['Qual_miles'] = df.loc[(df['Qual_miles'] >= lower_bound) & (df['Qual_miles'] <= upper_bound), 'Qual_miles'] 
sns.boxplot(df['Qual_miles'])
#from again from box plot we can see that outlier has removed

#3.cc2_miles
# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df['cc2_miles'].quantile(0.25)
Q3 = df['cc2_miles'].quantile(0.75)
IQR = Q3 - Q1
# Determine the bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# Remove outliers
df['cc2_miles'] = df.loc[(df['cc2_miles'] >= lower_bound) & (df['cc2_miles'] <= upper_bound), 'cc2_miles'] 
sns.boxplot(df['cc2_miles'])
#from again from box plot we can see that outlier has removed


#4.cc3_miles
# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df['cc3_miles'].quantile(0.25)
Q3 = df['cc3_miles'].quantile(0.75)
IQR = Q3 - Q1
# Determine the bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# Remove outliers
df['cc3_miles'] = df.loc[(df['cc3_miles'] >= lower_bound) & (df['cc3_miles'] <= upper_bound), 'cc3_miles'] 
sns.boxplot(df['cc3_miles'])
#from again from box plot we can see that outlier has removed

#5.Bonus_miles
# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df['Bonus_miles'].quantile(0.25)
Q3 = df['Bonus_miles'].quantile(0.75)
IQR = Q3 - Q1
# Determine the bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# Remove outliers
df['Bonus_miles'] = df.loc[(df['Bonus_miles'] >= lower_bound) & (df['Bonus_miles'] <= upper_bound), 'Bonus_miles'] 
sns.boxplot(df['Bonus_miles'])
#from again from box plot we can see that outlier has removed


#6.Bonus_trans
# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df['Bonus_trans'].quantile(0.25)
Q3 = df['Bonus_trans'].quantile(0.75)
IQR = Q3 - Q1
# Determine the bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# Remove outliers
df['Bonus_trans'] = df.loc[(df['Bonus_trans'] >= lower_bound) & (df['Bonus_trans'] <= upper_bound), 'Bonus_trans'] 
sns.boxplot(df['Bonus_trans'])
#from again from box plot we can see that outlier has removed


#7.Flight_miles_12mo
# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df['Flight_miles_12mo'].quantile(0.25)
Q3 = df['Flight_miles_12mo'].quantile(0.75)
IQR = Q3 - Q1
# Determine the bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# Remove outliers
df['Flight_miles_12mo'] = df.loc[(df['Flight_miles_12mo'] >= lower_bound) & (df['Flight_miles_12mo'] <= upper_bound), 'Flight_miles_12mo'] 
sns.boxplot(df['Flight_miles_12mo'])
#from again from box plot we can see that outlier has removed


#8.Flight_trans_12
# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df['Flight_trans_12'].quantile(0.25)
Q3 = df['Flight_trans_12'].quantile(0.75)
IQR = Q3 - Q1
# Determine the bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# Remove outliers
df['Flight_trans_12'] = df.loc[(df['Flight_trans_12'] >= lower_bound) & (df['Flight_trans_12'] <= upper_bound), 'Flight_trans_12'] 
sns.boxplot(df['Flight_trans_12'])
#from again from box plot we can see that outlier has removed

#now we again describe the data
df.describe()
"""
ID#        Balance  ...  Days_since_enroll       Award?
count  3999.000000    3733.000000  ...         3999.00000  3999.000000
mean   2014.819455   53831.927940  ...         4118.55939     0.370343
std    1160.764358   46937.887757  ...         2065.13454     0.482957
min       1.000000       0.000000  ...            2.00000     0.000000
25%    1010.500000   17481.000000  ...         2330.00000     0.000000
50%    2016.000000   38671.000000  ...         4096.00000     0.000000
75%    3020.500000   77540.000000  ...         5790.50000     1.000000
max    4021.000000  202636.000000  ...         8296.00000     1.000000

[8 rows x 12 columns]
"""
"""
from above it is clear that their is large diffrence between 
mean, max and min hence we apply either normalization 
or standardization.
We can not apply standardization because mean is not 0 and 
std deviation is not 1 and their is mixed data type
Hence we will apply normalization
"""
#normalization
def norm_func(i):
    x = (i - i.min()) / (i.max() - i.min())
    return x

df_norm = df.apply(norm_func)
df_norm
"""
ID#   Balance  ...  Days_since_enroll  Award?
0     0.000000  0.138885  ...           0.843742     0.0
1     0.000249  0.094968  ...           0.839884     0.0
2     0.000498  0.204080  ...           0.847842     0.0
3     0.000746  0.072919  ...           0.837955     0.0
4     0.000995  0.482402  ...           0.835905     1.0
       ...       ...  ...                ...     ...
3994  0.999005  0.091178  ...           0.168917     1.0
3995  0.999254  0.317737  ...           0.167953     1.0
3996  0.999502  0.363198  ...           0.168797     1.0
3997  0.999751  0.270924  ...           0.168676     0.0
3998  1.000000  0.014884  ...           0.168314     0.0

[3999 rows x 12 columns]

Hence here we can observed that all valuses are in the 
range 0 to 1
"""








# Check for NaNs and infinite values
print("NaN values before filling:", df_norm.isna().sum())
"""
NaN values before filling:
ID#                     0
Balance               266
Qual_miles           3999
cc1_miles               0
cc2_miles            3999
cc3_miles            3999
Bonus_miles           280
Bonus_trans            63
Flight_miles_12mo     569
Flight_trans_12       565
Days_since_enroll       0
Award?                  0
dtype: int64
NaN value is present in dataset after applying 
normalization function
"""

print("Any infinite values present:", np.isinf(df_norm).sum().sum())
"""
Any infinite values present: 0
hence there is no any infinite value present here
"""
# Fill NaNs with the mean 
df_norm.fillna(df_norm.mean(), inplace=True)


# if their is any infinite value replace infinite values
# with the maximum finite value in their respective columns
df_norm.replace([np.inf, -np.inf], np.nan, inplace=True)
df_norm.fillna(df_norm.max(), inplace=True)

# Recheck for NaNs and infinite values
print("NaN values after filling:", df_norm.isna().sum())
print("Any infinite values present:", np.isinf(df_norm).sum().sum())
#now their is no any NaN and infinite value








# Perform hierarchical clustering
#linkage function gives us hierarchical or aglomerative clustering
#ref the help of linkage 
from scipy.cluster.hierarchy import linkage, dendrogram
z = linkage(df_norm, method='complete', metric='euclidean')
# Plot the dendrogram
plt.figure(figsize=(10, 7))
plt.title("Dendrogram")
dendrogram(z)
plt.show()

#Kmeans
#for this we will used normalized data set df_normal

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
[2908.260739348439,
 2263.316171028324,
 1932.897637296432,
 1667.4170613324961,
 1484.8283001092923,
 1313.3078600029346]
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
df_norm = df_norm.iloc[:,[-1,0,1,2,3,4,5,6,7,8,9,10]]
df_norm
df_norm.iloc[:,2:11].groupby(df_norm.clusters).mean()
df_norm.to_csv('k_means_EastWestAirlines.csv')
import os
os.getcwd()




"""
benifits:
    
can lead to a more satisfying and rewarding experience 
for customers, enhancing their loyalty to the airline.


"""
