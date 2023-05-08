# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 14:46:15 2018

@author: Leo
"""

from pandas import Series, DataFrame
import pandas as pd
import numpy as np



# Series
S1 = Series(range(5))
S1
S2 = Series(range(5), index = ['a', 'b', 'c', 'd', 'e'])
S2

S2['c']             # Select row 'c'
S2['e'] = 100       # Assign 100 to row 'e'
S2[['a', 'c', 'e']] # Select row 'a', 'c' and 'e'

S2[S2 > 2]          # Select all elements larger than 2
S2 * 10             # Mulptiply 10 to all elements
np.sqrt(S2)         # Calulate the square root of all elements.

# row names
S2.axes             # Return row index labels
S2.empty            # Check if the Seires is empty
S2.hasnans          # Check if the Series has NaNs
S2.shape            # Shape of the Series
S2.size             # The number of elements 



# creat a Data frame from a dict
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
frame = DataFrame(data)
frame

# specify the column names order
DataFrame(data, columns=['year', 'state', 'pop'])


# use a Numpy array to create a dataframe.
index = list('abcde')
df = DataFrame(np.random.randn(5,3), 
               index = index, 
               columns = ['BAC', 'C', 'JPM'])
df

# column names
df.columns
df.shape

# Reindex the data framdfe

df1 = df.reindex(index=['d', 'b', 'a', 'c', 'f'])
df1

# reset the index use integers and drop the original index
df.reset_index(drop = True)

# drop rows or columns 
df
df.drop('c')                # Remove the row 'c'
df.drop(['a', 'c', 'e'])    # Remove the rows 'a', 'c' and 'e'
df.drop('BAC', axis = 1)    # Remove the column 'BAC'

# indexing
# Select columns
df['C']                     # Select the column 'C'
df.C                        # Select the column 'C'
df.loc[:, 'C']              # Select the column 'C'
df.iloc[:, 1]               # Select the column 'C'

df[['BAC', 'JPM']]          # Select the columns 'BAC' and 'JPM'
df.loc[:, ['BAC', 'JPM']]   # Select the columns 'BAC' and 'JPM'
df.iloc[:, [0, 2]]          # Select the columns 'BAC' and 'JPM'




# Select rows
df.loc['c']
df.iloc[2]

# select the 2-4 rows
df[1:4]
df.loc['b' : 'd']
df.iloc[1:4]

# Select a subset of the data
df.loc['b' : 'd', 'C':]     # Selct row 'b' to 'd',and all colomns 
# from 'C' to the end


# Select an element
df.at['b', 'JPM']
df.loc['b', 'JPM']
df.iloc[1, 2]

# Selecting values with a boolean array

df['C'] > 0                 # Select 
df.loc[df['C'] >0, :]
df.loc['c'] < 0
df.loc[:, df.loc['c'] < 0 ]


# Dataloading storage
 # set the working directory to where you saved iris.csv
import os 
path = "E:\GoogleDrive\PSU\DAAN862\Course contents\Lesson 3"
os.chdir(path)         
iris = pd.read_csv("iris.csv")
iris[:3]

iris = pd.read_table("iris.csv", sep = ',')
iris[:3]

iris2 = pd.read_csv("iris_no_header.csv", header = None)
iris2[:3]

iris2 = pd.read_csv("iris_no_header.csv", 
                  names = ['sl', 'sw', 'pl', 'pw', 'species'])
iris2[:3]

iris3 = pd.read_csv("iris.csv", nrows = 3)
iris3

iris4 = pd.read_excel('iris.xlsx', nrow = 3)
iris4[:3]

# export data 
iris3.to_csv('iris_export.csv', index = False)  # If the file exists, it will overwrite it.
pd.read_csv('iris_export.csv')
iris3.to_excel('iris_excel.xlsx')
pd.read_excel('iris_excel.xlsx')

# Statistic analysis.
iris_sub = iris.iloc[:, :4] # remove species clumn since sum can only applied to numeric variables.
iris_sub.head()              # display first 5 rows
iris_sub.sum()                          # column sums
iris_sub.sum(axis = 1, skipna = True).head()   # calculate Row sums and display the head

iris.describe()
iris.count()
iris_diff = df[df.columns[:4]].diff()
iris_diff[:3]

# count and unique values
iris.species.unique()
iris.species.value_counts()
iris.species.isin(['versicolor', 'setosa']).value_counts()

# Correlation and covariance
iris.sepal_length.corr(iris.sepal_width)
iris.sepal_length.cov(iris.sepal_width)
iris.corr()
iris.cov()
iris.corrwith(iris.sepal_length)
