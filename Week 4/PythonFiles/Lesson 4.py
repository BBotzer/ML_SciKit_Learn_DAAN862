# -*- coding: utf-8 -*-
"""
Created on Mon May  7 21:42:29 2018

@author: Leo
"""


import pandas as pd
import numpy as np


# Create a random data and randomly assign 3 nan 
np.random.seed(123)
data = np.random.normal(2, 2, 20)
data[2] = None

data[np.random.randint(3, 20, 3)] = np.nan
data = data.reshape(4, 5)
df = pd.DataFrame(data, columns = list('abcde'))
df

df.isnull()
df.isnull().sum(axis = 0)   # count nans in each column
df.isnull().values.sum()      # count total nans

df.dropna()             # drop rows contain nans
df.dropna(axis = 1)     # drop columns contain nans
df.dropna(how = 'all')  # drop rows whose all of values are nan

df.fillna(0)
# use a different fill value for each column
df.fillna({'b': 0.5, 'c': 0, 'e': 2}) 
df.fillna(method = 'ffill') # use previous row values to fill nan
df.fillna(df.mean())

# Removing duplicates
data = pd.DataFrame({'k1': ['one', 'two'] * 2 + ['two'],
                      'k2': [1, 1, 3, 4, 4]})
data

data.duplicated()
data.drop_duplicates()

data['k3'] = range(5)
data.drop_duplicates(['k1'])

data = pd.DataFrame({'City': ['New York', 'Chichago', 'Berlin', 
                               'London', 'Toronto', 'Manchester'], 
                     'Variable': [4, 3, 12, 6, 7.5, 8]})
data

# a dictionary use cities as keys and countries it belongs to as value
city_country = {'New York': 'USA',
                'Chichago': 'USA', 
                'Berlin': 'Germany',
                'London': 'UK',
                'Toronto': 'Canada',
                'Manchester': 'UK'}

# Transform a column by a dict
data['Country'] = data.City.map(city_country)
# Transform a column by function lambda
data['City'].map(lambda x: city_country[x])

# Replacing values
# Create a data and randomly assign -100 to three elements.
np.random.seed(189)
data = np.random.randint(-20, 20, 12)
data[np.random.randint(0, 11, 3)] = -100
data = data.reshape(4, 3)
df = pd.DataFrame(data, columns = list('ABC'))
df

# Replace -100 with NAN
df.replace(-100, np.NAN)

# Detecting and filtering outliers
np.random.seed(123)
data = pd.DataFrame(np.random.randn(1000, 4))
data.describe()

col2= data[2]
col2[np.abs(col2) > 3] # larger than 3 standard deviation

data[(np.abs(data) > 3).any(1)]

# set the value of outliers to 3 or -3 
data[(np.abs(data) > 3)] = np.sign(data) * 3 
data.describe()

# Discretization and Binning
values = np.random.randint(18, 54, 10)
values
bins = np.array([0] + list(range(1, 5))+ [np.inf]) * 10
bins
cats = pd.cut(values, bins)
cats
cats.codes
cats.categories
pd.value_counts(cats)
pd.cut(values, 5, precision = 2)

values2 = np.random.randn(100)
cats2 = pd.qcut(values2, 4)
cats2
pd.value_counts(cats2)
pd.value_counts(pd.qcut(values2, [0, 0.1, 0.5, 0.9, 1]))

# Dummy variable
df = pd.DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],  
                   'data1': range(6)})
pd.get_dummies(df['key'])

dummies = pd.get_dummies(df['key'], prefix='key')
df_with_dummy = df[['data1']].join(dummies)
df_with_dummy

# String operation
s = 'Python is a great tool for data analysis'
# split string by white space
words = s.split(' ')

# converting all words to uppercase
[w.upper() for w in words]

 # Convert list of words to single string 
' '.join(words)


# Membership
'data' in s
s.index('a')
s.find('new')
s.count('for')
s.replace('data', 'Data')

# pandas

mtcars = pd.read_csv('E:\GoogleDrive\PSU\DAAN862\Course contents\Lesson 4\mtcars.csv')
sub_mtcars = mtcars.iloc[0:5, 0:4]
sub_mtcars


sub_mtcars['company'] = sub_mtcars.model.str.split(' ', expand=True)[0]
sub_mtcars

sub_mtcars.model.str.contains('Mazda')
sub_mtcars.model.str.startswith('M')

sub_mtcars.model.str[:4]
