# -*- coding: utf-8 -*-
"""
Created on Mon May  7 21:42:29 2018

@author: Leo
"""


import pandas as pd
import numpy as np
##############################################################################
# Hierarchical indexing
data = pd.Series(np.random.randn(9),
                 index=[['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'],
                        [1, 2, 3, 1, 3, 1, 2, 2, 3]])
data
data.index
data['b']
data['b' : 'c']
data.loc[['b', 'd']]
data.loc[:, 2]

data.unstack()
data.unstack().stack()

# column has a hierarchical index
frame = pd.DataFrame(np.arange(12).reshape((4, 3)),
                     index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                     columns=[['Ohio', 'Ohio', 'Colorado'],
                              ['Green', 'Red', 'Green']])
frame
frame.index.names = ['key1', 'key2']    #set index level names as key1 and key2
frame.columns.names = ['state', 'color']  # set column level names as state and color
frame
frame['Ohio']
# Reordering and Sorting Levels
frame.swaplevel('key1', 'key2')
frame.sort_index(level = 0)     # Sort frame by key1
frame.sort_index(level = 1)     # Sort frame by key2

# Summary Statistics by Level
frame.sum(level = 'key2')                  # Colomn sums for level = key2
frame.sum(level = 'color', axis = 1)       # Row sums for level = color


# Indexing with a DataFrame’s columns
frame2 = pd.DataFrame({'a': range(7), 'b': range(7, 0, -1),
                      'c': ['one', 'one', 'one', 'two', 'two','two', 'two'],
                      'd': [0, 1, 2, 0, 1, 2, 3]})
frame2
frame2_indexed = frame2.set_index(['c', 'd'])
frame2_indexed
frame2.set_index(['c', 'd'], drop=False)
frame2_indexed.reset_index()


##############################################################################
# Merge two dataset


df1 = pd.DataFrame({'key': list('aabccde'), 
                    'value1': range(7)})
df1
df2 = pd.DataFrame({'key': list('ace'), 
                    'value2': range(4, 7)})
df2
pd.merge(df1, df2)
pd.merge(df1, df2, on = 'key')

# keys have different names
df1.columns = ['key1', 'value1']   # reset column names
df1
df2.columns = ['key2', 'value2']   # reset column names
df2
pd.merge(df1, df2, left_on = 'key1', right_on = 'key2')
pd.merge(df1, df2, left_on = 'key1', right_on = 'key2', how = 'outer')


# Merge on index
df2 = df2.set_index('key2')       # use key2 column as index
df2

pd.merge(df1, df2, left_on = 'key1', right_index = True)

df1 = df1.set_index('key1')       # use key1 column as index
df1
pd.merge(df1, df2, left_index = True, right_index = True)


# Concatenating along axis
s1 = pd.Series([4, 8], index = list('AB'))
s1
s2 = pd.Series([2, 12, 4], index = list('DFG'))
s2
s3 = pd.Series([1, 9], index = list('JK'))
s3
pd.concat([s1, s2, s3])             # row binding
pd.concat([s1, s2, s3], axis = 1, sort = True)   # column binding  

# concate two data frame
df3 = pd.DataFrame(np.random.randn(4, 3), columns = list('ABC'))
df3
df4 = pd.DataFrame(np.random.rand(3, 4), 
                   columns = list('BCDE'),
                   index = range(5, 8))
df4
pd.concat([df3, df4], sort=True)

###############################################################################
# Reshape
data = pd.DataFrame(np.arange(6).reshape((2, 3)),
                    index=pd.Index(['Ohio', 'Colorado'], name='state'),
                    columns=pd.Index(['one', 'two', 'three'],
                                     name='number'))
data
result = data.stack()
result
type(result)
result.index
result.unstack()              # Unstack the inner level
result.unstack('state')       # Unstack the outer level

s1 = pd.Series([0, 1, 2, 3], index=['a', 'b', 'c', 'd'])
s2 = pd.Series([4, 5, 6], index=['c', 'd', 'e'])
data2 = pd.concat([s1, s2], keys=['one', 'two'])
data2
data2.unstack()
data2.unstack().stack()
data2.unstack().stack(dropna=False)

df = pd.DataFrame({'left': result, 'right': result + 5},
                  columns=pd.Index(['left', 'right'], name='side'))
df
df.unstack('state')
df.unstack('state').stack('side')




# Pivoting “Wide” to “Long” Format
df = pd.DataFrame({'key': ['foo', 'bar', 'baz'],
                   'A': [1, 2, 3],
                   'B': [4, 5, 6],
                   'C': [7, 8, 9]})
df
melted = pd.melt(df,  ['key'])
melted

reshaped = melted.pivot('key', 'variable', 'value')
reshaped

reshaped.reset_index()
pd.melt(df, id_vars = ['key'], value_vars = ['A', 'B'])

pd.melt(df, value_vars = ['A', 'B', 'C'])

