# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 13:29:50 2018

@author: Leo
"""

import numpy as np

# 2.1 Array creation
np.array(range(10))         # convert a list to array
A = np.array([[1,2], [3,4]]) # convert a list of lists to 2-D array.
print(A)
np.zeros( (3,4) )           #Create a 3x4 zero array
np.zeros_like(A)            # Create an array of 0s with the same shape as A
np.ones((2,3,4), dtype=np.int16)   # create a 2x3x4 array with 1s
np.ones_like(A)             # Create an array of 1s with the same shape as A
np.empty((2,3))             # Create an 2x3 empty array
np.empty_like(A)            # Create an empty array with the same shape as A
np.arange(10, 30, 5)        # create an array with evenly spaced values
np.arange(0, 2, 0.3)      
np.linspace(0, 2, 9)        # create an array which divides the interval evenly

 
np.random.seed(123)         # set the random seed
# create an 3x2 array randomly sampled from a uniform distrbution
np.random.rand(3,2)         
# create an 3x4 arry randomly sampled from a normal distribution
np.random.randn(3, 4)       


# Array attributes
a = np.arange(12).reshape(3, 4)
a
a.shape                 # shape 
a.ndim                  # how many dimension
a.dtype.name            # data type
a.size                  # how many items 
type(a)



# Array indexing
# one-dimensional array
a = np.arange(6)**2 
a
a[2]                    # Select the 3rd element
a[2:5]                  # Select the 3rd - 5th elements
# equivalent to a[0:6:2] = -1000; from start to position 6, exclusive, 
# set every 2nd element to -1000
a[:6:2] = -100     
a[::-1]                 # reverse the array


# two-dimensional array
def f(x, y):
    return 10*x + y

b = np.fromfunction(f, (5, 4), dtype = int) # create an array from the function
b
b[2,3]                  # select the item at 2rd row and 3nd column
b[0:3, 1]               # select row 0-2 in the 1st column
b[ : ,1]                # Select the 2nd column
b[1:3, : ]              # all columns in the second and third row of b
b[-1]                   # the last row. Equivalent to b[-1,:]


# Boolean indexing
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
names
data = np.random.randn(7, 4)
data

names == 'Bob'          # Boolean array
data[names == 'Bob']    # select all columns with rows when names == 'Bob'
data[names != 'Bob', 3] # inverse selection


# Array math_elementwise
a = np.array( [12, 43, 4, 89, 65] )
b = np.arange( 5 )
c = a + b               # assign elementwise sum of a and b to c
c
a * b
b**3
2 * np.exp(b)
a >= 20

#statistic operations
a = np.random.random((2,3))
a
a.sum()                 # sum of all items
a.sum(axis = 0)         # column sum
a.sum(axis = 1)         # row sum
np.sum(a)               # equivalent to a.sum()
np.sum(a, axis = 0)     # equivalent to a.sum(axis = 0)
a.min()                 # minimum of a
a.min(axis = 0)         # minimun for each column
np.min(a)               # equivalent to a.min()
a.max()                 # maximum of a
np.max(a)               # equivalent to a.max()
a.std()                 # standard deviation of a
a.std(axis = 0)         # standard diviation for each column
np.std(a)               # equivalent to a.std()

# Linear algebra
A = np.array( [[1,1], [0,1]] )
B = np.array( [[2,0], [3,4]] )
A*B                      # elementwise product
A.dot(B)                 # matrix product
np.dot(A, B)             # matrix product
A.T                      # Transpose a matrix

# Universal functions

A = np.arange(12).reshape((3,4))
np.sqrt(A)              # compute square root of each element
B = np.exp(A)           # compute the exponential e^x for each element
np.modf(B)              # compute factional and integral part of array 
np.floor(B)             # compute floor of each element      



