# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 18:13:12 2022

@author: Brandon Botzer - btb5103


Question 1:
Perform the following actions:

Use the randn function to create an array with a dimension of 5X5, and use a 
for loop to calculate the sum of all elements in the diagonal 
of the array. (25 points)


Choose any three functions to apply to this array. (25 points)
"""

print("Question 1: \n")

import numpy as np
#Set the seed to 862 for comparison
np.random.seed(862)

#declare the total variable
total = 0

#build the 5x5 grid
grid = np.random.randn(5, 5)

#Run through the diagonal
for i in range(len(grid)):
    #sum the diagonal components
    total += grid[i][i]
    
print("Here is the grid:")
print(grid)
print("\nThe sum total of the diagonal is: " + str(total))


#Apply three functions to the array
print("\nNow to do apply three functions to the array...")

#determine how sum collapses the array
print("\nWhen summing the grid, 'sum' collapses the columns as seen here:")
print(sum(grid))
print("\n")

#Perform the dreaded and computationally expensive transposition of the grid
print("The origional grid before transposition:")
print(grid)
print("\nNow to transpose the grid...\n")
print(np.transpose(grid))

#recall the shape of the array as a tuple
print("\n\nWhat is the shape of this grid?")
print(grid.shape)



"""
Question 2:
Perform the following actions:

Use x = np.random.randint(0, 1000, size = (10, 10)) to generate 10x10 array 
and use a for loop to find out how many even numbers are in it. (25 points)

Randomly generate a 8x9 array from a normal distribute with 
mean = 1, sigma = 0.5. Calculate the mean of elements whose indexes have 
a relation of (i+j)%5 == 0  (i is row index and j is column index).


* * Submit your Python file with your results to this assignment with the 
extension (.py) if you are using Spyder or your Jupyter Notebook with the 
extension (.ipynb). In both cases, make sure to upload the printout of your 
code file as PDF file so I can add my comments.

With that being said, you have to upload two files per assignment submission!
"""

print("\n\nQuestion 2: \n")
#Set the random seed 862 for comparison
np.random.seed(862)

#Generate the grid for the problem
x = np.random.randint(0, 1000, size = (10, 10))

#The running count of even numbers
count = 0

#Loop through both the rows and columns
#This could have also been written in multiple lines by nesting the loops
#and running through the ranges
for i, j in ((xi, xj) for xi in range(len(x)) for xj in range(len(x))):
    
    #Test for even
    if x[i][j] % 2 == 0:
        #increment the count
        count += 1
print("Here is the grid in question:\n")
print(x)
        
print("\nThe number of even numbers in the grid is: " + str(count) +"\n\n")


#Set the random seed 862 for comparison
np.random.seed(862)

sigma = 0.5
mean = 1

#create the 8x9 array
#using the formating from randn()
normal89 = sigma * np.random.randn(8, 9) + mean

#Calculate the mean of elements whose indexes ahve the relation
# (i+j) % 5 == 0

#delcare the total
total = 0
#declare the count for the division later
cnt = 0

print("The values to be used for the mean calculation are: \n")

#Loop through all of the elements in the 2D array
for i, j in ((xi, xj) for xi in range(len(normal89)) for xj in range(len(normal89))):
    
    if (i+j) % 5 == 0:
        #increase the total value
        total += normal89[i][j]
        #increment the cnt
        cnt += 1
        print(normal89[i][j])
        
#Divide the total by the cnt to get the mean

avg = total / cnt

print("\nThe total sum of (i+j)%5 == 0 values is: " + str(total))
print("The total number of values is: " + str(cnt))

print("\nThe mean of the used values is: " + str(avg))


































