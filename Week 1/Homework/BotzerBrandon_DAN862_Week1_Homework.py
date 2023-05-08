
"""
Created on Mon Aug 22 16:47:53 2022

@author: Brandon Botzer - btb5103

DAN 862 Fall 2022

Week 1 Homework 1
"""



#Question 1

#Run the code provided to generate the list L1
import numpy as np

L1 = []

np.random.seed(56)

for i in np.random.randint(0, 100, 10):

    L1.extend([i] * np.random.randint(0, 100, 1)[0])

np.random.shuffle(L1)

print("Question 1: \n")

#What are the unique values? (5pts)
print("What are the unique values? (5pts)")

#Generate the set of numbers
L1_Unique = set(L1)
#Print the set
print(L1_Unique)



#How many unique values?  (5pts)
print("How many unique values?  (5pts)")
#Find the length of the L1_Unique set
print(len(L1_Unique))



#Create a dictionary with the unique items in L1 as dictionary keys
#and their count as the dictionary values (20pts)

#use the count for a list then make a dict

#Make a list of the unique items so I can index them
unique_List = list(L1_Unique)

countsList = []

#Iterate over this and zip the dicts together
for i in range(0, len(unique_List)):
    
    #add the number of times a value appears into the count list
    counts = L1.count(unique_List[i])

    countsList.append(counts)    

#build a tuple by zipping the unique List and the number of times each unique
#shows up

zippedUp = zip(unique_List, countsList)

unique_Dict = dict(zippedUp)

print("The Dictionary of the unique values and the number of times they appear:")
print(str(unique_Dict))




#Which values appears most freqeuntly?  Don't do this by hand... (10pts)

#This could be done in one line but I show it here in 3 for clarity

#Find the max appearances
mostShows = max(countsList)

#find the index of the most shows
indmostShows = countsList.index(mostShows)

#Use this index in the dict, list, tuple
print("The value which appears most frequently is: " + str(unique_List[indmostShows]))
print("It appears a total of " + str(mostShows) + " times!")


print("\n\n\nQuestion 2: \n")
#Question 2


L2 = [879, 394, 235, 580, 628, 81, 206, 238, 927, 853, 622, 603, 110, 143, 824, 324, 343, 506, 634, 325, 258, 900, 960, 286, 449, 890, 921, 170, 888, 851]

#copied into L3 so I can use L2 again
L3 = L2.copy()

#Use a while loop to calculate the sum of the even numbers in L2 (10 pts)

#Declare x to store the sum of the evens
x = 0

#While L2 does not equal an empty list
while L3 != []:
    y = L3.pop()
    #Test the pop'd value for even
    if y % 2 == 0:
        #Add the pop'd value to the running total
        x += y
    
print("The sum of the even numbers in L2 is: " + str(x))





#Write a function to caculate the mean of a list.  Use this function to
#calculate the mean of L2 (10 pts)


def meanMean(L):
    
    #Declare total
    total = 0
    
    #Add up all of the elements
    for i in range(0, len(L)):
        total += L[i]
             
    #Divide the total by the number of elements
    
    result = total / len(L)
    
    return result

#Print the answer with a call to meanMean()
print("The mean of L2 is: " + str(meanMean(L2)))




#Calculate the sum for elements in L2 which ARE larger than 500 (10 pts)


#copied into L3 so I can use L2 again
L3 = L2.copy()

#Declare x to store the sum of the evens
x = 0

#While L2 does not equal an empty list
while L3 != []:
    y = L3.pop()
    #Test the pop'd value for even
    if y > 500:
        #Add the pop'd value to the running total
        x += y
    
print("The sum of the numbers larger than 500 in L2 is: " + str(x))




print("\n\n\nQuestion 3: \n")

#Question 3

#Write the power function.  Don't use **


#There are two ways to do this.  If n is an integer, you can loop through
#And multiply x by itself.  This requries you to check for 'n' being 
#an int as well as taking the absolute value of 'n' for the loop.  However,
#you'll be unable to solve with non-integer values of 'n'

#Instead, I will do this for n being any real value and use log rules
#with numpy's exponent and log features (standard base is for ln)


def botz_Pow(x, n):
    
    y = np.exp(n * np.log(x))    
    
    return y

print("Two to the power of 10 is: " + str(botz_Pow(2,10)))

print("Three to the power of negative 3 is: " + str(botz_Pow(3,-3)))





