# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 12:59:26 2022

@author: Brandon Botzer - btb5103

"""


"""
Perform the following actions:

Import data mtcars.csv  Download mtcars.csv into Python. (10 points)
"""

#import the relevant libraries / packages
import os
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import csv


#Set the path for the CSV file

readPath = "J:\DSDegree\PennState\DAAN_862\Week 3\Homework"

#Change the directory
os.chdir(readPath)

#Read the CSV file in

mtcars = pd.read_csv("mtcars.csv")

print(mtcars)


"""
Explore the data and perform a statistical analysis of the data. (30 points)
"""

print("Explore the data and perform a statistical analysis of the data. (30 points)")

#I'll look at MPG for the statistical analysis:
    #A classic statistical analysis consists of:
        #mean, min, q1, median, q3, max, std dev across all car types


avgMPG = mtcars['mpg'].mean()
stdMPG = mtcars['mpg'].std()
minMPG = mtcars['mpg'].min()
q1MPG = mtcars['mpg'].quantile(0.25)
meadMPG = mtcars['mpg'].median()
q3MPG = mtcars['mpg'].quantile(0.75)
maxmPG = mtcars['mpg'].max()

print("Average MPG: " + str(avgMPG) + 
      "\nFive number summary: " + str(minMPG) + ", " + str(q1MPG) + ", " + 
      str(meadMPG) + ", " + str(q3MPG) + ", " + str(maxmPG) + 
      "\nStandard Deviation: " + str(stdMPG))



#Function to be used later... probably should have used it earlier...
def stats_Analysis(dataColumn):
    
    avg = dataColumn.mean()
    std = dataColumn.std()
    minimum = dataColumn.min()
    q1 = dataColumn.quantile(0.25)
    mead = dataColumn.median()
    q3 = dataColumn.quantile(0.75)
    maximum = dataColumn.max()
    
    print("Average: " + str(avg) + 
          "\nFive number summary: " + str(minimum) + ", " + str(q1) + ", " + 
          str(mead) + ", " + str(q3) + ", " + str(maximum) + 
          "\nStandard Deviation: " + str(std))

"""
Analyze mpg for cars with different gears, and show your findings. (20 points)
"""

print("\n\nAnalyze mpg for cars with different gears, and show your findings. (20 points)")

#Perform statistical analysis again based on gear instead of type


#Create a new trimmed array with just mpg and gear data
#gearData = mtcars[['mpg', 'gear']]   #Used for testing

#First determine which unique gear listings exist
uniqueGears = mtcars['gear'].unique()
#sort these in ascending
uniqueGears.sort()

#Full data of Gear 4 (can change mtcars -> gearData)
#test = mtcars[mtcars['gear'] == uniqueGears[0]]   #Used for testing

for i in range(0,len(uniqueGears)):
    
    print("Here is the statistical analysis for MPG based on " + str(uniqueGears[i]) + " Gears:")
    
    #remade a dataColumn of the data to be analyzed (mpg) for easier changes in the future
    dataC = mtcars['mpg'][mtcars['gear'] == uniqueGears[i]]
    
    #Run the statistical analysis function
    stats_Analysis(dataC)
    
    #Spacer
    print()
    

"""
Analyze mpg for cars with different carb, and show your findings. (20 points)
"""

print("\n\nAnalyze mpg for cars with different carb, and show your findings. (20 points)")

#Perform statistical analysis again based on carb instead of gear / type

#Create a new trimmed array with just mpg and gear data
#gearData = mtcars[['mpg', 'gear']]   #Used for testing

#First determine which unique gear listings exist
uniqueCarb = mtcars['carb'].unique()
#sort these in ascending
uniqueCarb.sort()

#Full data of Gear 4 (can change mtcars -> gearData)
#test = mtcars[mtcars['gear'] == uniqueGears[0]]   #Used for testing

for i in range(0,len(uniqueCarb)):
    
    print("Here is the statistical analysis for MPG based on " + str(uniqueCarb[i]) + " Carbs:")
    
    #remade a dataColumn of the data to be analyzed for easier changes in the future
    dataC = mtcars['mpg'][mtcars['carb'] == uniqueCarb[i]]
    
    #Run the statistical analysis function
    stats_Analysis(dataC)
    
    #Spacer
    print()



"""
Find out which attribute has the most impact on mpg. (20 points)
"""

print("\n\nFind out which attribute has the most impact on mpg. (20 points)")


#Compute the Correclation matrix

corrCoef = mtcars.corr()

print("The Correlation Matrix: \n " + str(corrCoef))

#Compute the Covariance matrix

covCoef = mtcars.cov()

print("\nThe Covariance Matrix:" + str(covCoef))

#Note: the correlation is the covariance divided by the std dev of the
#two std devs being compared.  Thus I'll stick to the correlation Coef
#for my final analysis
print("""\n\nNote: the correlation is the covariance divided by the two std devs 
      of the variables being compared.  Thus I'll stick to the 
      correlation Coef for my final analysis\n\n""")



#We can display all of the relevant correlation Coef
#I have dropped the mpg corr value here as it is natually 1.0 and irrelevant
print("We can display all of the relevant correlation coef")
print(str(corrCoef['mpg'].drop('mpg')))

#If we are just looking at the MOST impact, let us look at these from an absolute value
print("\nAbsolute values of the corr coef:\n" + str(corrCoef['mpg'].drop('mpg').abs()))

#If we'd like these ordered
print("\nOrdered absolute values of the corr coef: \n" + str(corrCoef['mpg'].drop('mpg').abs().sort_values(ascending=False)))


#Largest correclation Coefficient
print("\nThe largest value of the correlation coefficients is from: " + str(corrCoef['mpg'].drop('mpg').abs().idxmax()))













































