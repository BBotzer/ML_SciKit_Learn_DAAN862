# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 10:04:13 2022

@author: Brandon Botzer - btb5103
"""

"""

Question 1:
Upload Assignment4_data.csv  Download Assignment4_data.csvinto Python.

Please perform the following steps:

1) Explore the datasets. (10 points)
2) Find and handle missing values are in the data. (It is your choice how you handle the missing data.) ( 20 points)
3) Explore the variable column and Convert the "variable” column to dummy variables and join the dummies to the data. (20 points)
4) Convert the "one” column into 3 bins. (20 points)

"""


print("""
      Question 1:
      Upload Assignment4_data.csv  Download Assignment4_data.csvinto Python.

      Please perform the following steps:

      1) Explore the datasets. (10 points)
      2) Find and handle missing values are in the data. (It is your choice how you handle the missing data.) ( 20 points)
      3) Explore the variable column and Convert the "variable” column to dummy variables and join the dummies to the data. (20 points)
      4) Convert the "one” column into 3 bins. (20 points)
      """)


#imports (may not need all of these but better safe than sorry later)
import os
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import csv
from numpy import nan as NA


#Prevent pandas from displying all of the DF
pd.options.display.max_rows = 10

#Read in a CSV file
#Set the path for the CSV file

readPath = "J:\DSDegree\PennState\DAAN_862\Week 4\Homework"

#Change the directory
os.chdir(readPath)

#Read the CSV file in
data4 = pd.read_csv("Assignment4_data.csv")

print("The data frame to be used:\n")
print(data4)


#1) Explore the datasets. (10 points)
print("\n#1) Explore the datasets. (10 points)\n")


#print the first 5 rows
print("The data frame format:\n" + str(data4.head()))

#get the header info for later use
f = open('Assignment4_data.csv')
#Headers describing the data
h = list(csv.reader(f))[0]

#Check for duplicated data
dup = data4.duplicated()
#There are no duplicates in our data and without given more information as
#to what the data represents, we would not be dropping duplicates

print("\nStatisitcal data on the data frame:\n" + str(data4.describe()))



#2) Find and handle missing values are in the data. (It is your choice how you handle the missing data.) ( 20 points)

print("\n\n#2) Find and handle missing values are in the data. (It is your choice how you handle the missing data.) ( 20 points)\n")

#The missing data values are read in a NaNs.  
#I will fill them with the mean of each column
print("The missing data values are read in as NaNs.\nI will fill them with the mean of each column.\n")

#show the old data frame
print("Origional data frame: \n" + str(data4[6:9]))

#Fill the nan values with the means of the columns
#Only do this for the numeric columns (using the header) and ignore the categorical variable
data4 = data4.fillna(data4[h[0:5]].mean())

#Show the updated data frame
print("\nThe missing data has been filled with the column means: \n" + str(data4[6:9]))



#3) Explore the variable column and convert the "variable” column to dummy variables and join the dummies to the data. (20 points) (pg 208)

print("\n\n\n#3) Explore the variable column and convert the variable column to dummy variables and join the dummies to the data. (20 points)\n")

#Get the variable dummy matrix
varDummies = pd.get_dummies(data4["variable"])
print("The dummy matrix:\n" + str(varDummies))

#Join the dummy matrix with the data table.  
#Use the header values to indicate which values from data4 you'd like to keep
data4Dummies = data4[h[0:5]].join(varDummies)
print("\n\nThe new data frame with the dummy variables joined:\n" + str(data4Dummies))



#4) Convert the "one” column into 3 bins. (20 points) (pg 203)

print("\n\n\n#4) Convert the 'one' column into 3 bins. (20 points)\n")


#Get the one column data as a list so it will be a Categorical object
oneData = list(data4["one"])
#set the bins
bins = [-400, 0, 53, 100]
#Cut (bin) the data into the Categroical object (my own bin sizes)
onesVals = pd.cut(oneData, bins)
#Get the count in each bin
binCount = pd.value_counts(onesVals)
print("The bin counts are:\n" + str(binCount))

#I can also do this with automated evenly spaced bins
onesValsSpaced = pd.cut(oneData, 3)
#Get the count in each bin
binCountSpaced = pd.value_counts(onesValsSpaced)
print("\nWhen evenly spaced bins, the bin counts are now:\n" + str(binCountSpaced))

#I can also do this with even "quantiles" (Although they're are 3 not 4... so is it tritiles?)
onesValsQuants = pd.qcut(oneData, 3)
binCountQuants = pd.value_counts(onesValsQuants)
print("\nWhen more evenly distributed bin distributions, the bin counts are now:\n" + str(binCountQuants))






"""

Use the following speech by the Rev. Dr. Martin Luther King, Jr:

s = “I am happy to join with you today in what will go down in history as the greatest demonstration for freedom in the history of our nation. Five score years ago, a great American, in whose symbolic shadow we stand today, signed the Emancipation Proclamation. This momentous decree came as a great beacon light of hope to millions of Negro slaves who had been seared in the flames of withering injustice. It came as a joyous daybreak to end the long night of their captivity. But one hundred years later, the Negro still is not free. One hundred years later, the life of the Negro is still sadly crippled by the manacles of segregation and the chains of discrimination. One hundred years later, the Negro lives on a lonely island of poverty in the midst of a vast ocean of material prosperity. One hundred years later, the Negro is still languishing in the corners of American society and finds himself an exile in his own land. So we have come here today to dramatize a shameful condition."

1) Find out how many unique words in s. (10 points)
2) Which word appears the most? (10 points)
3) How many words start with ‘t’. (10 points).


"""


print("""
      \n\nQuestion 2:\n\nUse the following speech by the Rev. Dr. Martin Luther King, Jr:

      s = “I am happy to join with you today in what will go down in history as the greatest demonstration for freedom in the history of our nation. Five score years ago, a great American, in whose symbolic shadow we stand today, signed the Emancipation Proclamation. This momentous decree came as a great beacon light of hope to millions of Negro slaves who had been seared in the flames of withering injustice. It came as a joyous daybreak to end the long night of their captivity. But one hundred years later, the Negro still is not free. One hundred years later, the life of the Negro is still sadly crippled by the manacles of segregation and the chains of discrimination. One hundred years later, the Negro lives on a lonely island of poverty in the midst of a vast ocean of material prosperity. One hundred years later, the Negro is still languishing in the corners of American society and finds himself an exile in his own land. So we have come here today to dramatize a shameful condition."

      1) Find out how many unique words in s. (10 points)
      2) Which word appears the most? (10 points)
      3) How many words start with ‘t’. (10 points).
      """)


s = "I am happy to join with you today in what will go down in history as the greatest demonstration for freedom in the history of our nation. Five score years ago, a great American, in whose symbolic shadow we stand today, signed the Emancipation Proclamation. This momentous decree came as a great beacon light of hope to millions of Negro slaves who had been seared in the flames of withering injustice. It came as a joyous daybreak to end the long night of their captivity. But one hundred years later, the Negro still is not free. One hundred years later, the life of the Negro is still sadly crippled by the manacles of segregation and the chains of discrimination. One hundred years later, the Negro lives on a lonely island of poverty in the midst of a vast ocean of material prosperity. One hundred years later, the Negro is still languishing in the corners of American society and finds himself an exile in his own land. So we have come here today to dramatize a shameful condition."




#1) Find out how many unique words in s. (10 points)
print("\n\n1) Find out how many unique words are in s. (10 points)")

#Clean out commas and period punctuation while not adding extra spaces
s = s.replace(',', '')
s = s.replace('.', '')

#I am going to include words that are capitalized as the same word
#as their uncapitalized counterparts.  ie.  It == it

#Convert all words to lower case
s = s.lower()

#Split by spaces and strip the whitespace
words = [i.strip() for i in s.split(' ')]

#Find the unique words.  Numpy does this and sorts alphabetically
uniqueWords = np.unique(words)

#The number of unique words
uniqueCount = len(uniqueWords)

print("\nThe number of unique words in the Rev. Dr.'s speech is: " + str(uniqueCount))


#2) Which word appears the most? (10 points)
print("\n\n2) Which word appears the most? (10 points)\n")

#Create an 'empty' array
wordCount = np.empty(uniqueCount)

#fill the 'empty' array with the number of counts each word appears
for i in range(0, len(uniqueWords)):
    wordCount[i] = s.count(' '+ uniqueWords[i] + ' ')
    
#Join the two arrays into a dataframe
wordData = pd.DataFrame({'Word': uniqueWords, 
                         'Count': wordCount})

#Find the location (index) of the maximum count (most used word)
loc = wordData['Count'].idxmax()

#Most common word statistics
commWord = wordData.iloc[loc]

#Print out the information
print("The most common word is '" + str(commWord[0]) + "'.")
print("'" + str(commWord[0]) + "' is used " + str(commWord[1]) + " times within the speach.")


#3) How many words start with ‘t’. (10 points).
print("\n\n#3) How many words start with ‘t’. (10 points).\n")

#Create a new column for starting with a 't'
wordData.insert(2,'t start', np.zeros(uniqueCount))

#Assign True/False values to the 't start' column based on if the word starts with 't'
wordData['t start'] = wordData['Word'].str.startswith('t')

#Sum the 't start' column to get the total number of words that begin with 't'
tCount = wordData['t start'].sum()

#Print the information
print("The number of words that begin with the letter 't' is: " + str(tCount))









    


















