# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 10:35:56 2022

@author: Brandon Botzer - btb5103
"""


"""

Recall the datasets you used for SWENG 545 term project. (I do not as I have never taken that class... great)
All typos and inconsistency in course names have been cleaned. This time you will use Python to perform data understanding, cleaning and preprocessing. 

Perform the following actions:

1. Upload Registration.csv  Download Registration.csvand Course_info.xlsx  Download Course_info.xlsxinto Pandas. (5 points)
2. Explore and clean Registration data. (30 points)
3. Explore and clean Course_info data. (10 points)
4. Which course has the highest registration? (15 points)
5. Inner join two datasets. (20 points)
6. Create a data frame with student name as the index, course numbers as columns, and if the student registered a course as values(0, 1). ( 20 points)

"""


print("""
      Perform the following actions:

      1. Upload Registration.csv  Download Registration.csv and Course_info.xlsx  Download Course_info.xlsx into Pandas. (5 points)
      2. Explore and clean Registration data. (30 points)
      3. Explore and clean Course_info data. (10 points)
      4. Which course has the highest registration? (15 points)
      5. Inner join two datasets. (20 points)
      6. Create a data frame with student name as the index, course numbers as columns, and if the student registered a course as values(0, 1). ( 20 points)
      """)

#imports (may not need all of these but better safe than sorry later)
import os
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import csv
from numpy import NaN as NA

#regular expressions
import re


#1. Upload Registration.csv  Download Registration.csv and Course_info.xlsx  Download Course_info.xlsx into Pandas. (5 points)
print("\n\n1. Upload Registration.csv  Download Registration.csv and Course_info.xlsx  Download Course_info.xlsx into Pandas. (5 points)")

#Set the readpath
readPath = "J:\DSDegree\PennState\DAAN_862\Week 5\Homework"

#Change the directory
os.chdir(readPath)


#Read in the registration table
reglist = pd.read_csv("Registration.csv")

#Read in the course info
#I turned this into a CSV as the 'openpyxl' dependancy was
#giving me problems when I ran this on different machines
#It was a pathing issue...
clist = pd.read_excel('Course_info.xlsx')
#courselist = pd.read_csv('Course_info.csv')

print("\nDOWNLOADS ACCOMPLISHED!\n\nRegistration List:\n")
print(reglist.describe())
print("\nCourse List:\n")
print(clist.describe())



#2. Explore and clean Registration data. (30 points)
print("\n\n2. Explore and clean Registration data. (30 points)\n")

#Rename the data frame columns
reglist.columns = ['student_name', 'semester', 'course_name']

#convert all course names to upper and strip possible edge whitespace
reglist['course_name'] = reglist['course_name'].str.upper()
reglist['course_name'] = reglist['course_name'].str.strip()

#Sort the students alphabetically
reglist = reglist.sort_values('student_name')

#Drop dupplicates within the data frame
reglist = reglist.drop_duplicates()

#Reindex and drop the 'index' column
reglist = reglist.reset_index(drop = True)

print("The Registration Data has been explored and cleaned.  \nNote: We are currently not cleaning up the student names.  This would need to be done using REGEX listings.")
print(reglist)
print("\nCleaned Registration List:\n")
print(reglist.describe())



#3. Explore and clean Course_info data. (10 points)
print("\n\n3. Explore and clean Course_info data. (10 points)\n")

#strip the spaces off the course number
clist['Course number'] = clist['Course number'].str.strip()

#sort the list by the couse number in ascending order
clist = clist.sort_values('Course number')

#Drop the NaN 'unlisted course'
clist = clist.dropna(axis = 0)

#reset the list index (drop the 'index' column)
clist = clist.reset_index(drop = True)

#Rename the data frame columns (note to self, do this first next time...)
clist.rename(columns = {'Course number':'course_number', 'Course Name ': 'course_name', 'Course Type':'course_type'}, inplace = True)

#Convert all course names to uppercase and strip possible edge whitespace if it is there...
clist['course_name'] = clist['course_name'].str.upper()
clist['course_name'] = clist['course_name'].str.strip()

print("The Course Info data has been explored and cleaned.  We removed the non-existant course listing.")
print(clist)



#4. Which course has the highest registration? (15 points)
print("\n\n4. Which course has the highest registration? (15 points)\n")

#Gather all of the courses
courses = clist.course_name.unique()

#Create an empty array to store the unique courses counts
courseCounts = np.zeros(len(courses))

#Count the occurences of each unique course
for i in range(0, len(courses)):
    courseCounts[i] = np.count_nonzero(reglist.course_name == courses[i])

#Create Data frame of courses and the counts
courseData = pd.DataFrame({"courses":courses, 
                           "course_counts":courseCounts})

#Note to self: I tried to do this by creating the DF first
#and then iterating over each name in courses
    #for name in courses:
        #courseData.course_counts = np.count_nonzero(reglist.coursename == name)
#but I had an issue trying to put the count into the correct column location
#Counting and then assigning the DF proved to be the easier method but
#it perplexes me I do not know how to do this from the DF itself.


#Get the largest count id location and pass it to the courses to find the course name
popularCourse = courseData.courses[courseData.course_counts.idxmax()]

print("The most popular course is: " + str(popularCourse) + ".")
print("It is taken by " + str(int(courseData.course_counts.max())) + " students.\n")



#5. Inner join two datasets. (20 points)
print("\n\n5. Inner join two datasets. (20 points)\n")


#inner join the data sets (merge does inner by default)
joinData = pd.merge(reglist, clist)

#sort by student name
joinData = joinData.sort_values('student_name')

#reindex
joinData = joinData.reset_index(drop = True)

print("The two data frames have been inner joined.\n")
print(joinData)


#6. Create a data frame with student name as the index, course numbers as columns, and if the student registered a course as values(0, 1). ( 20 points)
print("\n\n6. Create a data frame with student name as the index, course numbers as columns, and if the student registered a course as values(0, 1). ( 20 points)\n")


#If I can pivot this so that the columns exist and populate with their own values,
#I can count the NaNs as 0 and replace the listed values with 1
#I need to figure out how to reshape this...



#create a new column vector poulated with ones
#I'll need this to aggregate (sum) the names later
joinData['value'] = np.ones(len(joinData))

#pivot the Table to unstack the courses stuck in the course_number column
#This pivot will index by student_name but it needs a way to deal with the duplicates
#Deal with the duplicates by summing the values which are 1 in all locations where
#a course is being taken by the person and 'NaN' where they are not taking the course
#This effectivly collapses the student_name index by name
#fill_value sets the NaNs to zeros
goodTab = joinData.pivot_table(values = 'value', index='student_name', columns = 'course_number', aggfunc='sum', fill_value=0)


#show the final data frame
print("Here is the final data frame for which students are taking which classes.\n")
print(goodTab)
























