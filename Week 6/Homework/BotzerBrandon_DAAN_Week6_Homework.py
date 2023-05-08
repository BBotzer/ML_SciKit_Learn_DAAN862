# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 18:10:08 2022

@author: Brandon Botzer - btb5103
"""


"""
Assignment:
    
1. Plot am-based histogram to compare mpg (20 points)
2. Use scatterplot to plot mpg VS. hp (20 points)
3. Create a scatterplot matrix for new data consisting of columns [disp, hp, drat, wt, qsect]. (20 points)
4. Create boxplots for new data consisting of columns [disp, hp, drat, wt, qsect]. (20 points)
5. Use plots to answer which variable has the most impact on mpg. (20 points)


A note about plotting:  I was having some trouble with the interactive
plotting in Spyder.  I was unable to click and select any of the plots
to zoom or pan.  I did modify the settings as shown in the online notes
but this caused no plots to show up.

I ended up using Activate Support, Autoload pylab and NumPy, and Inline backend
just to get the plots to display.  While still unable to dynamically interact,
Jupyter at least loaded the plots inline properly.

"""

#imports (may not need all of these but better safe than sorry later)
import os
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import csv
from numpy import NaN as NA

#Import a slew of plotting functions to play with
import matplotlib.pyplot as plt
import seaborn as sns

#Had to install plotly first and didn't really use much for this work outside of testing
import plotly.express as px

#regular expressions
import re




#Set the path for the CSV file
readPath = "J:\DSDegree\PennState\DAAN_862\Week 6\Homework"

#Change the directory
os.chdir(readPath)

#Read the CSV file in
mtcars = pd.read_csv("mtcars.csv")

print(mtcars)

#Note to self:  plt does not like to script in one off lines in the console
#It will compile correctly though


#1. Plot am-based histogram to compare mpg (20 points)

#Split am = 1 and am = 0, plot those two sets mpg as a histogram
#Set the alpha to less than 1 to make the histograms transparent
mtcars.groupby("am").mpg.hist(alpha = 0.4)



#Trying to do this with plotly... having a tough time of it
#Need to seperate out the 'am' data but I can't just pass the groupby
#y = mtcars.groupby("am").mpg
#plt.figure()
#plt.hist(y, histtype='barstacked')



#2. Use scatterplot to plot mpg VS. hp (20 points)

plt.figure()
plt.scatter(mtcars.hp, mtcars.mpg)
plt.ylabel("MPG")
plt.xlabel("HP")
plt.title("Economy vs Power")





#3. Create a scatterplot matrix for new data consisting of columns [disp, hp, drat, wt, qsec]. (20 points)

#I was playing around with different pairplots here.

#point data with diagonal kde
sns.pairplot(mtcars, vars = ['disp', 'hp', 'drat', 'wt', 'qsec', 'mpg'], diag_kind='kde')

#point data only
sns.pairplot(mtcars, vars = ['disp', 'hp', 'drat', 'wt', 'qsec', 'mpg'])

#All data kde (contors)
sns.pairplot(mtcars, vars = ['disp', 'hp', 'drat', 'wt', 'qsec', 'mpg'], kind='kde')


#ugly plot and not useful
#sns.pairplot(mtcars, vars = ['disp', 'hp', 'drat', 'wt', 'qsec', 'mpg'], kind='hist')

#4. Create boxplots for new data consisting of columns [disp, hp, drat, wt, qsect]. (20 points)

#get just the relevant columns
test = mtcars[['mpg','disp', 'hp', 'drat', 'wt', 'qsec']]

#One boxplot with all of the column values through it
plt.figure()
plt.boxplot(test)


#get just the relevant columns
test = mtcars[['mpg','disp', 'hp', 'drat', 'wt', 'qsec']]
#Plot six individual boxplots as the scaling is too wide on the previous
fig, axs = plt.subplots(3, 2, figsize=(5, 5))
fig.tight_layout(w_pad = 1)
axs[0, 0].boxplot(test.disp)
axs[0, 0].set_title('Displacement')
axs[1, 0].boxplot(test.hp)
axs[1, 0].set_title("Horse Power")
axs[0, 1].boxplot(test.drat)
axs[0, 1].set_title("D Rat")
axs[1, 1].boxplot(test.wt)
axs[1, 1].set_title("Weight")
axs[2, 1].boxplot(test.qsec)
axs[2, 1].set_title("Q sec")
axs[2, 0].boxplot(test.mpg)
axs[2, 0].set_title("MPG")
plt.subplots_adjust(wspace = 0.3, hspace = 0.4)




#5. Use plots to answer which variable has the most impact on mpg. (20 points)

#Plot a pairplot of the 6 variables with diagonals being kde
g = sns.pairplot(mtcars, vars = ['disp', 'hp', 'drat', 'wt', 'qsec', 'mpg'], diag_kind='kde')
#set the lower grids to also have kde contors
g.map_lower(sns.kdeplot, levels = 8)

#Based on the graphs, we can see that the 'wt' variable
#has the tightest contors with the 'mpg' variable
#Thus, there should be the highest correlation between them.
print("""\n\nBased on the graphs, we can see that the 'wt' variable has the tightest contors with the 'mpg' variable.\n
      Thus, there should be the highest correlation between them.""")
      


