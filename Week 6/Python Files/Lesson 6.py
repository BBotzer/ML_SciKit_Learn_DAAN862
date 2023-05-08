# -*- coding: utf-8 -*-
"""
Created on Thu May 10 13:38:15 2018

@author: Leo
"""

import matplotlib.pyplot as plt
import pandas as pd
import os


# Change the working directory
os.chdir("E:\GoogleDrive\PSU\DAAN862\Course contents\Lesson 6")
iris = pd.read_csv('iris.csv')


# line plot
# creat a new figure
plt.figure()
plt.plot(iris.sepal_length)


plt.figure()
plt.plot(iris.sepal_length, ls = '--', color = 'g', lw = 3)
plt.plot(iris.sepal_width, ls = '-.', color = 'blue', lw = 3 )
# or use plt.plot(iris.sepal_length, 'g--') 
plt.ylabel('Sepal Length')
plt.title('Iris data')
plt.ylim([0, 9])
plt.xticks([0, 50, 100, 150])
plt.legend()


# Scatter plots
plt.figure()
plt.scatter(iris.sepal_width, iris.sepal_length, s = 5, c = 'r')
plt.ylabel('Sepal Length')
plt.xlabel('Septal width')
plt.title('Iris Data')
plt.savefig('scatter.pdf')

# assign the color by its specy
plt.figure()
plt.scatter('sepal_width', 'sepal_length', data = iris, s = 7,
            c = iris.species.factorize()[0], label = iris.species.unique())
plt.ylabel('Sepal Length')
plt.xlabel('Septal width')
plt.title('Iris Data')
plt.xlim([1, 5])
plt.ylim([4, 9])

# histgram
plt.figure()
plt.hist(iris.sepal_length, 10, 
         density = 1,
         color = 'b',
         edgecolor = 'k' )
plt.xlabel('Sepal Length')
plt.ylabel('Probility density')
plt.title('Histogram', fontsize = 16, color = 'r')
plt.savefig('hist.png')


# mutiple subplots
fig, ax = plt.subplots(2, 2)
plt.subplot(2, 2, 1, )
plt.hist(iris.sepal_length, 10, density = 1, edgecolor = 'k')
plt.ylim(0, 1)
plt.yticks([0, 0.5, 1])
plt.xlabel('Sepal Length')


plt.subplot(2, 2, 2)
plt.hist(iris.sepal_width, 10, density = 1, edgecolor = 'k')
plt.ylim(0, 1)
plt.yticks([0, 0.5, 1])
plt.xlabel('Sepal Width')

plt.subplot(2, 2, 3)
plt.hist(iris.petal_length, 10, density = 1, edgecolor = 'k')
plt.ylim(0, 1)
plt.yticks([0, 0.5, 1])
plt.xlabel('Petal Length')

plt.subplot(2, 2, 4)
plt.hist(iris.petal_width, 10, density = 1, edgecolor = 'k')
plt.ylim(0, 1)
plt.yticks([0, 0.5, 1])
plt.xlabel('Petal Width')

plt.subplots_adjust(wspace = 0.3, hspace = 0.3)
plt.suptitle('A table of four subplots', 
             color = 'r', 
             y = 0.96, 
             fontsize = 14)

# you can also use:
fig, axs = plt.subplots(2, 2, figsize=(5, 5))
fig.tight_layout(w_pad = 1)
axs[0, 0].hist(iris.petal_length)
plt.xlabel('Petal Length')
plt.ylabel('Histogram')
axs[1, 0].scatter(iris.petal_width, iris.petal_length)
axs[0, 1].plot( iris.petal_length)
axs[1, 1].hist2d(iris.petal_width, iris.petal_length)
plt.subplots_adjust(wspace = 0.5, hspace = 0.5)



###############################################################################
# Pandas
# Line plots
iris.sepal_length.plot(color = 'r', label = 'Sepal length')
iris.sepal_width.plot(color = 'k', label = 'Sepal width')
plt.legend()


# Scatter plots
plt.style.use('ggplot')
colormap = iris.species.factorize()[0] 
iris.plot.scatter(x = 'petal_width', y = 'petal_length', 
                  c = colormap, s = 50)

# Scatterplot matrix
plt.style.use('classic')
pd.plotting.scatter_matrix(iris, c = colormap, s = 60, diagonal = 'kde')


# Histogram againest different class
fig, axs = plt.subplots(ncols = 2)
plt.tight_layout(w_pad = 1.2)  # modify the separation between figures
iris.groupby("species").petal_length.plot(kind='kde', ax = axs[1])
axs[1].set_xlabel('Petal Length')
iris.groupby("species").petal_length.hist(alpha=0.4, ax = axs[0])
axs[0].set_xlabel('Petal Length')
axs[0].set_ylabel('Histogram')

# box plots
color = dict(boxes='DarkGreen', whiskers='DarkOrange',
             medians='DarkBlue', caps='Gray')
iris_sub = iris.iloc[:, 0:4]
iris_sub.plot.box(color=color, sym='r+')
iris_sub.boxplot()
