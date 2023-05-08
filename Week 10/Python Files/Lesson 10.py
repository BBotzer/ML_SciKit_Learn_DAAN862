# -*- coding: utf-8 -*-
"""
Created on Fri May 25 23:08:52 2018

@author: Leo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split 
from sklearn import metrics

from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples = 60, centers = 2, 
                  random_state = 0, cluster_std = 0.5)

plt.scatter(X[:, 0], X[:, 1], c = y, s = 40)
plt.xlim(-1, 4)


# adding sepration lines
xfit = np.linspace(-1, 4)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, )
plt.plot([0.6], [2.1], 'x', color='red', markeredgewidth=2, markersize=10)

for m, b in [(1, 1), (0.5, 1.8), (-0.2, 2.9)]:
    plt.plot(xfit, m * xfit + b, '-k')

plt.xlim(-1, 4)


# Adding margin
xfit = np.linspace(-1, 4)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

for m, b, d in [(1, 1, 0.33), (0.5, 1.8, 0.55), (-0.2, 2.9, 0.35)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none',
                     color='#AAAAAA', alpha=0.4)

plt.xlim(-1, 4)

###########################################################################
os.chdir("J:\DSDegree\PennState\DAAN_862\Week 10\Python Files")
glass = pd.read_csv('glass.data', header = None)
glass.columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 
                 'Ca', 'Ba', 'Fe', 'Type']
# Classification
X_train, X_test, y_train, y_test = train_test_split(
        glass.iloc[:, 1 : 9], glass.Type, test_size=0.33, random_state=34)


# Support vector machine
# kernel = 'linear'
from sklearn import svm
svm_linear = svm.SVC(kernel = 'linear')
svm_linear.fit(X_train, y_train)

svm_linear.coef_
svm_linear_pred = svm_linear.predict(X_test)
metrics.accuracy_score(y_test, svm_linear_pred)

# kernel = 'rbf'
svm_rbf = svm.SVC(kernel = 'rbf', gamma = 0.1)
svm_rbf.fit(X_train, y_train)

svm_rbf_pred = svm_rbf.predict(X_test)
metrics.accuracy_score(y_test, svm_rbf_pred)

# kernel = 'poly'
svm_poly = svm.SVC(kernel = 'poly', degree = 2)
svm_poly.fit(X_train, y_train)

svm_poly_pred = svm_poly.predict(X_test)
metrics.accuracy_score(y_test, svm_poly_pred)





##############################################################################
# Ensemble methods: Random forests.
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators= 100, random_state = 0)
RF.fit(X_train, y_train)
RF_pred = RF.predict(X_test)
metrics.accuracy_score(RF_pred, y_test)
print(metrics.classification_report(y_test, RF_pred))
pd.DataFrame({'feature':glass.columns[1:9], 
              'importance':RF.feature_importances_})

# Accuracy vs n_estimator
from sklearn.model_selection import cross_val_score
n_estimator = range(2, 100, 2)
accuracy = []
for i in n_estimator:
    RF = RandomForestClassifier(n_estimators= i, random_state = 0)
    scores = cross_val_score(RF, X_train, y_train)    
    accuracy.append(scores.mean())

plt.figure()    
plt.plot(n_estimator, accuracy)
plt.title('Ensemble Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of base estimators in ensemble')

############################################################################
# Adaboost
from sklearn.ensemble import AdaBoostClassifier
Ada = AdaBoostClassifier(n_estimators = 20, learning_rate = 0.005,
                         random_state = 21)
Ada.fit(X_train, y_train)
Ada_pred = Ada.predict(X_test)
metrics.accuracy_score(y_test, Ada_pred)


# Accuracy vs n_estimator
from sklearn.model_selection import cross_val_score
n_estimator = range(1, 50, 1)
accuracy = []
for i in n_estimator:
    Ada = AdaBoostClassifier(n_estimators= i, learning_rate = 0.005,
                             random_state = 21)
    scores = cross_val_score(Ada, X_train, y_train)    
    accuracy.append(scores.mean())

plt.figure()    
plt.plot(n_estimator, accuracy)
plt.title('Adaboost Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of base estimators in ensemble')