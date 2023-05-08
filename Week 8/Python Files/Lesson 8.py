# -*- coding: utf-8 -*-
"""
Created on Fri May 11 23:58:33 2018

@author: QZS
"""

# Lesson 8 Classificatin methods
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split 
from sklearn import metrics

os.chdir("J:\DSDegree\PennState\DAAN_862\Week 8\Python Files")
glass = pd.read_csv('glass.data', header = None)
glass.columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 
                 'Ca', 'Ba', 'Fe', 'Type']

glass.describe()
glass.Type.value_counts()
glass.isnull().sum().sum()


glass.corr()

# plot correlation matrix
plt.matshow(glass.corr())
plt.title('Correlation Matrix', position = (0.5, 1.1))
plt.colorbar()
plt.xticks(range(11), list(glass.columns))
plt.yticks(range(11), list(glass.columns))
plt.ylabel('True label')
plt.xlabel('Predicted lable')

# explore the data

# Train test split
X = glass.iloc[:, 0 : 9]
y = glass.Type
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=34)


###############################################################################
# Logistic regression
from sklearn import linear_model
lr = linear_model.LogisticRegression()
lr.fit(X_train, y_train)

lr.coef_
lr.intercept_

lr_train_pred = lr.predict(X_train)
lr_test_pred = lr.predict(X_test)
metrics.accuracy_score(y_train, lr_train_pred)
metrics.accuracy_score(y_test, lr_test_pred)
train_cm = metrics.confusion_matrix(y_train, lr_train_pred)
train_cm
test_cm = metrics.confusion_matrix(y_test, lr_test_pred)
test_cm

plt.figure()
plt.matshow(test_cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted lable')



##############################################################################
# Naive Bayes
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(X_train, y_train)

NB_train_pred = NB.predict(X_train)
NB_test_pred = NB.predict(X_test)

# train size vs test accuracy
metrics.accuracy_score(y_train, NB_train_pred)
metrics.accuracy_score(y_test, NB_test_pred)

metrics.confusion_matrix(y_train, NB_train_pred)
metrics.confusion_matrix(y_test, NB_test_pred)

# predict the probility for the top 5 rows of test data.
np.set_printoptions(precision=2)  # display 2 decimal places
NB.predict_proba(X_test[:5])


# the relation between training size and accuracy for both models.
lr = linear_model.LogisticRegression()
nb = GaussianNB()
lr_scores = []
nb_scores = []
train_sizes = range(120, len(X)-20, 10)

from sklearn.model_selection import cross_val_score

for train_size in train_sizes:
    X_trains, _, y_trains, _ = train_test_split(
            X, y, train_size=train_size,random_state=11)
    print
    nb_accuracy = cross_val_score(nb, X_trains, y_trains, cv = 10).mean()
    nb_scores.append(nb_accuracy)
    lr_accuracy = cross_val_score(lr, X_trains, y_trains, cv = 10).mean()
    lr_scores.append(lr_accuracy)

lr_scores
nb_scores

plt.figure()    
plt.plot(train_sizes, nb_scores, label='Naive Bayes')
plt.plot(train_sizes, lr_scores, linestyle='--', label='LogisticRegression')
plt.title("Naive Bayes and Logistic Regression Accuracies")
plt.xlabel("Number of training instances")
plt.ylabel("Test set accuracy")
plt.legend()

##############################################################################
# Decision trees
from sklearn import tree
DT = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5)
DT.fit(X_train, y_train)

DT_pred = DT.predict(X_test)
metrics.accuracy_score(DT_pred, y_test)
print(metrics.classification_report(y_test, DT_pred))

# feature importance
DT.feature_importances_
pd.DataFrame({'variable':glass.columns[:9],
              'importance':DT.feature_importances_})


# to view the tree
from graphviz import Source
dot_data = tree.export_graphviz(DT, out_file=None, 
                                feature_names=X_train.columns)
Source(dot_data)


# different split creteria


##############################################################################
# Neural networks
# rescale data to [0, 1]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled[:3]
X_test_scaled[:3]

from sklearn.neural_network import MLPClassifier
NN = MLPClassifier(solver = 'lbfgs', alpha = 1e-5, 
                   hidden_layer_sizes = (10, 4), random_state = 1)
NN.fit(X_train_scaled, y_train)

NN_pred = NN.predict(X_test_scaled)
metrics.accuracy_score(NN_pred, y_test)
print(metrics.classification_report(y_test, NN_pred))


