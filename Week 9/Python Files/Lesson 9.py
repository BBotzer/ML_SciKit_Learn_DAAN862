# -*- coding: utf-8 -*-
"""
Created on Wed May 16 09:49:37 2018

@author: Leo
"""


# Lesson 9 Regression models
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np

#############################################################################
# Regression models
path = "J:\DSDegree\PennState\DAAN_862\Week 9\Python Files"
os.chdir(path)
mtcars = pd.read_csv("mtcars.csv")
mtcars.describe()


# Train_test split
# change the dimension from (32,) to ((32, 1))
X = np.expand_dims(mtcars.disp, axis = 1) 
y = mtcars.mpg
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                    random_state = 123)
# logistic regression
from sklearn import linear_model
lreg = linear_model.LinearRegression()
lreg.fit(X_train, y_train)

# the coefficient
lreg.coef_
lreg.intercept_

# model evaluation
lreg_train_pred = lreg.predict(X_train)
lreg_test_pred = lreg.predict(X_test)
metrics.r2_score(y_train, lreg_train_pred)
metrics.mean_squared_error(y_train, lreg_train_pred)
metrics.r2_score(y_test, lreg_test_pred)
metrics.mean_squared_error(y_test, lreg_test_pred)

# plot the data and the linear model
plt.figure()
plt.scatter(X, y, color = 'black', label = 'True values')
plt.plot(X, lreg.predict(X), color = 'blue', linewidth = 3, 
         label = 'Linear regression model')
plt.xlabel('Displacement')
plt.ylabel('MPG')
plt.legend()

# Residual Analysis
y_pred = lreg.predict(X)
residual = y - y_pred
plt.figure()
plt.scatter(y_pred, residual)
plt.hlines(0, xmin = 10, xmax = 30)
plt.xlim([10, 30])
plt.xlabel('Predicted values')
plt.ylabel('Residual')

# QQ plot
from scipy.stats import probplot
plt.figure()
probplot(residual, plot = plt);


#############################################################################
# Regression tree model
X = mtcars.loc[:, 'cyl':]  # select the columns from 'cyl' to the end
y = mtcars.mpg
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                    random_state = 123)

from sklearn import tree
tree_model = tree.DecisionTreeRegressor(min_samples_leaf = 5,
                                        random_state = 39)
tree_model.fit(X_train, y_train)

from graphviz import Source
Source(tree.export_graphviz(tree_model, out_file=None, 
                            feature_names=X.columns))

# model evaluation
tree_train_pred = tree_model.predict(X_train)
metrics.mean_squared_error(y_train, tree_train_pred)
metrics.r2_score(y_train, tree_train_pred)

tree_test_pred = tree_model.predict(X_test)
metrics.mean_squared_error(y_test, tree_test_pred)
metrics.r2_score(y_test, tree_test_pred)

# Second model use min_samples_leaf = 8 instead of 5
tree_model1 = tree.DecisionTreeRegressor(min_samples_leaf = 10,
                                        random_state = 39)
tree_model1.fit(X_train, y_train)

Source(tree.export_graphviz(tree_model1, out_file=None, 
                            feature_names=X.columns))

# model evaluation
tree_train_pred1 = tree_model1.predict(X_train)
metrics.mean_squared_error(y_train, tree_train_pred1)
metrics.r2_score(y_train, tree_train_pred1)

tree_test_pred1 = tree_model1.predict(X_test)
metrics.mean_squared_error(y_test, tree_test_pred1)
metrics.r2_score(y_test, tree_test_pred1)


###########################################################################
# Neural network

# convert gear and carb to dummy variables
mtcars.columns
dummies_gear = pd.get_dummies(mtcars['gear'], prefix ='gear')
dummies_gear[:3]
dummies_carb = pd.get_dummies(mtcars['carb'], prefix = 'carb')
dummies_carb[:3]
mtcars_dummies = mtcars.iloc[:, 1:10].join(dummies_gear)
mtcars_dummies = mtcars_dummies.join(dummies_carb)
mtcars_dummies.columns

# You can also convert these three column to categorical first,
# then convert to dummy together
# mtcars.carb = mtcars.carb.astype('category')
# mtcars.gear = mtcars.gear.astype('category')
# mtcars_dummies = pd.get_dummies(mtcars, columns = ['gear', 'carb'])


# Rescale data to [0, 1]
from sklearn.preprocessing import MinMaxScaler  
train, test = train_test_split(mtcars_dummies, test_size = 0.3,
                               random_state = 123)
scaler = MinMaxScaler()
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)
X_train_scaled = train[:, 1:]
X_test_scaled = test[:, 1:]
y_train_scaled = train[:, 0]
y_test_scaled = test[:, 0]


from sklearn import neural_network 
nn_model = neural_network.MLPRegressor(10, activation = 'logistic',
                                       max_iter = 10000, random_state = 21)
nn_model.fit(X_train_scaled, y_train_scaled)
nn_model.coefs_
nn_model.intercepts_

nn_train_pred = nn_model.predict(X_train_scaled)
metrics.r2_score(y_train_scaled, nn_train_pred)
metrics.mean_squared_error(y_train_scaled, nn_train_pred)

nn_test_pred = nn_model.predict(X_test_scaled)
metrics.r2_score(y_test_scaled, nn_test_pred)
metrics.mean_squared_error(y_test_scaled, nn_test_pred)


# Optimize the neural network model
from sklearn.model_selection import GridSearchCV
tuned_parameters = {'hidden_layer_sizes':[10, 15, 20, 25, 30, 35, 40, 45, 50],
                    'activation': ['logistic', 'tanh','relu']}

# use mse as metrics
nn = neural_network.MLPRegressor(max_iter =10000, random_state = 21)
nn_optimizer = GridSearchCV(nn, tuned_parameters, 
                            scoring = "neg_mean_squared_error",
                            cv = 10,
                            return_train_score=False, 
                            verbose = 0)

nn_optimizer.fit(X_train_scaled, y_train_scaled)
nn_optimizer.best_params_
nn_optimizer.best_estimator_
results = pd.DataFrame(nn_optimizer.cv_results_)[['param_activation',
                      'param_hidden_layer_sizes', 'mean_test_score', 
                      'std_test_score', 'rank_test_score']].round(2)


nn_best_model = nn_optimizer.best_estimator_
nn_best_model.fit(X_train_scaled, y_train_scaled)

nn_train_pred_best = nn_best_model.predict(X_train_scaled)
metrics.r2_score(y_train_scaled, nn_train_pred_best)
metrics.mean_squared_error(y_train_scaled, nn_train_pred_best)

nn_test_pred_best = nn_best_model.predict(X_test_scaled)
metrics.r2_score(y_test_scaled, nn_test_pred_best)
metrics.mean_squared_error(nn_test_pred_best , nn_test_pred)



