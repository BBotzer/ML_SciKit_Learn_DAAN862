# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 16:12:09 2022

@author: Brandon Botzer -  btb5103



Data Set Information

This dataset is composed of a range of biomedical voice measurements from 
42 people with early-stage Parkinson's disease recruited to a six-month trial 
of a telemonitoring device for remote symptom progression monitoring. 
The recordings were automatically captured in the patient's homes. 

Columns in the table contain subject number, subject age, subject gender, 
time interval from baseline recruitment date, motor UPDRS, total UPDRS, 
and 16 biomedical voice measures.

Each row corresponds to one of 5,875 voice recording from these individuals.

The main aim of the data is to predict the motor and total UPDRS 
scores ('motor_UPDRS' and 'total_UPDRS') from the 16 voice measures. 

    1. Perform exploratory analysis on the data and 
    Remove motor_UPDRS column (10 points)
    2. Use cross-validation to build a linear regression model to predict 
    total_UPDRS (25 points)
    3. Use cross-validation to build a regression tree model to predict 
    total_UPDRS (25 points)
    4. Use cross-validation to build a neural network model to predict 
    total_UPDRS (25 points)
    5. Compare their performance with MAE (mean abosolute error)
    , which model has better performance? 
    Is there any way to improve the model? (5 points)
    6. Try to optimize the tree model or 
    neural network model (Choose one). (10 points)

You can find how to perform cross-validation in Lesson 7.

Once completed, submit a Word or Pdf file to this assignment.
"""


"""
From the web:
    Attribute Information:

subject# - Integer that uniquely identifies each subject
age - Subject age

sex - Subject gender '0' - male, '1' - female

test_time - Time since recruitment into the trial. The integer part is the 
    number of days since recruitment.
    
motor_UPDRS - Clinician's motor UPDRS score, linearly interpolated [dropped]

total_UPDRS - Clinician's total UPDRS score, linearly interpolated

Jitter(%),Jitter(Abs),Jitter:RAP,Jitter:PPQ5,Jitter:DDP - Several measures 
    of variation in fundamental frequency
    
Shimmer,Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,Shimmer:APQ11,
    Shimmer:DDA - Several measures of variation in amplitude
    
NHR,HNR - Two measures of ratio of noise to tonal components in the voice

RPDE - A nonlinear dynamical complexity measure

DFA - Signal fractal scaling exponent

PPE - A nonlinear measure of fundamental frequency variation 

"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import os

from sklearn.model_selection import train_test_split 
from sklearn import metrics

#0. Import the data

#Set the path for the CSV file
readPath = "J:\DSDegree\PennState\DAAN_862\Week 9\Homework"

#Change the directory
os.chdir(readPath)

#Read the CSV file in
df = pd.read_csv("parkinsons_updrs.data")


#1. Perform exploratory analysis on the data and 
#Remove motor_UPDRS column (10 points)

print("\n1. Perform exploratory analysis on the data and " +
      "Remove motor_UPDRS column (10 points)\n\n")

#Remove motor_UPDRS column
df = df.drop('motor_UPDRS', axis = 1)

#Look at a correlation matrix (some of these will be non-sense)
#but I am currious if anything stands out
corr = df.corr()

#plot a correlation matrix
plt.matshow(df.corr())
plt.title("Correlation Matrix")
plt.colorbar()
plt.xticks(range(21), list(df.columns))
plt.yticks(range(21), list(df.columns))

print("\nFrom the Correlation Matrix it looks like Glucose, BMI, Insulin, " +
      "HOMA, and Resistin are interesting measures")

#change the subject# to subject
df = df.rename(columns={'subject#':'subject'})

#sort the df by the subject and the temporal test_time
#such that the data is in timed sequential order
df = df.sort_values(by=['subject', 'test_time'], ascending = True)



#Get the statistic description of the dataFrame
#Once again, some of these are non-sensical but I'm curious as to what 
#I am looking at12

desc = df.describe()
print("Statistical description of dataFrame: \n")
print(desc)

#I am looking at the data set here and am confused about the 'test_time'
#values.  'test_time' is the number of days since joining the program.
#Should these test_times be sorted within each subject so that
#the rest of the data is moving temporally forward? 

#this temporal sorting shouldn't matter as I'm just trying to predict
#'totaUPDRS' from the other 16 voice measures

#Looking at the correlation numbers, I see that 'HNR' has the largest
#absolute correlation to the total_UPDRS

#Get the 16 biomedical voice measures
voiceM = df.iloc[:, 5:]

#get the HNR data
HNR = df.iloc[:, 17]
#Make the HNR series 2-D array
HNR = np.expand_dims(HNR, axis = 1)

#Get the values of the total_UPDRS values (true values from tests)
totUP = df.iloc[:, 4]


#Make a pairplot to look for any relationships to total_UPDRS
#Note: This takes time to run so I'll comment it out for now
sns.pairplot(df)

    
    
    
    
#2. Use cross-validation to build a linear regression model to predict 
#total_UPDRS (25 points)

print("\n\n2. Use cross-validation to build a linear regression model to " + 
      "predict total_UPDRS (25 points)\n")

#import cross validation scoring
from sklearn.model_selection import cross_val_score
from sklearn import linear_model

#make the object
lreg = linear_model.LinearRegression()

#Create the Test Train split
X_train, X_test, y_train, y_test = train_test_split(HNR, 
                                                    totUP, 
                                                    test_size = 0.3)


#Run Cross Fold validation 10 times with the linear reg object
#using the training data
HNR_accuracies = cross_val_score(lreg, X_train, y_train, cv = 10)

#Get the R-squared accuracy stats
HNR_accuracyStats = pd.Series(HNR_accuracies).describe()

print("Statistical information (R-square) on the linear regression fits "+
      "for HNR and total_UPDRS: \n")
print(HNR_accuracyStats)
print("\nThis fit looks like garbage...\n")


print("Looking at the pair plot there seems to be nothing correlated "+
      "via a polynomial to total_UPDRS.\n\n")

#Create a single fit of the HNR fit so I can see a plot
#It just so happens this plot is the bottom left in the paper's Fig 1.

#fit the lreg for HNR
lreg = lreg.fit(X_train, y_train)

#Get some single model statistics even though we have some R-squared from CV
lreg_train_pred = lreg.predict(X_train)
lreg_test_pred = lreg.predict(X_test)

#print a variety of metrics and store them for later
lreg_train_pred_r2 = metrics.r2_score(y_train, lreg_train_pred)
lreg_train_pred_MAE = metrics.mean_absolute_error(y_train, lreg_train_pred)
lreg_test_pred_r2 = metrics.r2_score(y_test, lreg_test_pred)
lreg_test_pred_MAE = metrics.mean_absolute_error(y_test, lreg_test_pred)

print("\nLinear Regression metrics for HNR and total_UPDRS:\n")
print("Train R-Squared score: " + str(lreg_train_pred_r2))
print("Train Mean Absol Err : " + str(lreg_train_pred_MAE))
#print a variety of metrics
print("Test R-Squared score: " + str(lreg_test_pred_r2))
print("Test Mean Absol Err : " + str(lreg_test_pred_MAE))


#plot the HNR prediction vs the truth values of total_UPDRS
plt.figure()
plt.scatter(X_train, y_train, color = 'black', label = 'True Values')
plt.plot(X_train, lreg.predict(X_train), color = 'blue', linewidth = 3,
         label = 'Linear regression model')
plt.title("Linear Regression Line with Training Data")
plt.xlabel("HNR")
plt.ylabel("total_UPDRS")
plt.legend()






#I wonder if setting up a lasso or a ridge model would help?


#Let's try to make a CV fit where I optimize hyperparameters
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LassoCV

#build empty lists to append into later
lasscv_train_pred_r2 = []
lasscv_train_pred_MAE = []
lasscv_test_pred_r2 = []
lasscv_test_pred_MAE = []

#Run the Lasso CV test 1000 times to average error results together
for i in range(1000):

    #Create the Test Train split using all 16 variables
    #We'll have a possible dimensionality issue here with such a large space
    X_train, X_test, y_train, y_test = train_test_split(voiceM, 
                                                    totUP, 
                                                    test_size = 0.3)

    #make the  model object
    lasso_cv = LassoCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10],
                       max_iter=10000)
    #Train the model
    lasso_cv.fit(X_train, y_train)
    
    
    
    #print("\nLasso score on training data:")
    #print(lasso_cv.score(X_train, y_train))
    #print("\nLasso score on test data:")
    #print(lasso_cv.score(X_test, y_test))
    #print("Still not a good fit.  As expected.")
    
    
    #Get the r2 and MAE for the lasso_cv
    lasscv_train_pred = lasso_cv.predict(X_train)
    lasscv_test_pred = lasso_cv.predict(X_test)
    
    #append these vaules into the list to take an average later
    lasscv_train_pred_r2.append(metrics.r2_score(y_train, lasscv_train_pred))
    lasscv_train_pred_MAE.append(metrics.mean_absolute_error(y_train, 
                                                        lasscv_train_pred))
    lasscv_test_pred_r2.append(metrics.r2_score(y_test, lasscv_test_pred))
    lasscv_test_pred_MAE.append(metrics.mean_absolute_error(y_test, 
                                                       lasscv_test_pred))



print("\nThe Lasso CV results:")
print("\nLasso CV Training MAE:")
print(pd.Series(lasscv_train_pred_MAE).describe())

print("\nLasso CV Test MAE:")
print(pd.Series(lasscv_test_pred_MAE).describe())

print("\nLasso CV Train Average r2")
print(pd.Series(lasscv_train_pred_r2).mean())

print("\nLasso CV Test Average r2")
print(pd.Series(lasscv_test_pred_r2).mean())


#get MAE averages for plotting later
lasscv_train_pred_MAE_avg = pd.Series(lasscv_train_pred_MAE).mean()
lasscv_test_pred_MAE_avg = pd.Series(lasscv_test_pred_MAE).mean()
lasscv_train_pred_r2_avg = pd.Series(lasscv_train_pred_r2).mean()
lasscv_test_pred_r2_avg = pd.Series(lasscv_test_pred_r2).mean()



print("\\n\n\nSTART\nI am not currently using this section for analysis as " +
      "I have decided to use the Lasso Lin Reg.")

#Try a run where I pass all of the 16 voiceM values
#We'll have a possible dimensionality issue here with such a large space
#Let's see what happens anyway
lreg16 = linear_model.LinearRegression()

#Create the Test Train split
X_train, X_test, y_train, y_test = train_test_split(voiceM, 
                                                    totUP, 
                                                    test_size = 0.3)

#Run Cross Fold validation 10 times with the linear reg object
#using the training data
#Does this need a 'scoring' input? metrics.r2_score?
#metrics.mean_absolute_error(y_true, y_pred)  what are the true and pred?
voiceM_accuracies = cross_val_score(lreg16, X_train, y_train, cv = 10)


#Get the R-squared accuracy stats
voiceM_accuracyStats = pd.Series(voiceM_accuracies).describe()

print("\n\nStatistical information (R-square) on the linear regression fits" +
      " with all 16 voice Measures: \n")
print(voiceM_accuracyStats)
print("\nEven when I give full reign to the 16 voice measures "+
      "I still have a very poor fit.  This is not surprising based "+
      "on the pair plot.\n\n")

#What do the coefficients of this look like?
#Create a single fit to build some coefs
lreg16.fit(X_train, y_train)

#Output the coefs for the 16 voiceMeasures
print("Lin Reg with 16 voice measures coefs:")
print(lreg16.coef_)

print("END and resume results for analysis.\n\n\n")


"""
cross_validate is not working right now.  
The cross_validate function is throwing errors!

#Try this with Corss Validate Function
from sklearn.model_selection import cross_validate

scoring = ['accuracy', 'precision_macro', 'recall_macro']

scores = cross_validate(lreg, X_train, y_train, scoring = 'accuracy', 
                        cv = 10, return_train_score = True)

pd.DataFrame(scores).mean()
"""





#3. Use cross-validation to build a regression tree model to predict 
#total_UPDRS (25 points)


#We'll use all 16 voice measures to build this regression tree


#Create the Test Train split again with the 16 voiceM
X_train, X_test, y_train, y_test = train_test_split(voiceM, 
                                                    totUP, 
                                                    test_size = 0.3)

#Import the Regression Tree
from sklearn import tree

#Trying to use cross_validate but having issues with it
from sklearn.model_selection import cross_validate

#Build the Object
#When min_samples_leaf = 5 I find a stupidly large tree
#I've increased min_samples_leaf to 100 to create a 'simpler' tree
#Doing so also increased the model fit
tree_model = tree.DecisionTreeRegressor(min_samples_leaf = 100)


#Run the Cross Validation on the Reg tree
tree_train_accuracies = cross_val_score(tree_model, X_train, 
                                        y_train, cv = 10)

tree_train_accuracyStats = pd.Series(tree_train_accuracies).describe()

print("\nStatistical information on the Decision Tree Regressor fits "+
      "for 16 voice measures and total_UPDRS: \n")
print(tree_train_accuracyStats)
print("\nThis fit looks like garbage too...\n")

#Let us see if we can make a graph and if a fit exists
#The fit does not yet exist.... ugh... I probably need to use GridSearchCV
#to get hyperparams and then use them to fit???

#Lets just do a single fit for now...

tree_model.fit(X_train, y_train)

from graphviz import Source

#Write out the decision tree to the active directory (Remove Source() to work)
dot_data = tree.export_graphviz(tree_model, out_file = None,
                            feature_names = X_train.columns)
#Display the tree 
Source(dot_data)
#Set filename
fname = "ParkingsonVoiceM_DTree"
#Save the tree out to a pdf file
Source(dot_data).render(fname)





#Let's do a full CV fit with hyperparameters via GridSearchCV


#Make the decision tree reg object
dec_tree_model = tree.DecisionTreeRegressor()

#Set up hyperparameters to make combinations of (leave scoring as default)
params = [{'min_samples_leaf': [10,50,100,250],
           'min_samples_split':[10,50,100,250],
           'max_depth': [5,10,20]}]

#Set up the model object to use for fitting
tree_cv_model = GridSearchCV(dec_tree_model,
                             param_grid = params,
                             cv = 10)

#Fit all of the combinations using CV for each run
tree_cv_model.fit(X_train, y_train)

#What are the best hyperparameters
print("What are the best hyperparameters for the Dec Tree Reg:\n")
print(tree_cv_model.best_params_)

#What is the score with the best hyperparameters
print("\nThe score on the training data of the Dec Tree " + 
      "Reg with the best hyperparameters:")
print(tree_cv_model.score(X_train, y_train))

print("\nThe score on the test data of the Dec Tree " + 
      "Reg with the best hyperparameters:")
print(tree_cv_model.score(X_test, y_test))

print("\nWe may be overfitting the data.  Let's make the tree as a pdf.")

#Make a pdf of this optimized tree
#Write out the decision tree to the active directory
#Need to use the model.best_estimator_ to pass the DecTreeReg Object
dot_data = tree.export_graphviz(tree_cv_model.best_estimator_, 
                                out_file = None, 
                                feature_names = X_train.columns)
#Display the tree 
Source(dot_data)
#Set filename
fname = "ParkingsonVoiceM_DTree_HyperOptimized"
#Save the tree out to a pdf file
Source(dot_data).render(fname)


#While the above gives us a good idea as to what the Tree is doing
#with CV, let's run it many times to get better MAE and r2 values.


#build an empty list to append into later
tcvmod_train_pred_r2 = []
tcvmod_train_pred_MAE = []
tcvmod_test_pred_r2 = []
tcvmod_test_pred_MAE = []


#only did 100 here as it takes 12 seconds a run
for i in range(100):
    
    #Create the Test Train split again with the 16 voiceM
    X_train, X_test, y_train, y_test = train_test_split(voiceM, 
                                                        totUP, 
                                                        test_size = 0.3)    
    
    #Make the decision tree reg object
    dec_tree_model = tree.DecisionTreeRegressor()

    #Set up hyperparameters to make combinations of (leave scoring as default)
    params = [{'min_samples_leaf': [10,50,100,250],
               'min_samples_split':[10,50,100,250],
               'max_depth': [5,10,20]}]

    #Set up the model object to use for fitting
    tree_cv_model = GridSearchCV(dec_tree_model,
                                 param_grid = params,
                                 cv = 10)

    #Fit all of the combinations using CV for each run
    tree_cv_model.fit(X_train, y_train)
    
    #Get the decision tree reg prediction values for train and test
    tree_cv_model_train_pred = tree_cv_model.predict(X_train)
    tree_cv_model_test_pred = tree_cv_model.predict(X_test)

    
    #append the r2 and MAE scores
    tcvmod_train_pred_r2.append(metrics.r2_score(y_train, 
                                                 tree_cv_model_train_pred))
    
    tcvmod_train_pred_MAE.append(metrics.mean_absolute_error(y_train, 
                                                        tree_cv_model_train_pred))
    
    tcvmod_test_pred_r2.append(metrics.r2_score(y_test, 
                                                tree_cv_model_test_pred))
    
    tcvmod_test_pred_MAE.append(metrics.mean_absolute_error(y_test, 
                                                       tree_cv_model_test_pred))


#print out the 100 run Tree Reg CV metrics
print("\nDecission Tree Regressor metrics:\n")
print("\Dec Tree CV Training MAE:")
print(pd.Series(tcvmod_train_pred_MAE).describe())

print("\nDec Tree CV Test MAE:")
print(pd.Series(tcvmod_test_pred_MAE).describe())

print("\nDec Tree CV Train Average r2")
print(pd.Series(tcvmod_train_pred_r2).mean())

print("\nDec Tree CV Test Average r2")
print(pd.Series(tcvmod_test_pred_r2).mean())


#get MAE averages for plotting later
tcvmod_train_pred_MAE_avg = pd.Series(tcvmod_train_pred_MAE).mean()
tcvmod_test_pred_MAE_avg = pd.Series(tcvmod_test_pred_MAE).mean()
tcvmod_train_pred_r2_avg = pd.Series(tcvmod_train_pred_r2).mean()
tcvmod_test_pred_r2_avg = pd.Series(tcvmod_test_pred_r2).mean()








#4. Use cross-validation to build a neural network model to predict 
#total_UPDRS (25 points)

print("\n\n4. Use cross-validation to build a neural network model to " + 
      "predict total_UPDRS (25 points)")

#Use the NN optimizer during this

#PREPROCESSING THE DATA FOR THE NEURAL NET
print("\nWe must preprocess the data for the neural net.\n")

#There are no indicator variables to make dummies in voiceM
print("There are no indicator variables to make dummies in voiceM")
print(voiceM.head)

#We do need to rescale the variables to range [0, 1]
from sklearn.preprocessing import MinMaxScaler

#I need to rejoin the voiceM (X) and the totUP data (y) to scale it all
#This is different from Week 8 as we are not classifying (0/1 values)
#This week we are trying to find a value (float) and must scale as a result

#get the voiceM with totUP data
#columns to drop
drops = ['subject', 'age', 'sex', 'test_time']
#build a new dataFrame by droping the unneeded columns from the orig df
nn_data = df.drop(drops, axis = 1)

#split the train and test data
train, test = train_test_split(nn_data, test_size = 0.3)

#make the scaler object
scaler = MinMaxScaler()

#fit the train data in the scaler
scaler.fit(train)

#scale the data to [0,1]
train = scaler.transform(train)
test = scaler.transform(test)

#Split the totUP and 16 voiceM measures again for train and test
X_train_scaled = train[:, 1:]
X_test_scaled = test[:, 1:]
y_train_scaled = train[:, 0]
y_test_scaled = test[:, 0]



#Generate the Neural Network Model
from sklearn import neural_network

#Set up neural netword hyperparameters to search over
#I am a bit at a loss of where to go for this so there is a 
#large search space in the parameters 
#I am going to scale down the searach space but increase the itterations
#nn_params = [{'hidden_layer_sizes':[5,10,50,100, 500],
#             'activation':['identity', 'logistic', 'relu'],
#              'solver':['sgd', 'lbfgs', 'adam'],
#              'alpha':[0.0001, 0.001, 0.01, 0.1, 1, 10]}]

#Best params from the above set were:
    #Best NN parameters:
    #{'activation': 'relu', 'alpha': 0.1, 
     #'hidden_layer_sizes': 500, 'solver': 'lbfgs'}

nn_params = [{'hidden_layer_sizes':[50, 60, 80,100, 200],
              'activation':['relu'],
              'solver':['sgd', 'lbfgs', 'adam'],
              'alpha':[0.01, 0.1, 1, 10]}]

#set the NN model.  max_iter was set past 200 as it did not converge
#1st attempt: max_iter = 200
#2nd attempt: max_iter = 100000  (reduced param space r2 value = 0.516)
#3rd attempt: max_iter = 10000000 (reduced param space again r2 = .507)


#Curious about the runtime
import time


nn_model = neural_network.MLPRegressor(max_iter=100000)

#run the search over the hyperparameters using CV
nn_optimizer = GridSearchCV(nn_model, nn_params,
                            scoring = 'neg_mean_absolute_error',
                            cv = 10, n_jobs = -1, verbose = True)
#start the clock
ti = time.perf_counter()
#Fit the NN
nn_optimizer.fit(X_train_scaled, y_train_scaled)

#Check the best parameters
print("\nBest NN parameters:")
print(nn_optimizer.best_params_)

#Save the results of the CV'd optimizer
nn_results = pd.DataFrame(nn_optimizer.cv_results_)[['param_hidden_layer_sizes',
                                                     'param_activation',
                                                     'param_solver',
                                                     'param_alpha',
                                                     'mean_test_score',
                                                     'std_test_score',
                                                     'rank_test_score']]


#Get the best NN model
nn_best_model = nn_optimizer.best_estimator_

#Fit the best NN model
nn_best_model.fit(X_train_scaled, y_train_scaled)

#end clock as final fitting has completed
tf = time.perf_counter()

#Get the predictions for the training and test sets
nn_best_model_train_pred = nn_best_model.predict(X_train_scaled)
nn_best_model_test_pred = nn_best_model.predict(X_test_scaled)

#Print some metrics and store values for later

nn_best_model_train_pred_r2 = metrics.r2_score(y_train_scaled, 
                                               nn_best_model_train_pred)

nn_best_model_train_pred_MAE = metrics.mean_absolute_error(y_train_scaled, 
                                                    nn_best_model_train_pred)

nn_best_model_test_pred_r2 = metrics.r2_score(y_test_scaled, 
                                              nn_best_model_test_pred)

nn_best_model_test_pred_MAE = metrics.mean_absolute_error(y_test_scaled, 
                                                   nn_best_model_test_pred)

print("\nNeural Network metrics:\n")
print("The best NN model used: \n" + str(nn_best_model))
print("\nTrain R-Squared score: " + 
      str(nn_best_model_train_pred_r2))
print("Train Mean Absol Err : " + 
      str(nn_best_model_train_pred_MAE))
#print a variety of metrics
print("Test R-Squared score: " + 
      str(nn_best_model_test_pred_r2))
print("Test Mean Absol Err : " + 
      str(nn_best_model_test_pred_MAE))


print("\n\nTime to run the max_iter=10000000 NN was: " +
      str(tf - ti))

#Runtime for me was 23 mins + a few for the tree and lreg









#5.  Compare their performance with MAE (mean abosolute error), 
#which model has better performance? 
#Is there any way to improve the model? (5 points)

#Let's make a plot of all of the MAE and r2 values in two bar charts

#Set up the MAE values in a Data Frame
mae_bar_test_df = pd.DataFrame(data = [[lasscv_test_pred_MAE_avg, 
                                   tcvmod_test_pred_MAE_avg, 
                                   nn_best_model_test_pred_MAE]], 
                          columns = ['LR_Lass_MAE', 'TREE_MAE', 'NN_MAE'])
#Grab the MAE values
mae_test_values = [mae_bar_test_df.LR_Lass_MAE[0], 
              mae_bar_test_df.TREE_MAE[0], 
              mae_bar_test_df.NN_MAE[0]]

#plot bar chart
plt.figure()
plt.bar(mae_bar_test_df.columns, mae_test_values)
plt.title("Test Mean Absoulte Error")


#Set up the MAE values in a Data Frame
mae_bar_train_df = pd.DataFrame(data = [[lasscv_train_pred_MAE_avg, 
                                   tcvmod_train_pred_MAE_avg, 
                                   nn_best_model_train_pred_MAE]], 
                          columns = ['LR_Lass_MAE', 'TREE_MAE', 'NN_MAE'])
#Grab the MAE values
mae_train_values = [mae_bar_train_df.LR_Lass_MAE[0], 
              mae_bar_train_df.TREE_MAE[0], 
              mae_bar_train_df.NN_MAE[0]]

#plot bar chart
plt.figure()
plt.bar(mae_bar_train_df.columns, mae_train_values)
plt.title("Training Mean Absoulte Error")


#Set up the r2 values in a DF
r2_bar_test_df = pd.DataFrame(data = [[lasscv_test_pred_r2_avg,
                                  tcvmod_test_pred_r2_avg,
                                  nn_best_model_test_pred_r2]],
                         columns = ['LR_Lass_r2', 'TREE_r2', 'NN_r2'])

#Grab the r2 values
r2_test_values = [r2_bar_test_df.LR_Lass_r2[0],
             r2_bar_test_df.TREE_r2[0],
             r2_bar_test_df.NN_r2[0]]

#plot bar chart
#plot bar chart
plt.figure()
plt.bar(r2_bar_test_df.columns, r2_test_values)
plt.title("Test R-Squared Values")



#Set up the r2 values in a DF
r2_bar_train_df = pd.DataFrame(data = [[lasscv_train_pred_r2_avg,
                                  tcvmod_train_pred_r2_avg,
                                  nn_best_model_train_pred_r2]],
                         columns = ['LR_Lass_r2', 'TREE_r2', 'NN_r2'])

#Grab the r2 values
r2_train_values = [r2_bar_train_df.LR_Lass_r2[0],
             r2_bar_train_df.TREE_r2[0],
             r2_bar_train_df.NN_r2[0]]

#plot bar chart
#plot bar chart
plt.figure()
plt.bar(r2_bar_test_df.columns, r2_train_values)
plt.title("Training R-Squared Values")




#Try a grouped bar chart

labels = ['LR Lass', 'TREE', 'NN']

#MAE Grouped bar chart

#number the label locations
x = np.arange(len(labels))

#width of the bars
width = 0.36

#Plot the Grouped MAE
fig, ax = plt.subplots()
rec1 = ax.bar(x - width/2, mae_train_values, width, label='Training MAE')
rec2 = ax.bar(x + width/2, mae_test_values, width, label='Test MAE')
ax.set_ylabel('Mean Absolute Error')
ax.set_title('Mean Absoulute Errors for Train and Test Data')
ax.set_xticks(x, labels)
ax.legend()
fig.tight_layout()

#Plot the Grouped r2 bar chart
fig, ax = plt.subplots()
rec1 = ax.bar(x - width/2, r2_train_values, width, label='Training R-Squared')
rec2 = ax.bar(x + width/2, r2_test_values, width, label='Test R-Squared')
ax.set_ylabel('R-Squared')
ax.set_title('R-Squared Value for Train and Test Data')
ax.set_xticks(x, labels)
ax.legend()
fig.tight_layout()





#The NN was by far the best as it had the best MAE and r2.  It may have overfit
#which is something to watch out for.  Though the Decision Tree may have 
#overfit as well.  The depth could more than likely be pruned on the DT.

#The LassoLR did not have strong r2 fits but this could be due to fitting
#over multiple subjects.  The MAE was a better indicator as a result.


#I do not know if this would help but it is a thought I've had...
#The Parkinson's data we have is set up to go from 0 through 180 days.
#I am currious if we averaged days together if we would remove some of the
#noise that may be occuring.  At the same time this would reduce
#the number of data points and may hurt more than it helps.
#The authors suggest that many of the variables are confounding
#and it may help to use an Added Variable Plot to account for this.








#6. Try to optimiaze the tree model or NN (10 pts)

#I have searched for better hyperparams on the dec tree regressor
#I have also optimized the NN in part 5




#Data set provided by
#A Tsanas, MA Little, PE McSharry, LO Ramig (2009)
#'Accurate telemonitoring of Parkinson.s disease progression by non-invasive 
#speech tests',
#IEEE Transactions on Biomedical Engineering (to appear). 




