#%%
"""
Created on Wed Nov 16 18:04:10 2022

author:Brandon Botzer btb5103


Problem: 

A website sent advertisements by emails to users who interested in their 
product. Your task is to find a good model to predict if an advertisement 
will be clicked with given datasets. 

user_features.csv Download user_features.csv - features describing our users

product_feature.csv Download product_feature.csv - features describing 
products shown in the advertisements. 

click_history.csv Download click_history.csv. - contains which products users 
had previously seen and whether that user ordered products in 
this website before.

**This is not a real dataset


    Question 1: Data Understanding

Explore the basic information of the datasets. (5points)

    Question 2: Data Cleaning and Preprocessing

Clean and preprocess the datasets (such as missing values, outliers, 
                                   dummy, merging etc.). (15points)

    Question 3: Model Generation and Evaluation

Please split the data into train and test sets with a ratio of 0.7:0.3. 
Build and optimize classification models you learned in this course. (30points)

    Question 4: Which model has the best performance? What have you learned 
    from the models you built? Explain why you tested the models you did, 
    and whether the results matched your expectation. Be sure to include 
    relevant information about how the model works, to explain why you are 
    choosing to use this set of models 
    (rather than randomly selecting models to test without knowing 
     why or how they work) (25 points)

"""
#%%
#imports
import os  #used to load data
import numpy as np  #numpy
import pandas as pd  #pandas
import matplotlib.pyplot as plt  #plotting
import seaborn as sns  #pairplots

from sklearn import metrics  #metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

import time


#Load the data

#set the read path
readpath = "J:\DSDegree\PennState\DAAN_862\Week 14_Final\Final"

#change directory to readpath
os.chdir(readpath)

#read the CSV files
click_history_df = pd.read_csv("click_history.csv")
product_feats_df = pd.read_csv("product_features.csv")
user_feats_df = pd.read_csv("user_features.csv")


print("\nAs I go through cleaning the datasets and creating the models, " + 
      "I will generally 'print' to the screen what I am doing." + 
      "\nI will also display as comments '#' in the code what is occuring " +
      "during intermediate steps as well as provide some of my thoughts on " +
      "Why I am performing these steps.  You'll see in most cases, " +
      "the comments and insight provided are longer than the code itself.")


#Question 1: Explore the Datasets
print("\n\nExplore the datasets:")

print("\nClick History DF:")
print(click_history_df.info)

print("\nProduct Features DF:")
print(product_feats_df.info)

print("\nUser Features DF:")
print(user_feats_df.info)
print(user_feats_df.columns)


#Cleaning items (at minimum)
#missing values, outliers, dummy, merging, check duplicates

#Check for duplicates
def num_duplicates(df):
    
    return len(df) - len(df.drop_duplicates())


#check for duplicates to investigate for potential drop candidates
print("\nChecking for duplicates...")
print("There are " + str(num_duplicates(click_history_df)) + 
      " duplicates in Click_History.")

print("There are " + str(num_duplicates(user_feats_df)) + 
      " duplicates in User Features.")  

print("There are " + str(num_duplicates(product_feats_df)) + 
      " duplicates in Product Features.")  

      
#check for missing data
def missing_vals(df):
    
    #get the current counts of non-Missing data
    counts = list(df.count())
    
    #build a list of the length of the full DF
    full = []
    for i in range(0, len(counts)):
        full.append(len(df))
    
    #use numpy to subtract lists to find number of missing values of each col
    return np.subtract(full, counts)


print("\nFor the columns in Click History, there are " + 
      str(missing_vals(click_history_df)) + " missing values.")

print("For the columns in Product Features, there are " + 
      str(missing_vals(product_feats_df)) + " missing values.")

print("For the columns in User Features, there are " + 
      str(missing_vals(user_feats_df)) + " missing values.")


#we need to do something about the missing values in user_features
#specifically in the number_of_clicks_before column
#There are two options, we can drop this data, or we can artificially create it

#I elect to drop the data as I konw absolutly nothing about the column
#and what it represents.

#drop the nan values in User_Features
user_feats_df = user_feats_df.dropna(how = 'any')

#reset the indexing so that dropped value indexes are removed
#This will help with assigning dummy preferences later
user_feats_df = user_feats_df.reset_index(drop = True)


#We should now check if there are outliers in the data set
#I'll run a quick pass of max and min to see if anything jumps out
print("\n\nI'll now check on outliers:")

print("\n\nThe max values of User Features are: \n" + str(user_feats_df.max()))
print("\nThe minvalues of User Features are: \n" + str(user_feats_df.min()))
print("\n\nUser Features looks good.")


print("\n\nThe max values of Click History are: \n" + str(click_history_df.max()))
print("\nThe minvalues of Click Hisotry are: \n" + str(click_history_df.min()))
print("\n\nClick Hisotry looks good.")

print("\n\nThe max values of Product Features are: \n" + str(product_feats_df.max()))
print("\nThe minvalues of Product Features are: \n" + str(product_feats_df.min()))
print("\n\nProduct Features has a problem with a negative average " + 
      "review score as well as a rather large number of reviews.")

print("\nI will fix the issues with the Product Feature outlier values.")



#find unique categories and preferences
prefs_unique = list(product_feats_df.category.unique())


#expand the preference (interest) values
prefs_list = list(user_feats_df.personal_interests.explode())

#empty list for preferences later
p = []

     
#find the unique items from the preferences
#Set only pulls unique items and then I convert back to a list
#prefs_unique = list(set(p))

user_prefs = []

#build the dummy columns (one-hot encoding) and set them to zero
#will flip these values in the loop if they are interests

user_feats_df[prefs_unique] = 0

for j in range(0, len(prefs_list)):
    
    #break up the string into list items
    #strip off the brackets
    pref_string = prefs_list[j].strip("[]")
    #sperate by ','
    ps = pref_string.split(sep = ',')
    
    for k in range(0, len(ps)):
        #stip out the ' and spaces and store in a temp string
        tempstr = ps[k].strip("' ")
        #create a list of cleaned words
        user_prefs.append(tempstr)
        #shift those words back into the prefs_list
        prefs_list[j] = user_prefs
    #clear the storage for the next user
    user_prefs=[]
    
    
    #prefs_list is now a list of the word category/preference for each user
    
    #check each word in current prefs_list to see if it matches anything in prefs_uniq
    for a_pref in prefs_list[j]:
        #itterate through possible preferences
       for u_pref in prefs_unique:
           #Check these words against the list of possible prefereances
           if a_pref == u_pref:
                #assign the dummy value a 1 (else it will stay 0)
                user_feats_df.loc[j, a_pref] = 1
                
#drop the personal interests column as we no longer need it
user_feats_df.drop(labels="personal_interests", axis = 1, inplace=True)

              
#Get dummy variables for the categories
cat_dummies = pd.get_dummies(product_feats_df.category, prefix="cat")

#join the product features DF and the category dummy variables to eachother
#joining on the product_feats_df index but either will work as
#the dummies are drawn from that DF
product_feats_df = product_feats_df.join(cat_dummies, how = 'left')
       
#can now drop the category column
product_feats_df.drop(labels="category", axis = 1, inplace=True)


#Create the dummy variables
#Make on sale a dummy
product_feats_df["on_sale"] = product_feats_df["on_sale"].astype(int)


#make ordered_before a 1 / 0 value (one-hot encode it)
user_feats_df["ordered_before"] = user_feats_df["ordered_before"].astype(int)

#make number of clicks a dummy?  Or just convert 6+ to 6 on grounds that 
    #any more than six might as well just be 6
#Turn 6+ into 6 on ghte grounds that any more than 6 might as well be 6

user_feats_df.loc[user_feats_df["number_of_clicks_before"] == "6+","number_of_clicks_before"] = 6

#convert the rest of user_feats_df[number of clicks] to integers
user_feats_df["number_of_clicks_before"] = user_feats_df["number_of_clicks_before"].astype(int)


#drop the average user reviews that have values less than one
#Values should range from 1 - 5 since those are the typical selection 
#criteria when reviewing.  It is not possible for a user on most
#modern platforms to rate anything less than 1.  Thus, the average can
#never be less than one.
product_feats_df = product_feats_df[product_feats_df.avg_review_score >=1]


#look at the number of reviews on a product
#quick visual inspection shows a set that are wild positive outliers

#Set the 'outlier' limit as 10 times the mode
#This takes care of the large shift that the outliers caused
#while still allowing an EXTREAMLY popular product (~7000 reviews) to exist
limit = int(product_feats_df["number_of_reviews"].mode() * 10)

product_feats_df = product_feats_df[product_feats_df.number_of_reviews <= limit]



#begin the combined Data Frame
full_df = click_history_df

#bring in all of the product data to each user click result
full_df= full_df.merge(product_feats_df, 
              how = 'outer', 
              left_on="product_id", 
              right_on="product_id")

#bring in the user preferences for each user
full_df = full_df.merge(user_feats_df, 
                        how='outer',
                        left_on="user_id",
                        right_on="user_id")

#there are users who have preferences who we do not have click data on
#likewise, there are users sho have click data but no preferences
#we must drop these users who do not match up across information regions
full_df = full_df.dropna(how = 'any')

#full_df is now a data frame with clean fully functional information
print("\nProduct features issues have been resolved.  The data set has " +
      "been fully cleaned and constructed with proper dummy variables." +
      "  This data is located in the 'full_df' Data Frame.")
print("\nInformation on 'full_df':\n")
print(full_df.info())




#%%
#Section to create plots for initial data analysis
print("\nCreating a Correlation Matrix...")
#plot a correlation matrix
plt.matshow(full_df.corr())
plt.title("Corr Matrix on Click Items")
plt.colorbar()
plt.xticks(range(30), list(full_df.columns))
plt.yticks(range(30), list(full_df.columns))
print("Correlation Matrix Created.  The categories seem to correlate with " +
      "advertising clicks.")

print("\nCreating a pairplot to see how items are related...")
sns.pairplot(full_df)
print("Pairplot created.")




#%%
#I might create a histogram where the values are the avg review score * click
#this would suppress the non-click values and see if there is a threshold
#where if a review is high enough people would click it
good_thresh = full_df.clicked * full_df.avg_review_score
plt.figure()
plt.hist(good_thresh)
plt.title("Histogram of Average Review Scores of Clicked ads\n with non-clicked" +
          " in zero column")
#looking at the distribution between 1 and 5, there is no real trend.
#The non-clicked values take up half of the data set as well
plt.figure()
plt.hist((full_df.clicked.astype(int)))
plt.title("Split of number of non-clicked and clicked ads")

bad_thresh = full_df["avg_review_score"].loc[full_df.clicked == False]
plt.figure()
plt.hist(bad_thresh)
plt.title("Distribution of Avg Rev Scores of non-clicked ads")

#there seems to be no identifier here for the non-clicked items either
#Even items with a high avg_review_score are not clicked at a high rate

#This means that there could to be some relationship between multiple vars
#to predict the click / non-click.  This seems to be difficult given that 
#the pair plot had next to no correlations between the parameters themselves

#Given that there seems to be no direct relationship
#I will attempt to model the data with a Neural Network Classifier in an 
#effort to tease out the codependancies
#There may also be some benefit to a Random Forest Classifier
#as those are known to sift trough multipole dependancies
    #Though I am not 100% sure given how many 1/0 columns we have...
    
#Models like logistic regression would perform poorly here but I will run 
#one to show that it is a poor fit compared to the NN and RF


#Drop the users and the specific products
#We want the model to be independant of user and of the product as well
#it needs to work for new users and new products down the line
drops = ["user_id", "product_id"]
#data with user and product ids dropped
data = full_df.drop(drops, axis = 1)

#first to split the relevant variables into useable datasets
#Results of Click / noclick
y = data.clicked.astype(int)

#clean data to build the models 
#I'll perform train-test splits in each section independantly
    #given the large size of the data set, this should minimally impact results
clean_data = data.iloc[:, 1:]



#%%
#BEGIN THE NEURAL NETWORK MODEL

#split the data into train / test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(clean_data, y, 
                                                    test_size = 0.3)

print("Train/Test Split is now complete for the Neural Network.")

print("\nWe must now rescale the data for the Neural Network.")
#We do need to rescale the variables to range [0, 1]
from sklearn.preprocessing import MinMaxScaler
#make the scaler object
scaler = MinMaxScaler()

#Fit Scaler on the Training data and transform it
X_train_scaled = scaler.fit_transform(X_train, y=None)
#then apply same scaler to the test data (test data does not influence scaler)
X_test_scaled = scaler.transform(X_test)
print("Scaling for the Neural Network is complete.")



print("\nWe will now create, fit, and predict using the Neural Network." + 
      "\nThis takes time as we will run Cross Validation over a grid of " +
      "parameters.  \nResults will be printed shortly...")
#import the Neural Network
from sklearn.neural_network import MLPClassifier

#For CV tests
nn_params = [{'hidden_layer_sizes':[(10,10,10), (7,7,7)],
              'activation':['relu'],
              'solver':['adam'],
              'alpha':[0.5, 0.75]}]

#Build the NN object
nn_model = MLPClassifier(max_iter=100000)

#build the optimizer
nn_optimize = GridSearchCV(nn_model, nn_params, cv = 10, n_jobs=-1,
                           verbose=True)


print("\nSetting up the Neural Network.... running....")

tick = time.perf_counter()
#fit the NN optimizer using CV
nn_optimize.fit(X_train_scaled, y_train)

#fit the data (initial test for single run)
#nn_model.fit(X_train_scaled, y_train)

#predict the data (initial test for single run)
#nn_pred = nn_model.predict(X_test_scaled)

#predit the data (FOR CV WINNER)
nn_pred = nn_optimize.predict(X_test_scaled)

tock = time.perf_counter()

#save the best NN results:
print("\nBest NN parameters:")
print(nn_optimize.best_params_)

#Save the results of the CV'd optimizer
nn_results = pd.DataFrame(nn_optimize.cv_results_)[['param_hidden_layer_sizes',
                                                     'param_activation',
                                                     'param_solver',
                                                     'param_alpha',
                                                     'mean_test_score',
                                                     'std_test_score',
                                                     'rank_test_score']]


print("Neural Network Complete.")
print("The time requrired to run was: " + str(tock-tick))

print("\nThe accuracy metrics for the Neural Network are:")
print("  " + str(metrics.accuracy_score(y_test, nn_pred)))
print("\nMetrics classificaiton report:\n")
print(metrics.classification_report(y_test, nn_pred))


#%%
#DETERMINE A GOOD VALUE FOR N_ESTIMATOR FOR THE RANDOM FOREST
#3 hour runtime on my PC

from sklearn.ensemble import RandomForestClassifier

#data frame for storage for later
RFdf = pd.DataFrame()

#Use the train test from before as it is outside the loop
runs = range(0,6)

n_estim = range(2,201,2) 

tick = time.perf_counter()

#Loop the number of runs
for j in runs:

    #reset accuracy to an empty list
    accuracy = []
    
    #Reset the train test split
    X_train, X_test, y_train, y_test = train_test_split(clean_data,
                                                        y,
                                                        test_size=0.3)
     
    #loop over various values of n_estimators
    for i in n_estim:
        
        #Build the Random Forest Classifier 
        #set a random state so that you can pin down n_estimators
        RF = RandomForestClassifier(n_estimators = i, random_state = 226)
        
        #Fit and get scores
        RF_scores = cross_val_score(RF, X_train, y_train, cv = 10)
        
        #append score values into accuracy list
        accuracy.append(RF_scores.mean())
        
    
    #put int the new data frame column
    runname = "run_" + str(j)
    RFdf.insert(j, runname, accuracy)
    #keep track of where you are in the process
    print(runname)

tock = time.perf_counter()

#create some formating for the plot
s = []
for i in range(2,201,2):
    s.append(i)
#set the index to count by 2's for the plot
RFdf = RFdf.set_index([s])

#plot the Ensemble Accuracy
plt.figure()
RFdf.plot()
plt.title("Random Forsest Accuracy")
plt.ylabel('Accuracy',)
plt.xlabel("N_Estimators in Model")
plt.legend(loc = 4, fontsize = 6)

#get the average best value of n
#rather, get the n which hits the maximum at a moment from all of the runs
#and take the mean of those values

#Do not readjust the index since you've don this above
#This value is correct
avg_best_n = (int(RFdf.idxmax().mean()))


print("\nThe time requrired to run was: " + str(tock-tick))
print("\nThe average best value for the Random Forest's n_estimators is: " + 
      str(avg_best_n))

#For future runs, the average best n_estimator is roughly 174
#Based on the graph, I can get away with 100-115 with minimal losses
#This will save on computational time
#I will set this feature down in the next section as prior_bet_n
#It must be a list [] to sit within the rf_params grid


#%%
#BEGIN THE RANDOM FOREST CLASSIFIER WITH N_ESTIMATORS FROM BEFORE

#split the data into train / test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(clean_data, y, 
                                                    test_size = 0.3)

from sklearn import ensemble

#the n_estimator is the greatest predictor here



#set up a large grid of possible n_estimator values
#from a prior graph, I sam setting the n_estimators value to 115
#This has similar accuracy to the avg_best_n value of 174 found in a 
    #previous run

#set as a list to use in the rf_params grid
prior_best_n = [115]
#Random Forest Parameters
#This grid has been parsed down from a series of prior runs
rf_params = [{'n_estimators':prior_best_n,
              'criterion':['gini'],
              'max_depth':[10,15,20,25,50], 
              'min_samples_split':[10,20,50,100],
              'min_samples_leaf':[10,50,100,200,1000]}]

#Random Forest Object
rf = ensemble.RandomForestClassifier()

print("I will search for the best parameters for the random forest.")
print("This may take some time... generating forests...")

tick = time.perf_counter()

#Run CV on the random forest
rf_optimize = GridSearchCV(estimator=rf, param_grid=rf_params, n_jobs=-1)

rf_optimize.fit(X_train, y_train)

#model evaluation
rf_pred = rf_optimize.predict(X_test)

rf_report = metrics.classification_report(y_test, rf_pred)
tock = time.perf_counter()

print("Random Forest Complete.")
print("The time requrired to run was: " + str(tock-tick))

print("\nMetrics for the Random Forest at the Best parameter values of: " +
      str(rf_optimize.best_params_))
print("The Accuracy report of the Random Forest Classifier using best " +
      "parameters: ")
print(rf_report)

#save the best parameters for as a dict for later
rf_dt_params = rf_optimize.best_params_

#%%
#Make the tree plot with graphviz
from graphviz import Source
from sklearn import tree

print("\nBuilding the decision tree... ")
#since I used a grid search before, it does not have the tree_ attribute
#I will need to use the parameters from that tree and build a single tree here

#build the sinlge tree using the best params from the RF classifier
#despite the 'best' leaf value being '1', I determined that a similar
    #accuracy could be obtained with a leaf value of '100.  This also
    #resulted in a much simpler tree
rf_dt = tree.DecisionTreeClassifier(criterion=rf_dt_params['criterion'],
                                    max_depth=rf_dt_params['max_depth'],
                                    min_samples_split=rf_dt_params['min_samples_split'],
                                    min_samples_leaf=100)

#fit the tree on test data
rf_dt.fit(X_train, y_train)

#store the dot data for a tree export
dot_data = tree.export_graphviz(rf_dt, out_file=None,
                                feature_names=X_train.columns)

#Display the tree (does not work for me in Spyder)
Source(dot_data)
#Set filename
fname = "J:\DSDegree\PennState\DAAN_862\Week 14_Final\Final\Random_Forest_DecisionTree_msl_test"
#Save the tree out to a pdf file
Source(dot_data).render(fname)

print("Decision tree has been built.  Check your folder for the "+
      str(fname) + ".pdf file.")

#Information about the trees generated for this data set
print("Looking at the trees generated, it is of no surprise that it " + 
      "is a wide but shallow tree.  This occurs due to the partitioning of " +
      "the 'number of reviews' and 'average review score' being partitioned " +
      "in various increments.  The other features are true/false booleans.")

print("\nAfter running a variety of min_sample_leaf values, I determined" +
      " that a similar scoring tree could be built with " +
      "min_sample_leaf = 100.  This resulted in a much slimmer and easier" +
      " to traverse tree.")

print("\nThese pdfs have been uploaded into Canvas as:"+
      "\n  Random_Forest_DecisionTree_msL_1" +
      "\n  Random_Forest_DecisionTree_msL_10" + 
      "\n  Random_Forest_DecisionTree_msL_100\n")

#%%
#BEGIN THE ADABOOST CLASSIFIER TESTS TO DETERMINE A GOOD N_ESTIMATOR VALUE
#15 minute runtime on my PC

#split the data into train / test 
from sklearn.model_selection import train_test_split

#Import AdaBoost Classifier
from sklearn.ensemble import AdaBoostClassifier
#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

#build empty dataframe for the adaboost items
ada_df = pd.DataFrame()

#Send through a single run
runs = range(0,1)

#make a smaller step size for adaboost
#Number of estimators to user for n_estimators
ada_n_estim = range(10,100,1)

tick = time.perf_counter()
#Loop for 10 runs
for j in runs:
    

    #empty list to append to for later
    ada_accuracy = []

    #Reset the train test split
    X_train, X_test, y_train, y_test = train_test_split(clean_data,
                                                        y,
                                                        test_size=0.3)
    
    #loop over the count of n_estimators 0 - 200 by 2 from above
    for i in ada_n_estim:
        
        #build the model object
        ##Values for the classifier were taken from the best random forest parameters
        base_est = DecisionTreeClassifier(criterion='gini', max_depth=25, 
                                          min_samples_leaf=10,
                                          min_samples_split=10)
        
        #build the adaboost object with a Dec Tree Classifier, modify learning rate?
        ada = AdaBoostClassifier(base_estimator=base_est,
                                 n_estimators = i, learning_rate = 1,
                                 random_state=226)
        
        #run the CV on the AdaBoost classifier and get scores
        ada_score = cross_val_score(ada, X_train, y_train, cv = 10)
        
        #put scores for each n_estimator into the ada_accuacy
        ada_accuracy.append(ada_score.mean())
        
                
    #put int the new data frame column
    runname = "run_" + str(j)
    ada_df.insert(j, runname, ada_accuracy)
    
tock = time.perf_counter()

#create some formating for the plot
s = []
for i in ada_n_estim:
    s.append(i)
#set the index to count by 2's for the plot
ada_df = ada_df.set_index([s])

#plot the Ensemble Accuracy
plt.figure()
ada_df.plot()
plt.title("AdaBoost Accuracy")
plt.ylabel('Accuracy',)
plt.xlabel("N_Estimators in Model")  
plt.legend(loc = 1, fontsize = 6)

#don't reindex this
avg_best_ada_n = int(ada_df.idxmax().mean())

#run the adaBoost with the best_average_n_estimator
ada_best_est = AdaBoostClassifier(n_estimators=avg_best_ada_n, 
                                  learning_rate = 0.1)
#fit
ada_best_est.fit(X_train, y_train)

#predict
ada_best_est_pred = ada_best_est.predict(X_test)

#metrics
ada_best_report = metrics.classification_report(y_test, ada_best_est_pred)

#Had to add one since the ada n_estimators  run from 1 - 100
print("\nMetrics for the AdaBoost at the Best n_estimator value of: " +
      str(avg_best_ada_n))
print(ada_best_report)
print("The time requrired to run was: " + str(tock-tick))

#print("\nBased on the plot, I can get away with an n_estimator value of 60.")
print("\nLet's see if I can optimize AdaBoost a little more.")


#%%

#Begin the Adaboost Classifier based on Values returned from testing (warm-ups)

#Train test split
X_train, X_test, y_train, y_test = train_test_split(clean_data, y, 
                                                    test_size = 0.3)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

#ada_params has been trimmed down from [0.001, 0.1, 1, 10]
ada_params = [{'learning_rate': [0.1, 0.5]}]

#build the model object
#Values for the classifier were taken from the best random forest parameters
base_est = DecisionTreeClassifier(criterion='gini', max_depth=25, 
                                  min_samples_leaf=10,
                                  min_samples_split=10)

#the best estimators have also been parsed down from prior runs to 
#50 based on the AdaBoost Accuracy graph
ada_class = AdaBoostClassifier(base_estimator=base_est, n_estimators=50)

tick = time.perf_counter()
#Run the grid search
ada_optimize = GridSearchCV(estimator=ada_class, param_grid=ada_params,
                            cv=10, n_jobs=-1)
#Fit the model
ada_optimize.fit(X_train, y_train)
#predict on the test data
ada_pred = ada_optimize.predict(X_test)

tock = time.perf_counter()
#get the metric report on accuracy
ada_report = metrics.classification_report(y_test, ada_pred)

print("AdaBoost Complete.")
print("The time requrired to run was: " + str(tock-tick))

print("\nMetrics for AdaBoost at the Best parameter values of: " +
      str(ada_optimize.best_params_))
#print("The Accuracy report of the AdaBoost Classifier using n_est = 60 " +
#     "parameters: ")
print(ada_report)





#%%
#BEGIN LOGISTIC CLASSIFIER HERE

#I can't imagine a possible way for the Logistic classifier to work well
    #given the high dimensionality of the data
#I run it here to see what a poorly chosen model type for the problem
   #would result in

#Train test split
X_train, X_test, y_train, y_test = train_test_split(clean_data, y, 
                                                    test_size = 0.3)

from sklearn import linear_model

#Cs for inverse regularization strength
C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

tick = time.perf_counter()
#Create the linear model and run it through CV with the inv reg strength grid
lrc_optimize = linear_model.LogisticRegressionCV(Cs=C, max_iter= 100000,
                                                 n_jobs = -1)

#fit the model
lrc_optimize.fit(X_train, y_train)

#predict outcomes
lrc_pred = lrc_optimize.predict(X_test)

tock = time.perf_counter()

#create metrics report
lrc_report = metrics.classification_report(y_test, lrc_pred)

print("Logistic Regression Complete.")
print("The time requrired to run was: " + str(tock-tick))

print("\nMetrics for Logistic Regression at the coeficient values of:\n " +
      str(lrc_optimize.coef_) + 
      "\nwhile looking at " + str(lrc_optimize.n_features_in_) + 
      " total features.")
print("\nThe Accuracy report of the Logistic Regression Classifier:")
print(lrc_report)

#As expected, the Log Reg Classifier had worse accuracy when compared to
#the NN and the various Forests




#%%

#I did not model cluster based models given the large dimensionality of the 
    #data set
#I may have been able to reduct the dimensionality by pairing items such as
    #if a user has a preference to tools, and the item they buy is a tool
    #that could be a new category and I could assign it a positive value

#But, given next to no correlation between many parameters, 
#the dimensionality reduction seemed like it would gain very little.



#Question4: An overall summary of the models...

#Generally speaking, the Neural Network often had the best 
#performance followed closely by the Random Forest Classifier and
#AdaBoost.

#The NN performing well was to be expected as there was a high
#degree of dimensionality in the data set (27 features in \ntotal).

#The NN was optimized to determine the optimial number of layers 
#wihtin the network and was able to look at the high dimensionality.
#This would allow the NN to look at how the features were conected 
#to one another (covariance).  

#The random forest (the compilation of Decision Trees), as well as 
#AdaBoost (seperating weights on the branching points), were able to make
#many decisions on the boolean features within the data set.
#This allowed for an accurate model similar to the neural net.  
#The tradeoff within the RF and AdaBoost Decision Tree was that the 
#minimum samples at the leaf (min_samples_leaf) often was over fitted 
#at a value of '1'.  

#The model was able to maintain a similar level of accuracy at a min_leaf_val
#of '100'.  Using a min_leaf_value of 100 allowed for a much simpler
#decision tree to be built as seen in the \nattached PDFs.

#Due to the high dimensioality of the data, I knew that trying 
#to fit the data based on clusters would be difficult.  The \ndata 
#would be too sparse to gain needed insight.  The correlation and 
#pairplots also showed that this would gain little for the model.
#Likewise, a Logistic Regression would not perform as well as the  
#Log Reg model

print("Please see the comment in the code for Question 4 explanation.")


