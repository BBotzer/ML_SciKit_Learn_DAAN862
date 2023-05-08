# -*- coding: utf-8 -*-
"""
Created on Thu May 10 15:19:36 2018

@author: Leo
"""
# Lesson 6 


import pandas as pd
from sklearn import datasets  
iris = datasets.load_iris()         # load iris data

X,y = iris.data, iris.target
iris.feature_names
iris.target_names
X.shape, y.shape
##############################################################################
# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.4, random_state=0)
X_train.shape, y_train.shape
X_test.shape, y_test.shape

from sklearn import tree
clf = tree.DecisionTreeClassifier().fit(X_train, y_train)
clf.score(X_train, y_train)
clf.score(X_test, y_test)


# Cross validation
from sklearn.model_selection import cross_val_score
# Use decision tree as the classifier
clf = tree.DecisionTreeClassifier()
accuracies = cross_val_score(clf, X_train, y_train, cv = 10)
accuracies
pd.Series(accuracies).describe()


#
from sklearn.model_selection import cross_validate
scoring = ['accuracy', 'precision_macro', 'recall_macro']
scores = cross_validate(clf, X_train, y_train, scoring = scoring,
                        cv = 10, return_train_score = True)

pd.DataFrame(scores).mean()


###############################################################################
#Tuning the hyper parameters 
# Grid search
from sklearn.model_selection import GridSearchCV
tuned_parameters = {'max_depth':[5, 10, 15, 20, 25, 30],
                    'max_features': [2, 3, 4]}

# use accuracy as metrics
tree_clf = GridSearchCV(clf, tuned_parameters, scoring = "accuracy", cv = 10,
                        return_train_score=True)

tree_clf.fit(X_train, y_train)
tree_clf.best_params_

results = pd.DataFrame(tree_clf.cv_results_)
results = results.sort_values("mean_test_score", ascending = False)
results[["params", "mean_test_score", "std_test_score"]]



##############################################################################
# Classification metrics
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# accuracy score
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
accuracy_score(y_test, y_pred, normalize=False)


# Cohen's Kappa
from sklearn.metrics import cohen_kappa_score
cohen_kappa_score(y_test, y_pred)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
pd.DataFrame(cm, index = iris.target_names, columns = iris.target_names)

# binary case:
tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
(tn, fp, fn, tp)


# Classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, target_names=iris.target_names))



# precision, recall, f1-score for binary classification
from sklearn import metrics
metrics.precision_score(y_test, y_pred, average = 'weighted')
metrics.recall_score(y_test, y_pred, average = 'weighted')
metrics.f1_score(y_test, y_pred, average = 'weighted') 



##############################################################################
# Regression metrics
# Use diabete data as the example
diabetes = datasets.load_diabetes()
diabetes.DESCR
diabetes.feature_names
X, y = diabetes.data, diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state = 123)
    
# Use linear regression as the regression method
from sklearn import linear_model
lr = linear_model.LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)


from sklearn.metrics import explained_variance_score
explained_variance_score(y_test, y_pred)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_pred)

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)

from sklearn.metrics import r2_score
r2_score(y_test, y_pred)  

y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]
r2_score(y_true, y_pred, multioutput='variance_weighted')

##############################################################################
# Clustering
# use k-means method
X,y = iris.data, iris.target
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=43)
labels_pred = kmeans.fit_predict(X)
labels_true = y

from sklearn import metrics
metrics.adjusted_rand_score(labels_true, labels_pred)   


metrics.adjusted_mutual_info_score(labels_true, labels_pred)  
metrics.mutual_info_score(labels_true, labels_pred) 
metrics.normalized_mutual_info_score(labels_true, labels_pred)

metrics.homogeneity_score(labels_true, labels_pred) 
metrics.completeness_score(labels_true, labels_pred)
metrics.v_measure_score(labels_true, labels_pred)

# Silhoutte coefficient
kmeans.fit(X)
labels = kmeans.labels_
metrics.silhouette_score(X, labels, metric = 'euclidean')