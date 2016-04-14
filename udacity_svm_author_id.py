#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time

sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
#########################################################
### your code goes here ###
### Using linear karnel,what is the accuracy of the classifier?
from sklearn.svm import SVC

clf_l = SVC(kernel="linear")
t0 = time()
clf_l = clf_l.fit(features_train, labels_train)
print "training time:", round(time() - t0, 3), "s"

t_p = time()
pred_l = clf_l.predict(features_test)
print "predicting time", round(time() - t_p, 3), "s"

from sklearn.metrics import accuracy_score

acc = accuracy_score(pred_l, labels_test)
print "The accuracy of the classifier with linear karnel:", round(acc, 3)
#########################################################
###what is the accuracy of the classifier, with 1% training data?

clf_less = SVC(kernel="linear")
t_less = time()
features_train_1 = features_train[:len(features_train) / 100]
labels_train_1 = labels_train[:len(labels_train) / 100]
clf_less = clf_less.fit(features_train_1, labels_train_1)
print "training time with 1% feature:", round(time() - t_less, 3), "s"

t_p_less = time()
pred_less = clf_less.predict(features_test)
print "predicting time with 1% feature:", round(time() - t_p_less, 3), "s"

# from sklearn.metrics import accuracy_score
acc_less = accuracy_score(pred_less, labels_test)
print "The accuracy of the classifier with 1% training data:", round(acc_less, 3)

#########################################################
# Accuracy check with 'rbf' kernel and several values of C (10, 100, 1000 and 10000)

clf_rbf = SVC(kernel="rbf", C=10000.0)
t_rbf = time()
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]
clf_rbf = clf_rbf.fit(features_train_1, labels_train_1)
print "training time with rbf karnel:", round(time() - t_rbf, 3), "s"

t_p_rbf = time()
pred_rbf = clf_rbf.predict(features_test)
print "predicting time with rbf karnel:", round(time() - t_p_rbf, 3), "s"

# from sklearn.metrics import accuracy_score
acc_rbf = accuracy_score(pred_rbf, labels_test)
print "The accuracy of the classifier with rbf karnel:", round(acc_rbf, 3)

# What class does your SVM (0 or 1, corresponding to Sara and Chris respectively)
# predict for element 10 of the test set? The 26th? The 50th?
#(Use the RBF kernel, C=10000, and 1% of the training set.)

print "Prediction for 10th element:", pred_rbf[10],",", "26th element:", pred_rbf[26],",", "50th element:", pred_rbf[50]

# repeat with full training set,(by tuning C and training on a large dataset)
# What is the accuracy of the optimized SVM?

clf_rbf = SVC(kernel="rbf", C=10000.0)
t_full = time()
clf_full = clf_rbf.fit(features_train, labels_train)
print "training time with full data set:", round(time() - t_full, 3), "s"

t_p_full = time()
pred_full = clf_rbf.predict(features_test)
print "predicting time with full data set", round(time() - t_p_full, 3), "s"

# from sklearn.metrics import accuracy_score
acc_full = accuracy_score(pred_full, labels_test)
print "The accuracy of optimized SVM:", round(acc_full, 3)

## There are over 1700 test events--how many are

import numpy as np

print sum(pred_full == 1),"are predicted to be in the 'Chris' (1) class"
