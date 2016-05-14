#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### code goes here

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.30, random_state=42)
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf2 = clf.fit(features_train,labels_train)
pred2 = clf.predict(features_test)

from sklearn.metrics import accuracy_score
acc2 = accuracy_score(pred2,labels_test)
print "accuracy of identifier:", round(acc2,3)

# No of POI predicted for test set
a = 0
for i in labels_test:
    if i != 0.0:
        a += 1
    else:
        a

print "POIs predicted for the test set:" , a
print "Number of people in test set:" ,len(labels_test)

#if identifier predicted 0.0 for everyone in test set
pred_zero = [0.0]* len(labels_test)
acc_zero = accuracy_score(pred_zero,labels_test)
print "accuracy of biased identifier:", round(acc_zero,3)

#print 'true labels:', labels_test
#print 'prediction:', pred2
# Count true positive, i.e. count of 1.0 in prediction
tp = 0
for i in pred_zero:
    if i!=0.0:
        tp +=1
    else:
        tp
print 'true positive:', tp

# calculate precision
from sklearn.metrics import precision_score
print 'precision:', precision_score(labels_test,pred2)

# calculate recall
from sklearn.metrics import recall_score
print 'recall:', recall_score(labels_test,pred2)

# How many true positive for given below data set
predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

true_positive = [i for i,j in zip(predictions,true_labels) if i == 1 and j==1]
print "Count of true positive:",len(true_positive)

#How many true negatives in given data
true_negative = [i for i,j in zip(predictions,true_labels) if i == 0 and j==0]
print "Count of true negative:", len(true_negative)

# How many false positive
false_positive = [i for i,j in zip(predictions,true_labels) if i == 1 and j==0]
print 'Count of false positive:', len(false_positive)

# How many false negative
false_negative = [i for i,j in zip(predictions,true_labels) if i == 0 and j==1]
print 'Count of false negative:', len(false_negative)

#Precision
print 'precision:', round(precision_score(true_labels,predictions),3)

#Recall
print 'recall:', round(recall_score(true_labels,predictions),3)