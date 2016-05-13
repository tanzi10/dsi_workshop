#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### calculate accuracy of decision tree on training set
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,labels)
pred = clf.predict(features)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred,labels)
print "accuracy:",round(acc,3)

#calculate accuracy holding 30% for testing set
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.30, random_state=42)
clf2 = clf.fit(features_train,labels_train)
pred2 = clf.predict(features_test)

acc2 = accuracy_score(pred2,labels_test)
print "updated accuracy:", round(acc2,3)
