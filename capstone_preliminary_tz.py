__author__ = 'tanzinazaman'


import pandas as pd
import numpy as np


#Loading dataset given at Kaggle compitition
import random
filename = 'expedia_train.csv'
n = sum(1 for line in open(filename))-1 #number of records in file (excludes header)
percent = 1
s = n*percent/100 #desired sample size
skip = sorted(random.sample(xrange(1, n+1),n-s)) #the 0-indexed header will not be included in the skip list
df = pd.read_csv(filename,skiprows=skip)

print'number of rows is given dataset:',n
print'number of rows in sample dataset:',s

# check if any NAN is in data frame
print df.isnull().sum().sum(), 'rows have NaN values in the sample dataset\n'
# Find which columns have NaN values 
print df.isnull().any()

keep_columns = ['site_name','posa_continent','user_location_country','user_location_region','user_location_city',
                'user_id','is_mobile','is_package','channel','srch_adults_cnt',
                'srch_children_cnt','srch_rm_cnt','srch_destination_id','srch_destination_type_id','hotel_continent',
               'hotel_country','hotel_market','is_booking','hotel_cluster'] #'orig_destination_distance',
a_df = df.loc[:,(keep_columns)]

from sklearn.cross_validation import train_test_split
train, test = train_test_split(a_df, test_size = 0.3)

feature_train = train[train.columns[:-1]]
label_train = train[train.columns[-1]]

feature_test = test[test.columns[:-1]]
label_test = test[test.columns[-1]]

# Fitting model: Decision tree
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(min_samples_split = 40)
clf = clf.fit(feature_train,label_train)
pred = clf.predict(feature_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred,label_test)
print "accuracy:",round(acc,3)

from sklearn.metrics import precision_score
print 'precision:', round(precision_score(label_test,pred,average='binary',),3)

from sklearn.metrics import f1_score
print 'f1 score', round(f1_score(label_test,pred,average='binary',),3)

from sklearn.metrics import recall_score
print 'recall score', round(recall_score(label_test,pred,average='binary',),3)

#Fitting model: SVM
from sklearn.svm import SVC
clf_svc= SVC(kernel="linear")
clf_svc= clf_svc.fit(feature_train,label_train)
pred_svc = clf_svc.predict(label_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred_svc,label_test)
print "accuracy:",round(acc,3)

from sklearn.metrics import precision_score
print 'precision:', round(precision_score(label_test,pred_svc,average='binary',),3)

from sklearn.metrics import f1_score
print 'f1 score', round(f1_score(label_test,pred_svc,average='binary',),3)

from sklearn.metrics import recall_score
print 'recall score', round(recall_score(label_test,pred_svc,average='binary',),3)

#Fitting model: Logistic regression
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(multi_class = 'ovr',solver ='lbfgs' )
clf = clf.fit(feature_train,label_train)
pred = clf.predict(feature_test)
clf.score(X,y)
