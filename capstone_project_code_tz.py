get_ipython().magic(u'matplotlib inline')

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy


from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
    roc_curve,
    auc,
    confusion_matrix,
    roc_auc_score
)


# Function to select and load top 10 cluster data
def load_top10cluster():
    parent_df = pd.read_csv('expedia_train.csv')
    top_cluster_list = dict(parent_df['hotel_cluster'].value_counts().head(10)).keys()
    sample_df = parent_df[parent_df['hotel_cluster'].isin(top_cluster_list)]
    #top_cluster_df.to_csv('top_10_cluster.csv')
    #sample_df = sample_df.drop(sample_df.columns[0], axis=1)
    df_lst =[parent_df]
    del df_lst
    return sample_df, top_cluster_list

sample_df, cluster_id = load_top10cluster()
print cluster_id
print sample_df.shape

#Create more feature
sample_df["date_time"] = pd.to_datetime(sample_df["date_time"])
sample_df["year"] = sample_df["date_time"].dt.year
sample_df["month"] = sample_df["date_time"].dt.month
sample_df['day'] = sample_df["date_time"].dt.day
sample_df['day of week'] = sample_df["date_time"].dt.weekday

#Creating training and test set
from sklearn.cross_validation import train_test_split
train, test = train_test_split(sample_df, test_size = 0.3)
print train.shape
print test.shape


# Attribute of the given dataset
feature = ['user_location_region', 'srch_adults_cnt','srch_rm_cnt','is_booking','cnt',
           'hotel_continent', 'hotel_country', 'hotel_market', 'year', 'month', 'day',
           'day of week']

# Categorical feature of given dataset           
cat_features = ['user_location_region', 'hotel_market', 'hotel_country', 'year', 'month',
               'day', 'day of week']
# The set of features that are not categorical.
quant_features = list(set(feature) - set(cat_features))


# For each categorical feature, create dummy variables and add these new data frames to a list.
train_dummy = pd.DataFrame()
test_dummy = pd.DataFrame()
train_dummies = []
test_dummies = []
for feature_name in cat_features:
    train_dummies.append(pd.get_dummies(train[feature_name], prefix=feature_name, sparse=True))
    test_dummies.append(pd.get_dummies(test[feature_name], prefix=feature_name, sparse=True))


# Now concatenate together all of these dummy data frames into one data frame consisting of all dummy variables.
train_aug = pd.DataFrame()
test_aug = pd.DataFrame()
for df in train_dummies:
    train_aug = pd.concat([train_aug, df], axis=1)
for df in test_dummies:
    test_aug = pd.concat([test_aug, df], axis=1)


#Check if levels of categorical features are same in both training and test set
train_columns_to_drop = set(train_aug.columns) - set(test_aug.columns)
test_columns_to_drop = set(test_aug.columns) - set(train_aug.columns)

train_aug.drop(list(train_columns_to_drop), axis=1, inplace=True)
test_aug.drop(list(test_columns_to_drop), axis=1, inplace=True)


#Add the quantitative variables into processed categorical variables
train_aug = pd.concat([train_aug, train[quant_features]], axis=1)
test_aug = pd.concat([test_aug, test[quant_features]], axis=1)

# Sanity check: make sure both datasets contain the same dummy features and levels.
assert (test_aug.columns == train_aug.columns).all()


#Sparse data frames are a bit buggy, so to continue, I dumped Pandas and just turned the data into sparse
# matrices.
train_aug = scipy.sparse.csr_matrix(train_aug.values)
test_aug = scipy.sparse.csr_matrix(test_aug.values)


#Training the logistic regression for each cluster with SGD learning
trained_classifier = []
for i, val in enumerate(cluster_id):
    train_labels = train['hotel_cluster'] == val
    clf = SGDClassifier(loss='log', shuffle=True, penalty='l2')
    clf.fit(train_aug, train_labels)
    trained_classifier.append(clf)
    clf = None
    del clf


#Testing the classifiers
new_prediction = {}
precision = {}
recall = {}
fpr = {}
tpr = {}
a_u_c = {}

#Create a matrix that stores all of the predicted probabilities for each test row, for each classifier.
predicted_probs = np.zeros((test.shape[0], len(cluster_id)))

for i, val in enumerate(cluster_id):
    test_labels = test['hotel_cluster'] == val
    # Here,Column 1 corresponds to target = 1.
    new_prediction = clf.predict_proba(test_aug)[:, 1]
    predicted_probs[:, i] = new_prediction
    precision[i], recall[i], _ = precision_recall_curve(test_labels, new_prediction)
    fpr[i], tpr[i], _ = roc_curve(test_labels, new_prediction)
    a_u_c[i] = roc_auc_score(test_labels, new_prediction)



#Create ROC curve for all 10 cluster
fig = plt.figure(figsize=(10,10))
for i, val in enumerate(cluster_id):
    plt.plot(fpr[i], tpr[i], label='AUC for cluster {} = {}'.format(val, round(a_u_c[i],2)))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.ylim([0.0, 1.02])
    plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], color='k', linestyle='--', linewidth=1)
fig.savefig('ROC_curve.png',bbox_inches='tight')   


#Create precision-recall curve for all 10 cluster
fig = plt.figure(figsize=(10,10))
for i, val in enumerate(cluster_id):
    p, r = precision[i], recall[i]
    f1 = np.mean((2 * (p * r)) / (p + r))
    plt.plot(r, p, label='Mean F1 Score for Cluster {} = {}'.format(val, round(f1, 2)))
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.ylim([0.0, 1.02])
    plt.title('Precision/Recall Diagnostic Curve')
plt.legend(loc="lower left", fontsize=10)
fig.savefig('precision_recall_class_one.png',bbox_inches='tight')   

# Confusion matrix: Get the cluster_id associated with the highest probability for each test row.
predicted_cluster = np.array(cluster_id)[predicted.argmax(axis=1)]
actual_cluster = test['hotel_cluster']  # The actual label.
cm = confusion_matrix(actual_cluster, predicted_cluster)
cmap=plt.cm.Blues
plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.title('Confusion Matrix for Classifier Ensemble')
plt.colorbar()
tick_marks = np.arange(len(cluster_id))
plt.xticks(tick_marks, cluster_id, rotation=45)
plt.yticks(tick_marks, cluster_id)
plt.tight_layout()
plt.ylabel('True Hotel Cluster')
plt.xlabel('Predicted Hotel Cluster')

