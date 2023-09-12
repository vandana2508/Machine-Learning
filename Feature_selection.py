#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Reeshma Mantena
        Soumya Shetty
        Vamsi Krishna Muppala
        Vandana Priya Muppala
File name: Feature_selection.py
Specification: Analyzing the dataset, building and evaluating models(K-NN, Random Forest, XGBoost) after feature selection using Python.
For: CS-5331 Machine Learning and Information Security Section 001   

"""

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('data.csv', sep = '|')
X = dataset.drop(['Name', 'md5', 'legitimate'], axis = 1).values
y = dataset['legitimate'].values

# Tree-based feature selection:
from sklearn.feature_selection import SelectFromModel
import sklearn.ensemble as ske
fsel = ske.ExtraTreesClassifier().fit(X, y)
model = SelectFromModel(fsel, prefit=True)
X_new = model.transform(X)
nb_features = X_new.shape[1]
indices = np.argsort(fsel.feature_importances_)[::-1][:nb_features]
for f in range(nb_features):
    print("%d. feature %s (%f)" % (f + 1, dataset.columns[2+indices[f]], fsel.feature_importances_[indices[f]]))
features = []
for f in sorted(np.argsort(fsel.feature_importances_)[::-1][:nb_features]):
    features.append(dataset.columns[2+f])
    
    
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#------------------------K-NN--------------------------------------
from sklearn.neighbors import KNeighborsClassifier
classifier1 = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
classifier1.fit(X_train, y_train)

# Predicting the Test set results
y_pred1 = classifier1.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
#Finding Accuracy
cm = confusion_matrix(y_test, y_pred1)
from sklearn.metrics import accuracy_score
accuracies = accuracy_score(y_test, y_pred1)
print ("The accuracy for KNN after feature selection is %f" % accuracies)


#------------------------------------------------------------------

#-----------------Random-Forest------------------------------------
from sklearn.ensemble import RandomForestClassifier
classifier2 = RandomForestClassifier(n_estimators = 70, criterion = 'entropy')
classifier2.fit(X_train, y_train)

#predict the test results
y_pred2 = classifier2.predict(X_test)

#Makeing the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred2)

#Finding Accuracy
from sklearn.metrics import accuracy_score
accuracies = accuracy_score(y_test, y_pred2)
print ("The accuracy for Random forest after feature selection is %f" % accuracies)

#------------------------------------------------------------------

#-------------------XGBoost----------------------------------------
from xgboost import XGBClassifier
classifier3 = XGBClassifier(max_depth=20, learning_rate=0.3, n_estimators=150)
classifier3.fit(X_train, y_train)

#predict the test results
y_pred3 = classifier3.predict(X_test)

#Makeing the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred3)

#Finding Accuracy
from sklearn.metrics import accuracy_score
accuracies = accuracy_score(y_test, y_pred3)
print ("The accuracy for XGBoost after feature selection is %f" % accuracies)

#Roc Curve
fpr,tpr,threshold=roc_curve(y_test, y_pred3)
auc=metrics.roc_auc_score(y_test, y_pred3)
plt.plot(fpr,tpr,label="data, auc=" +str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()

#PR Curve
pres,rec,threshold=metrics.precision_recall_curve(y_test, y_pred3)
auc=metrics.auc(rec,pres)
plt.plot(rec,pres,label="data, auc="+str(auc))
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc=4)
plt.ylim([0,1.2])
plt.show()

#-----------------------------------------------------------------