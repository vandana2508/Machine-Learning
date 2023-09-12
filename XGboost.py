#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Reeshma Mantena
        Soumya Shetty
        Vamsi Krishna Muppala
        Vandana Priya Muppala
File name: XGBoost.py
Specification: Analyzing the dataset, building and evaluating a model with XGBoost using Python.
For: CS-5331 Machine Learning and Information Security Section 001   

"""

# XGboost

# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data.csv', sep = '|')
X = dataset.drop(['Name', 'md5', 'legitimate'], axis = 1).values
y = dataset['legitimate'].values

# Splitting the dataset into the Training set and Test set(80/20)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Fitting xgboost to the training Set
from xgboost import XGBClassifier
classifier = XGBClassifier(max_depth=20, learning_rate=0.3, n_estimators=150)
classifier.fit(X_train, y_train)

#predict the test results
y_pred = classifier.predict(X_test)

#Makeing the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Finding Accuracy
from sklearn.metrics import accuracy_score
accuracies = accuracy_score(y_test, y_pred)
print ("The accuracy (80/20) is %f" % accuracies)

#--------------------------------------------------------------------------

# Splitting the dataset into the Training set and Test set(70/30)
from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size = 0.30, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train1 = sc.fit_transform(X_train1)
X_test1 = sc.transform(X_test1)

#Fitting xgboost to the training Set
from xgboost import XGBClassifier
classifier11 = XGBClassifier(max_depth=20, learning_rate=0.3, n_estimators=150)
classifier11.fit(X_train1, y_train1)

#predict the test results
y_pred11 = classifier11.predict(X_test1)

#Makeing the confusion matrix
from sklearn.metrics import confusion_matrix
cm11 = confusion_matrix(y_test1, y_pred11)

#Finding Accuracy
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
accuracies = accuracy_score(y_test1, y_pred11)
print ("The accuracy (70/30) is %f" % accuracies)


#---------------------------------------------------------------------------------

# Splitting the dataset into the Training set and Test set(60/40)
from sklearn.model_selection import train_test_split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size = 0.40, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train2 = sc.fit_transform(X_train2)
X_test2 = sc.transform(X_test2)

#Fitting xgboost to the training Set
from xgboost import XGBClassifier
classifier21 = XGBClassifier(max_depth=20, learning_rate=0.3, n_estimators=150)
classifier21.fit(X_train1, y_train1)

#predict the test results
y_pred21 = classifier21.predict(X_test2)

#Makeing the confusion matrix
from sklearn.metrics import confusion_matrix
cm21 = confusion_matrix(y_test2, y_pred21)

#Finding Accuracy
from sklearn.metrics import accuracy_score
accuracies = accuracy_score(y_test2, y_pred21)
print ("The accuracy (60/40) is %f" % accuracies)




