#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Reeshma Mantena
        Soumya Shetty
        Vamsi Krishna Muppala
        Vandana Priya Muppala
File name: Random Forest.py
Specification: Analyzing the dataset, building and evaluating a model with Random Forest using Python.
For: CS-5331 Machine Learning and Information Security Section 001   

"""

#Random Forest

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

from sklearn.ensemble import RandomForestClassifier
classifier01 = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0)
classifier01.fit(X_train, y_train)

classifier02 = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier02.fit(X_train, y_train)

classifier03 = RandomForestClassifier(n_estimators = 70, criterion = 'entropy', random_state = 0)
classifier03.fit(X_train, y_train)


#predict the test results
y_pred01 = classifier01.predict(X_test)
y_pred02 = classifier02.predict(X_test)
y_pred03 = classifier03.predict(X_test)


#Makeing the confusion matrix
from sklearn.metrics import confusion_matrix
cm01 = confusion_matrix(y_test, y_pred01)
cm02 = confusion_matrix(y_test, y_pred02)
cm03 = confusion_matrix(y_test, y_pred03)

#Finding Accuracy
from sklearn.metrics import accuracy_score
accuracies = accuracy_score(y_test, y_pred03)
print ("The accuracy (80/20) is %f" % accuracies)

#---------------------------------------------------------------------------------------------
# Splitting the dataset into the Training set and Test set(70/30)
from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X1_train = sc.fit_transform(X1_train)
X1_test = sc.transform(X1_test)

from sklearn.ensemble import RandomForestClassifier
classifier11 = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0)
classifier11.fit(X1_train, y1_train)

classifier12 = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier12.fit(X1_train, y1_train)

classifier13 = RandomForestClassifier(n_estimators = 70, criterion = 'entropy', random_state = 0)
classifier13.fit(X1_train, y1_train)


#predict the test results
y_pred11 = classifier11.predict(X1_test)
y_pred12 = classifier12.predict(X1_test)
y_pred13 = classifier13.predict(X1_test)


#Makeing the confusion matrix
from sklearn.metrics import confusion_matrix
cm11 = confusion_matrix(y1_test, y_pred11)
cm12 = confusion_matrix(y1_test, y_pred12)
cm13 = confusion_matrix(y1_test, y_pred13)

#Finding Accuracy
from sklearn.metrics import accuracy_score
accuracies = accuracy_score(y1_test, y_pred13)
print ("The accuracy (70/30) is %f" % accuracies)

#--------------------------------------------------------------------------------------
# Splitting the dataset into the Training set and Test set(60/40)
from sklearn.model_selection import train_test_split
X2_train, X2_test, y2_train, y2_test = train_test_split(X, y, test_size = 0.40, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X2_train = sc.fit_transform(X2_train)
X2_test = sc.transform(X2_test)

from sklearn.ensemble import RandomForestClassifier
classifier21 = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0)
classifier21.fit(X1_train, y1_train)

classifier22 = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier22.fit(X_train, y_train)

classifier23 = RandomForestClassifier(n_estimators = 70, criterion = 'entropy', random_state = 0)
classifier23.fit(X_train, y_train)


#predict the test results
y_pred21 = classifier21.predict(X2_test)
y_pred22 = classifier22.predict(X2_test)
y_pred23 = classifier23.predict(X2_test)


#Makeing the confusion matrix
from sklearn.metrics import confusion_matrix
cm21 = confusion_matrix(y2_test, y_pred21)
cm22 = confusion_matrix(y2_test, y_pred22)
cm23 = confusion_matrix(y2_test, y_pred23)

#Finding Accuracy
from sklearn.metrics import accuracy_score
accuracies = accuracy_score(y2_test, y_pred23)
print ("The accuracy (60/40) is %f" % accuracies)