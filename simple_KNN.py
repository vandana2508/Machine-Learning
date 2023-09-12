#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Reeshma Mantena
        Soumya Shetty
        Vamsi Krishna Muppala
        Vandana Priya Muppala
File name: simple_KNN.py
Specification: Analyzing the dataset, building and evaluating a model with K-Nearest neighbor using Python.
For: CS-5331 Machine Learning and Information Security Section 001   

"""

# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
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

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Finding Accuracy
from sklearn.metrics import accuracy_score
accuracies = accuracy_score(y_test, y_pred)
print ("The accuracy (80/20) is %f" % accuracies)

from sklearn.model_selection import cross_val_score
# creating odd list of K for KNN
myList = list(range(1,50))

# subsetting just the odd ones
neighbors = filter(lambda x: x % 2 != 0, myList)
neighbors = list(range(1,50))
# empty list that will hold cv scores
cv_scores = []

# perform 20-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=20, scoring='accuracy')
    cv_scores.append(scores.mean())

# changing to misclassification error
MSE = [1 - x for x in cv_scores]
MSE_list = np.array(MSE)
neighbors_list = np.array(neighbors)
# determining best k
optimal_k = neighbors[MSE_list.tolist().index(min(MSE_list))]
print ("The optimal number of neighbors(80/20) is %d" % optimal_k)
print ("The MSE (80/20) is %f" % min(MSE_list))

# plot misclassification error vs k
plt.plot(neighbors_list, MSE_list)
plt.xlabel('Number of Neighbors K  (80/20)')
plt.ylabel('Misclassification Error')
plt.show()

#--------------------------------------------------------------------
# Splitting the dataset into the Training set and Test set(70/30)
from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X1_train = sc.fit_transform(X1_train)
X1_test = sc.transform(X1_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
classifier.fit(X1_train, y1_train)

# Predicting the Test set results
y1_pred = classifier.predict(X1_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y1_test, y1_pred)

#Finding Accuracy
from sklearn.metrics import accuracy_score
accuracies = accuracy_score(y1_test, y1_pred)
print ("The accuracy (70/30) is %f" % accuracies)

from sklearn.model_selection import cross_val_score
# creating odd list of K for KNN
myList = list(range(1,50))

# subsetting just the odd ones
neighbors = filter(lambda x: x % 2 != 0, myList)
neighbors = list(range(1,50))
# empty list that will hold cv scores
cv_scores = []

# perform 20-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X1_train, y1_train, cv=20, scoring='accuracy')
    cv_scores.append(scores.mean())

# changing to misclassification error
MSE = [1 - x for x in cv_scores]
MSE_list = np.array(MSE)
neighbors_list = np.array(neighbors)
# determining best k
optimal_k = neighbors[MSE_list.tolist().index(min(MSE_list))]
print ("The optimal number of neighbors(70/30) is %d" % optimal_k)
print ("The MSE (70/30) is %f" % min(MSE_list))

# plot misclassification error vs k
plt.plot(neighbors_list, MSE_list)
plt.xlabel('Number of Neighbors K  (70/30)')
plt.ylabel('Misclassification Error')
plt.show()

#-----------------------------------------------------------------------------
# Splitting the dataset into the Training set and Test set(60/40)
from sklearn.model_selection import train_test_split
X2_train, X2_test, y2_train, y2_test = train_test_split(X, y, test_size = 0.40, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X2_train = sc.fit_transform(X2_train)
X2_test = sc.transform(X2_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
classifier.fit(X2_train, y2_train)

# Predicting the Test set results
y2_pred = classifier.predict(X2_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y2_test, y2_pred)

#Finding Accuracy
from sklearn.metrics import accuracy_score
accuracies = accuracy_score(y2_test, y2_pred)
print ("The accuracy (60/40) is %f" % accuracies)

from sklearn.model_selection import cross_val_score
# creating odd list of K for KNN
myList = list(range(1,50))

# subsetting just the odd ones
neighbors = filter(lambda x: x % 2 != 0, myList)
neighbors = list(range(1,50))
# empty list that will hold cv scores
cv_scores = []

# perform 20-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X2_train, y2_train, cv=20, scoring='accuracy')
    cv_scores.append(scores.mean())

# changing to misclassification error
MSE = [1 - x for x in cv_scores]
MSE_list = np.array(MSE)
neighbors_list = np.array(neighbors)
# determining best k
optimal_k = neighbors[MSE_list.tolist().index(min(MSE_list))]
print ("The optimal number of neighbors(60/40) is %d" % optimal_k)
print ("The MSE (60/40) is %f" % min(MSE_list))

# plot misclassification error vs k
plt.plot(neighbors_list, MSE_list)
plt.xlabel('Number of Neighbors K  (60/40)')
plt.ylabel('Misclassification Error')
plt.show()

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
