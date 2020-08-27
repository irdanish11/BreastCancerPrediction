# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 21:22:56 2020

@author: Danish
"""


from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
clf = svm.SVC(kernel='linear', C=0.01).fit(X_train, y_train)
Y_score = clf.fit(X_train, y_train).decision_function(X_test)

# Generate predictions
Y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, Y_pred)
print('SVM Model accuracy is: ', accuracy)