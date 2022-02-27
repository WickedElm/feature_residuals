#!/usr/bin/env python

import os
import pickle
from sklearn.neighbors import KNeighborsClassifier

"""
Takes in data and performs a classification using
the sklearn KNN Classifier.
"""

class ClfKNN():
    def __init__(self, n_neighbors=5, class_weight=None):
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, X, y):
        self.model.fit(X, y)

    def forward(self, X):
        y = self.model.predict(X)
        return y

    def predict_proba(self, X):
        probabilities = self.model.predict_proba(X)
        return probabilities

    def score(self, X, y):
        acc = self.model.score(X, y)
        return acc

    def save(self, path=None):
        if path:
            with open(path, 'wb') as f:
                pickle.dump(self, f)

    @staticmethod
    def load(path=None):
        if path and os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
