#!/usr/bin/env python

import numpy as np
import os
import pickle
from sklearn.svm import LinearSVC

"""
Takes in data and performs a classification using
the sklearn Random Forest classsifier.
"""

class ClfLinearSVM():
    def __init__(self, class_weight=None, max_features='auto'):
        self.model = LinearSVC(class_weight=class_weight)
        pass

    def fit(self, X, y):
        self.model.fit(X, y)

    def forward(self, X):
        y = self.model.predict(X)
        return y

    def predict_proba(self, X):
        # Defaulted to satisfy interface
        probabilities = np.zeros((X.shape[0], 1)) 
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
