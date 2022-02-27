#!/usr/bin/env python

import os
import pickle
from sklearn.naive_bayes import GaussianNB

"""
Takes in data and performs a classification using
the sklearn Random Forest classsifier.
"""

class ClfGaussianNaiveBayes():
    def __init__(self, class_weight=None, max_features='auto'):
        self.model = GaussianNB()
        self.class_weight = class_weight
        pass

    def fit(self, X, y):
        self.model.fit(X, y, sample_weight=self.class_weight)

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
