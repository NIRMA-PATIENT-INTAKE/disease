import numpy as np
from sklearn.linear_model import LogisticRegression

from distool.base.estimators import BaseEstimator


class DiseaseClassifier(BaseEstimator):
    """Disease Classifier"""

    def __init__(self):
        self.log_reg = LogisticRegression()

    def fit(self, features: np.array, y: np.array) -> "DiseaseClassifier":
        self.log_reg.fit(features, y)
        return self

    def predict_proba(self, features: np.array) -> np.array:
        return self.log_reg.predict_proba(features)

    def predict(self, x):
        return self.log_reg.predict(x)
