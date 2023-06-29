import numpy as np
from fedot.api.main import Fedot
from sklearn.linear_model import LogisticRegression

from distool.base.estimators import BaseEstimator


class BaseDiseaseClassifier(BaseEstimator):
    threshold: float = 0.5

    def predict(self, x):
        return self.predict_proba(x) > self.threshold


class DiseaseClassifier(BaseDiseaseClassifier):
    """Disease Classifier"""

    def __init__(self):
        self.log_reg = LogisticRegression()

    def fit(self, features: np.array, y: np.array) -> "BaseDiseaseClassifier":
        self.log_reg.fit(features, y)
        return self

    def predict_proba(self, features: np.array) -> np.array:
        return self.log_reg.predict_proba(features)


class FedotDiseaseClassifier(BaseDiseaseClassifier):
    def __init__(self):
        self.model = Fedot(problem="classification", timeout=5, preset="best_quality")

    def fit(self, features: np.array, y: np.array) -> "FedotDiseaseClassifier":
        if not hasattr(features, "shape"):
            features = np.array(features)

        if not hasattr(y, "shape"):
            y = np.array(y)

        self.model.fit(features, y)
        return self

    def predict_proba(self, x: np.array) -> np.array:
        return self.model.predict_proba(x)


class UrgencyClassifier(BaseEstimator):
    """Urgency Classifier"""

    def __init__(self):
        self.log_reg = LogisticRegression()

    def fit(self, features: np.array, y: np.array) -> "UrgencyClassifier":
        self.log_reg.fit(features, y)
        return self

    def predict_proba(self, features: np.array) -> np.array:
        return self.log_reg.predict_proba(features)

    def predict(self, x):
        return self.log_reg.predict(x)
