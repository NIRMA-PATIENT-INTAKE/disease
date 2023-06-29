import numpy as np
from fedot.api.main import Fedot
from sklearn.linear_model import LogisticRegression

from distool.base.estimators import BaseEstimator


class BaseDiseaseClassifier(BaseEstimator):
    threshold: float = 0.5
    id2class: dict = {}

    def predict(self, x):
        logits = self.predict_proba(x)
        class_ids = np.argmax(logits, axis=1)
        classes = np.array([self.id2class[class_id] for class_id in class_ids])
        return classes


class DiseaseClassifier(BaseDiseaseClassifier):
    """Disease Classifier"""

    def __init__(self):
        self.log_reg = LogisticRegression()

    def fit(self, features: np.array, y: np.array) -> "BaseDiseaseClassifier":
        self.log_reg.fit(features, y)
        self.id2class = {i: c for i, c in enumerate(self.log_reg.classes_)}

        return self

    def predict_proba(self, features: np.array) -> np.array:
        return self.log_reg.predict_proba(features)


class FedotDiseaseClassifier(BaseDiseaseClassifier):
    def __init__(self, **options) -> None:
        self.model = Fedot(
            **options,
            problem="classification",
            timeout=5,
            preset="best_quality",
            safe_mode=True,
        )

    def fit(self, features: np.array, y: np.array) -> "FedotDiseaseClassifier":
        if not hasattr(features, "shape"):
            features = np.array(features)

        if not hasattr(y, "shape"):
            y = np.array(y)

        self.id2class = {i: c for i, c in enumerate(np.unique(y))}
        self.model.fit(features=features, target=y)

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
