import numpy as np
from fedot.api.main import Fedot
from sklearn.linear_model import LogisticRegression

from distool.base.estimators import BaseEstimator


class BaseDiseaseClassifier(BaseEstimator):
    """BaseDiseaseClassifier

    This is an abstract base class that provides a common interface for all disease classifiers in the system.
    A disease classifier is an object that can fit models and make predictions about diseases.
    """

    threshold: float = 0.5
    id2class: dict = {}

    def predict(self, x):
        """Predict class labels for samples in X.

        Args:
            x: array-like, shape (n_samples, n_features)
                Samples.

        Returns:
            array-like, shape (n_samples,)
                Predicted class label per sample.
        """
        logits = self.predict_proba(x)
        class_ids = np.argmax(logits, axis=1)
        classes = np.array([self.id2class[class_id] for class_id in class_ids])
        return classes


class DiseaseClassifier(BaseDiseaseClassifier):
    """Disease Classifier

    This class is a specific implementation of the BaseDiseaseClassifier that uses Logistic Regression for classification.

    Attributes:
        log_reg: A Logistic Regression classifier.
    """

    def __init__(self):
        """Initializes a new instance of the DiseaseClassifier class."""
        self.log_reg = LogisticRegression()

    def fit(self, features: np.array, y: np.array) -> "BaseDiseaseClassifier":
        """Fit the model according to the given training data.

        Args:
            features: array-like, shape (n_samples, n_features)
                Training vector, where n_samples is the number of samples and n_features is the number of features.
            y: array-like, shape (n_samples,)
                Target vector relative to X.

        Returns:
            self: object
        """
        self.log_reg.fit(features, y)
        self.id2class = {i: c for i, c in enumerate(self.log_reg.classes_)}

        return self

    def predict_proba(self, features: np.array) -> np.array:
        """Probability estimates.

        The returned estimates for all classes are ordered by the label of classes.

        Args:
            features: array-like, shape = [n_samples, n_features]
                The input samples.

        Returns:
            p: array-like, shape = [n_samples, n_classes]
                The class probabilities of the input samples. The order of the classes corresponds to that in the attribute `classes_`.
        """
        return self.log_reg.predict_proba(features)


class FedotDiseaseClassifier(BaseDiseaseClassifier):
    """FedotDiseaseClassifier

    This class is a specific implementation of the BaseDiseaseClassifier that uses the FEDOT framework for classification.

    Attributes:
        model: A FEDOT model.
    """

    def __init__(self, **options) -> None:
        """Initializes a new instance of the FedotDiseaseClassifier class."""
        self.model = Fedot(
            **options,
            problem="classification",
            timeout=5,
            preset="best_quality",
            safe_mode=True,
        )

    def fit(self, features: np.array, y: np.array) -> "FedotDiseaseClassifier":
        """Fit the model according to the given training data.

        Args:
            features: array-like, shape (n_samples, n_features)
                Training vector, where n_samples is the number of samples and n_features is the number of features.
            y: array-like, shape (n_samples,)
                Target vector relative to X.

        Returns:
            self: object
        """
        if not hasattr(features, "shape"):
            features = np.array(features)

        if not hasattr(y, "shape"):
            y = np.array(y)

        self.id2class = {i: c for i, c in enumerate(np.unique(y))}
        self.model.fit(features=features, target=y)

        return self

    def predict_proba(self, x: np.array) -> np.array:
        """Probability estimates.

        The returned estimates for all classes are ordered by the label of classes.

        Args:
            x: array-like, shape = [n_samples, n_features]
                The input samples.

        Returns:
            p: array-like, shape = [n_samples, n_classes]
                The class probabilities of the input samples. The order of the classes corresponds to that in the attribute `classes_`.
        """
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
