from abc import ABC, abstractmethod
from typing import List


class BaseEstimator(ABC):
    """BaseEstimator

    This is an abstract base class that provides a common interface for all estimators in the system.
    An estimator is an object that can fit models and make predictions.
    """

    @abstractmethod
    def fit(self, x, y):
        """Fit the model according to the given training data.

        Args:
            x: array-like, shape (n_samples, n_features)
                Training vector, where n_samples is the number of samples and n_features is the number of features.
            y: array-like, shape (n_samples,)
                Target vector relative to x.

        Returns:
            self: object
        """
        pass

    @abstractmethod
    def predict_proba(self, x):
        """Predict class probabilities for X.

        Args:
            x: array-like, shape (n_samples, n_features)
                Vector to be predicted, where n_samples is the number of samples and n_features is the number of features.

        Returns:
            array-like, shape (n_samples, n_classes)
                Returns the probability of the sample for each class in the model.
        """
        pass

    def predict(self, x):
        """Predict class labels for samples in X.

        Args:
            x: array-like, shape (n_samples, n_features)
                Samples.

        Returns:
            array-like, shape (n_samples,)
                Predicted class label per sample.
        """
        pass


class BaseTransformer(ABC):
    """Base transformer

    This is an abstract base class that provides a common interface for all transformers in the system.
    A transformer is an object that can transform input data into a desired format.
    """

    @abstractmethod
    def fit(self, x):
        """Fit the transformer according to the given training data.

        Args:
            x: array-like, shape (n_samples, n_features)
                Training vector, where n_samples is the number of samples and n_features is the number of features.

        Returns:
            self: object
        """
        pass

    @abstractmethod
    def transform(self, messages: List[str]) -> List["Anamnesis"]:
        """Transform the input data into a desired format.

        Args:
            messages: List of strings
                Input data to be transformed.

        Returns:
            List of Anamnesis objects
                Transformed data.
        """
        pass
