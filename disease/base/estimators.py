from abc import ABC, abstractmethod


class BaseEstimator(ABC):
    """ BaseEstimator
    """
    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def predict_proba(self, x):
        pass

    def predict(self, x):
        pass


class BaseTransformer(ABC):
    """ Base transformer
    """

    @abstractmethod
    def fit(self, x):
        pass

    @abstractmethod
    def transform(self, x):
        pass
