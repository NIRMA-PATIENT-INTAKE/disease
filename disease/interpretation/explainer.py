from abc import ABC, abstractmethod

from disease.feature_extraction import SymptomExtractor
from disease.base.estimators import BaseEstimator


class BaseExplainer(ABC):
    @abstractmethod
    def explain(self, X):
        pass


class SymptomBasedExplainer(BaseExplainer):
    def __init__(self, vectorizer: SymptomExtractor, classifier: BaseEstimator) -> None:
        """ Symptom Based Explainer
        :param vectorizer: fitted symptom extractor
        :param classifier: fitted classifier
        """
        self._vectorizer = vectorizer
        self._classifier = classifier

    def explain(self, X):
        pass
