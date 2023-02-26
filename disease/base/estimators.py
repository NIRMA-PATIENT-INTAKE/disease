from abc import ABC, abstractmethod
from typing import List

from disease.feature_extraction.anamnesis import Anamnesis


class BaseEstimator(ABC):
    """BaseEstimator"""

    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def predict_proba(self, x):
        pass

    def predict(self, x):
        pass


class BaseTransformer(ABC):
    """Base transformer"""

    @abstractmethod
    def fit(self, x):
        pass

    @abstractmethod
    def transform_single(self, message: str) -> Anamnesis:
        pass

    @abstractmethod
    def transform(self, messages: List[str]) -> List[Anamnesis]:
        pass
