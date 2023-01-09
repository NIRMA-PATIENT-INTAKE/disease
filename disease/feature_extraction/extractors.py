from typing import Iterable
from disease.base.estimators import BaseTransformer


class SymptomExtractor(BaseTransformer):
    """ Symptom based text vectorization
    """

    def fit(self, x: Iterable[str]):
        pass

    def transform(self, x: Iterable[str]):
        pass
