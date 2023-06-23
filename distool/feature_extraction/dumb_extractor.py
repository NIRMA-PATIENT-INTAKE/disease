from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, List, Union

import numpy as np

from distool.base.estimators import BaseTransformer
from distool.feature_extraction.anamnesis import Anamnesis
from distool.feature_extraction.symptom_collection import SymptomCollection


class DumbSymptomExtractor(BaseTransformer):
    """Symptom based text vectorization"""

    def fit(self, x: Iterable[str]):
        pass

    def _transform(self, message: str) -> Anamnesis:
        anamnesis: Anamnesis = Anamnesis()

        for symptom in SymptomCollection.get_symptoms():
            if symptom.id_name in message:
                anamnesis.update_symptom_status(symptom)

        return anamnesis

    def transform(
        self, messages: List[str], as_anamnesis: bool = False
    ) -> Union[List[Anamnesis], np.array]:
        with ThreadPoolExecutor() as executor:
            features = list(executor.map(self._transform, messages))

        if not as_anamnesis:
            features = np.array([anamnesis.get_marks() for anamnesis in features])

        return features
