from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, List, Union

import numpy as np

from distool.base.estimators import BaseTransformer
from distool.feature_extraction.anamnesis import Anamnesis
from distool.feature_extraction.symptom_collection import SymptomCollection


class DumbSymptomExtractor(BaseTransformer):
    """A simple symptom extractor that checks for the presence of symptom names in a text.

    This class is a specific implementation of the BaseTransformer.
    """

    def fit(self, x: Iterable[str]):
        """Fits the transformer according to the given training data.

        Args:
            x: Iterable over raw text data.
        """
        pass

    def _transform(self, message: str) -> Anamnesis:
        """Transforms a single message into an Anamnesis instance.

        Args:
            message: A string representing a user message.

        Returns:
            An Anamnesis instance.
        """
        anamnesis: Anamnesis = Anamnesis()

        for symptom in SymptomCollection.get_symptoms():
            if symptom.id_name in message:
                anamnesis.update_symptom_status(symptom)

        return anamnesis

    def transform(
        self, messages: List[str], as_anamnesis: bool = False
    ) -> Union[List[Anamnesis], np.array]:
        """Transforms a list of messages into a list of Anamnesis instances or a numpy array.

        Args:
            messages: A list of strings representing user messages.
            as_anamnesis: A boolean indicating whether to return the result as a list of Anamnesis instances. If False, the result is returned as a numpy array.

        Returns:
            A list of Anamnesis instances or a numpy array.
        """
        with ThreadPoolExecutor() as executor:
            features = list(executor.map(self._transform, messages))

        if not as_anamnesis:
            features = np.array([anamnesis.get_marks() for anamnesis in features])

        return features
