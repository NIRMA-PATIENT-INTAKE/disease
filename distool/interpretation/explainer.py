from abc import ABC, abstractmethod

import numpy as np

from distool.estimators.classifiers import DiseaseClassifier, FedotDiseaseClassifier
from distool.feature_extraction import SmartSymptomExtractor
from distool.feature_extraction.symptom_collection import SymptomCollection
from distool.feature_extraction.symptom_status import SymptomStatus


class BaseExplainer(ABC):
    @abstractmethod
    def explain(self, X):
        pass


class SymptomBasedExplainer(BaseExplainer):
    """
    An explainer based on symptoms.

    This class is a specific implementation of the BaseExplainer.

    Attributes:
        _vectorizer: The symptom extractor.
        _classifier: The classifier.
    """

    def __init__(
        self, vectorizer: SmartSymptomExtractor, classifier: DiseaseClassifier
    ) -> None:
        """Initializes a new instance of the SymptomBasedExplainer class.

        Args:
            vectorizer: The symptom extractor.
            classifier: The classifier.
        """
        self._vectorizer = vectorizer
        self._classifier = classifier

    def explain(self, feature: np.array) -> str:
        """Explains the given feature.

        Args:
            feature: A numpy array representing the feature to explain.

        Returns:
            A string representing the explanation.
        """
        symptom_analysis = list(zip(SymptomCollection.get_symptoms(), feature))
        predict_proba = self._classifier.predict_proba([feature])[0]

        disease_proba = predict_proba[np.argmax(predict_proba)]
        disease_name = self._classifier.log_reg.classes_[np.argmax(predict_proba)]
        has_symptoms = [
            s.id_name for s, f in symptom_analysis if f == SymptomStatus.YES.value
        ]
        no_symptoms = [
            s.id_name for s, f in symptom_analysis if f == SymptomStatus.NO.value
        ]

        return (
            f"Наблюдается {disease_name} с вероятностью {round(disease_proba * 100)}%.\n"
            f"Это потому что у вас наблюдаются следующие симптомы: "
            f"{''.join(has_symptoms)}\n"
            f"И отрицаются следующие: "
            f"{''.join(no_symptoms)}"
        )


class FedotBasedExplainer(BaseExplainer):
    """
    An explainer based on the FEDOT framework.

    This class is a specific implementation of the BaseExplainer.

    Attributes:
        _vectorizer: The symptom extractor.
        _classifier: The classifier.
    """

    def __init__(
        self, vectorizer: SmartSymptomExtractor, classifier: FedotDiseaseClassifier
    ) -> None:
        """Initializes a new instance of the FedotBasedExplainer class.

        Args:
            vectorizer: The symptom extractor.
            classifier: The classifier.
        """
        assert isinstance(
            classifier, FedotDiseaseClassifier
        ), "Classifier should be FEDOT classifier"

        self._vectorizer = vectorizer
        self._classifier = classifier

    def explain(self, feature: np.array) -> str:
        """Explains the given feature.

        Args:
            feature: A numpy array representing the feature to explain.

        Returns:
            A string representing the explanation.
        """
        explainer = self._classifier.model.explain(
            features=feature[np.newaxis, :], visualization=True
        )
        return explainer
