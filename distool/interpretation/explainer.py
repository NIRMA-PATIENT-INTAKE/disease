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
    def __init__(
        self, vectorizer: SmartSymptomExtractor, classifier: DiseaseClassifier
    ) -> None:
        """Symptom Based Explainer
        :param vectorizer: fitted symptom extractor
        :param classifier: fitted classifier
        """
        self._vectorizer = vectorizer
        self._classifier = classifier

    def explain(self, feature: np.array) -> str:
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
    def __init__(
        self, vectorizer: SmartSymptomExtractor, classifier: FedotDiseaseClassifier
    ) -> None:
        """Symptom Based Explainer
        :param vectorizer: fitted symptom extractor
        :param classifier: fitted classifier on fedot framework
        """
        assert isinstance(
            classifier, FedotDiseaseClassifier
        ), "Classifier should be FEDOT classifier"

        self._vectorizer = vectorizer
        self._classifier = classifier

    def explain(self, feature: np.array) -> str:
        explainer = self._classifier.model.explain(
            features=feature[np.newaxis, :], visualization=True
        )
        return explainer
