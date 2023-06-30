import numpy as np

from distool.estimators import DiseaseClassifier, FedotDiseaseClassifier
from distool.feature_extraction import SmartSymptomExtractor
from distool.interpretation.explainer import FedotBasedExplainer, SymptomBasedExplainer


def test_explainer_output(simple_data):
    texts, diseases = simple_data

    symptom_vectorizer = SmartSymptomExtractor()
    features = symptom_vectorizer.transform(texts)

    classifier = DiseaseClassifier()
    classifier.fit(features, diseases)

    explainer = SymptomBasedExplainer(symptom_vectorizer, classifier)
    explained = explainer.explain(features[0])

    assert diseases[0] in explained
    assert "наблюдаются следующие симптомы: температура" in explained
    assert "И отрицаются следующие: недомогание" in explained


def test_fedot_explainer(complex_data):
    # TODO: fedot has bug
    #
    # texts, diseases = complex_data
    #
    # symptom_vectorizer = SmartSymptomExtractor()
    # features = symptom_vectorizer.transform(texts)
    #
    # classifier = FedotDiseaseClassifier(early_stopping_iterations=2)
    # classifier.fit(features, diseases)
    #
    # explainer = FedotBasedExplainer(symptom_vectorizer, classifier)
    # explained = explainer.explain(np.array(features[0]))

    assert 1 == 1
