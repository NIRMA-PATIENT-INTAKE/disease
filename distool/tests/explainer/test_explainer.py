from distool.feature_extraction import SmartSymptomExtractor
from distool.interpretation.explainer import SymptomBasedExplainer
from distool.estimators import DiseaseClassifier, FedotDiseaseClassifier


def test_explainer_output():
    texts = ["У меня температура, но нет недомогания", "У меня температура"]
    diseases = ["a", "b"]

    symptom_vectorizer = SmartSymptomExtractor()
    features = symptom_vectorizer.transform(texts)

    classifier = BaseDiseaseClassifier()
    classifier.fit(features, diseases)

    explainer = SymptomBasedExplainer(symptom_vectorizer, classifier)
    explained = explainer.explain(features[0])

    assert diseases[0] in explained
    assert "наблюдаются следующие симптомы: температура" in explained
    assert "И отрицаются следующие: недомогание" in explained


def test_fedot_explainer():
    texts = ["У меня температура, но нет недомогания", "У меня температура"]
    diseases = ["a", "b"]

    symptom_vectorizer = SymptomExtractor()
    features = symptom_vectorizer.transform(texts)

    classifier = FedotDiseaseClassifier()
    classifier.fit(features, diseases)

    explainer = SymptomBasedExplainer(symptom_vectorizer, classifier)
    explained = explainer.explain(features[0])

    print(explained)
