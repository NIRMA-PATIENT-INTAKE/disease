from distool.feature_extraction import SymptomExtractor
from distool.interpretation.explainer import SymptomBasedExplainer
from distool.models import DiseaseClassifier


def test_explainer_output():
    texts = ["У меня температура, но нет недомогания", "У меня температура"]
    diseases = ["a", "b"]

    symptom_vectorizer = SymptomExtractor()
    features = symptom_vectorizer.transform(texts)

    classifier = DiseaseClassifier()
    classifier.fit(features, diseases)

    explainer = SymptomBasedExplainer(symptom_vectorizer, classifier)
    explained = explainer.explain(features[0])

    assert diseases[0] in explained
    assert "наблюдаются следующие симптомы: температура" in explained
    assert "И отрицаются следующие: недомогание" in explained
