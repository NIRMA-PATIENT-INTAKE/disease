from disease.feature_extraction import SymptomExtractor
from disease.models import DiseaseClassifier


def test_classifier():
    texts = ["У меня температура, но нет недомогания", "У меня температура"]
    diseases = ["a", "b"]

    symptom_vectorizer = SymptomExtractor()
    features = symptom_vectorizer.transform(texts)

    classifier = DiseaseClassifier()
    classifier.fit(features, diseases)

    assert (classifier.predict(features) == diseases).all()
