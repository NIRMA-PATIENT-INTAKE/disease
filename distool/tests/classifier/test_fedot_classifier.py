from distool.estimators import FedotDiseaseClassifier
from distool.feature_extraction import SmartSymptomExtractor


def test_classifier():
    texts = ["У меня температура, но нет недомогания", "У меня температура"]
    diseases = ["a", "b"]

    symptom_vectorizer = SmartSymptomExtractor()
    features = symptom_vectorizer.transform(texts)

    classifier = FedotDiseaseClassifier()
    classifier.fit(features, diseases)

    assert (classifier.predict(features) == diseases).all()
