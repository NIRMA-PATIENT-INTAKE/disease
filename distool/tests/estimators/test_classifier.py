from distool.estimators import DiseaseClassifier
from distool.feature_extraction import SmartSymptomExtractor


def test_classifier(simple_data):
    texts, diseases = simple_data

    symptom_vectorizer = SmartSymptomExtractor()
    features = symptom_vectorizer.transform(texts)

    classifier = DiseaseClassifier()
    classifier.fit(features, diseases)

    assert (classifier.predict(features) == diseases).all()
