import time

from sklearn.metrics import accuracy_score

from distool.estimators import FedotDiseaseClassifier
from distool.feature_extraction import SmartSymptomExtractor


def test_classifier(complex_data):
    texts, diseases = complex_data

    symptom_vectorizer = SmartSymptomExtractor()
    features = symptom_vectorizer.transform(texts)

    classifier = FedotDiseaseClassifier(early_stopping_iterations=2)

    # bug with nans
    for _ in range(100):
        try:
            classifier.fit(features, diseases)
            break
        except ValueError as e:
            print(repr(e), "Trying...")
            time.sleep(1)

    accuracy = accuracy_score(classifier.predict(features), diseases)

    # TODO: accuracy and time complexity trade off
    assert accuracy >= 0.5
