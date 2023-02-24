from disease.feature_extraction import SymptomExtractor
from disease.interpretation.explainer import SymptomBasedExplainer
from disease.models import DiseaseClassifier

texts = [
    "У меня болит живот, но нет температуры",
    "У меня температура, но нет недомогания",
    "Я завтра иду домой",
]
diseases = ["гастрит", "отит", "-1"]

symptom_vectorizer = SymptomExtractor()
features = symptom_vectorizer.transform(texts)

classifier = DiseaseClassifier()
classifier.fit(features, diseases)
predicted_diseases = classifier.predict(features)
print("Predicted diseases:", predicted_diseases)

explainer = SymptomBasedExplainer(symptom_vectorizer, classifier)
print(explainer.explain(features[1]))
