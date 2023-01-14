from disease.feature_extraction import SymptomExtractor
from disease.interpretation.explainer import SymptomBasedExplainer
from disease.models import DiseaseClassifier

texts = [
    "У меня болит живот, но нет температуры",
    "У меня температура и болит ухо",
    "Я завтра иду домой",
]
diseases = ["гастрит", "отит", "-1"]

symptom_vectorizer = SymptomExtractor()
symptom_vectorizer.fit(texts)

# всего три симптома: болит живот, температура и боль в ухе
# каждый симптом может принимать три значения: 1 если есть, 0 если не упомянуто, -1 если отрицается
# [
#   [1, -1, 0],
#   [0, 1, 1],
#   [0, 0, 0]
# ]
features = symptom_vectorizer.transform(texts)

classifier = DiseaseClassifier()
classifier.fit(features, diseases)

predicted_diseases = classifier.predict(texts)
print(predicted_diseases)
# ["гастрит", "отит", "-1"]

explainer = SymptomBasedExplainer(symptom_vectorizer, classifier)
print(explainer.explain(features[[0]]))
# Гастрит с 86% точностью, потому что у пациента наблюдается боль в животе и отрицается температура
