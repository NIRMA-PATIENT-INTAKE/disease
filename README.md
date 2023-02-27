# Классификатор заболеваний

**Patient-intake** - модуль, обеспечивающий дистанционное взаимодействие пациента и медицинского учреждения с помощью диалоговых агентов. Реализован на языке Python.

Модуль позволяет предсказывать возможное заболевание пациента на основе введенной им информации, присваивать диагнозу степень срочности приема, а также выводить интерпретацию результатов работы модели. Для обучения модели создан датасет, содержащий информацию о болезнях и их симптомах.

Модуль поддерживает анализ текстов на русском языке.

## Применение

Модуль является инструментом для разработки медицинских чат-ботов. Может быть использован для управления потоками пациентов в медицинских учреждениях, а также будет полезен для реализации концепции второго мнения при взаимодействии с пациентом.

## Установка модуля

    pip install distool

## Пример использования

```python
from distool.feature_extraction import SymptomExtractor
from distool.interpretation.explainer import SymptomBasedExplainer
from distool.models import DiseaseClassifier

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
# Predicted diseases: ['гастрит' 'отит' '-1']

explainer = SymptomBasedExplainer(symptom_vectorizer, classifier)

print(explainer.explain(features[1]))
# Наблюдается отит с вероятностью 59%.
# Это потому что у вас наблюдаются следующие симптомы: температура
# И отрицаются следующие: недомогание
```

## Разработка

Нужно установить зависимости и подключить pre-commit.g

    pip install -r requirements.py
    pre-commit install
    bash init.py

### Контрибьютинг

Каждый новый функционал должен покрываться документацией и тестом.

### Тестирование

Чтобы запустить тесты нужно ввести:

    pytest

### Pull request

Каждый PR должен быть отревьювен.

## История проекта

Работа над **Patient-intake** проводится в рамках НИР Университета ИТМО №622275 "Разработка модуля для предварительного диагноза: поддержание логистики потоков пациентов и концепции второго мнения при взаимодействии с пациентом через диалоговые системы" (2022-2023)

## Участники проекта

[<img src="https://images.weserv.nl/?url=avatars.githubusercontent.com/u/53980712?v=4&h=200&w=200&fit=cover&mask=circle&maxage=7d" width="30" /> Анна Чижик](https://github.com/Frantsuzova) </br> руководитель проекта

[<img src="https://images.weserv.nl/?url=avatars.githubusercontent.com/u/23355371?v=4&h=200&w=200&fit=cover&mask=circle&maxage=7d" width="30" /> Мичил Егоров](https://github.com/EgorovM) </br> тимлид

[<img src="https://images.weserv.nl/?url=avatars.githubusercontent.com/u/39134692?v=4&h=200&w=200&fit=cover&mask=circle&maxage=7d" width="30" /> Дмитрий погребной](https://github.com/DmitryPogrebnoy) </br> разработчик

[<img src="https://images.weserv.nl/?url=avatars.githubusercontent.com/u/56276056?v=4&h=200&w=200&fit=cover&mask=circle&maxage=7d" width="30" /> Айталина Кривошапкина](https://github.com/aytakr) </br> разработчик

[<img src="https://images.weserv.nl/?url=avatars.githubusercontent.com/u/109297759?v=4&h=200&w=200&fit=cover&mask=circle&maxage=7d" width="30" /> Мария Якубова](https://github.com/Shentorin) </br> аналитик

## Обратная связь

Если у Вас есть какие-то вопросы, можете обращаться к нам по адресу: patientintake@yandex.ru
