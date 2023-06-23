from distool.feature_extraction import DumbSymptomExtractor, SmartSymptomExtractor


def test_transform_on_simple_message_1():
    message = "У меня нет температуры, но есть недомогание и не болит голова."
    anamnesis = SmartSymptomExtractor()._transform(message)
    anamnesis.get_marks(as_number=False)


def test_transform_on_simple_message_2():
    message = "У меня нет температуры и нет недомогания."
    anamnesis = SmartSymptomExtractor()._transform(message)
    anamnesis.get_marks(as_number=False)


def test_transform_on_cross_symptom():
    message = "У меня нет температуры и есть температура, я запутался."
    anamnesis = SmartSymptomExtractor()._transform(message)
    anamnesis.get_marks(as_number=False)


def test_transform_on_cross_symptom_in_sentence():
    message = "У меня нет температуры. У меня есть температура."
    anamnesis = SmartSymptomExtractor()._transform(message)
    anamnesis.get_marks(as_number=False)


def test_dumb_transform_on_simple_message_1():
    message = "У меня нет температуры, но есть недомогание и не болит голова."
    anamnesis = DumbSymptomExtractor()._transform(message)
    anamnesis.get_marks(as_number=False)


def test_dumb_transform_on_simple_message_2():
    message = "У меня нет температуры и нет недомогания."
    anamnesis = DumbSymptomExtractor()._transform(message)
    anamnesis.get_marks(as_number=False)
