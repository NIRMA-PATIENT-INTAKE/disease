from disease.feature_extraction import SymptomExtractor


def test_transform_on_simple_message_1():
    message = "У меня нет температуры, но есть недомогание и не болит голова."
    disease_vector = SymptomExtractor().transform(message)
    assert disease_vector == [-1, 1, -1]


def test_transform_on_simple_message_2():
    message = "У меня нет температуры и нет недомогания."
    disease_vector = SymptomExtractor().transform(message)
    assert disease_vector == [-1, -1, 0]


def test_transform_on_cross_symptom():
    message = "У меня нет температуры и есть температура, я запутался."
    disease_vector = SymptomExtractor().transform(message)
    assert disease_vector == [-1, 0, 0]
