from disease.feature_extraction import SymptomExtractor
from disease.feature_extraction.symptom_status import SymptomStatus


def test_transform_on_simple_message_1():
    message = "У меня нет температуры, но есть недомогание и не болит голова."
    anamnesis = SymptomExtractor().transform(message)
    assert anamnesis.get_marks() == [
        SymptomStatus.NO,
        SymptomStatus.YES,
        SymptomStatus.NO,
    ]


def test_transform_on_simple_message_2():
    message = "У меня нет температуры и нет недомогания."
    anamnesis = SymptomExtractor().transform(message)
    assert anamnesis.get_marks() == [
        SymptomStatus.NO,
        SymptomStatus.NO,
        SymptomStatus.NO_INFO,
    ]


def test_transform_on_cross_symptom():
    message = "У меня нет температуры и есть температура, я запутался."
    anamnesis = SymptomExtractor().transform(message)
    assert anamnesis.get_marks() == [
        SymptomStatus.CONFUSED,
        SymptomStatus.NO_INFO,
        SymptomStatus.NO_INFO,
    ]


def test_transform_on_cross_symptom_in_sentence():
    message = "У меня нет температуры. У меня есть температура."
    anamnesis = SymptomExtractor().transform(message)
    assert anamnesis.get_marks() == [
        SymptomStatus.NO,
        SymptomStatus.NO_INFO,
        SymptomStatus.NO_INFO,
    ]
