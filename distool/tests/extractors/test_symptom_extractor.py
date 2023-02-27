from distool.feature_extraction import SymptomExtractor
from distool.feature_extraction.symptom_status import SymptomStatus


def test_transform_on_simple_message_1():
    message = "У меня нет температуры, но есть недомогание и не болит голова."
    anamnesis = SymptomExtractor()._transform(message)
    assert anamnesis.get_marks(as_number=False) == [
        SymptomStatus.NO,
        SymptomStatus.YES,
        SymptomStatus.NO,
    ]


def test_transform_on_simple_message_2():
    message = "У меня нет температуры и нет недомогания."
    anamnesis = SymptomExtractor()._transform(message)
    assert anamnesis.get_marks(as_number=False) == [
        SymptomStatus.NO,
        SymptomStatus.NO,
        SymptomStatus.NO_INFO,
    ]


def test_transform_on_cross_symptom():
    message = "У меня нет температуры и есть температура, я запутался."
    anamnesis = SymptomExtractor()._transform(message)
    assert anamnesis.get_marks(as_number=False) == [
        SymptomStatus.CONFUSED,
        SymptomStatus.NO_INFO,
        SymptomStatus.NO_INFO,
    ]


def test_transform_on_cross_symptom_in_sentence():
    message = "У меня нет температуры. У меня есть температура."
    anamnesis = SymptomExtractor()._transform(message)
    assert anamnesis.get_marks(as_number=False) == [
        SymptomStatus.NO,
        SymptomStatus.NO_INFO,
        SymptomStatus.NO_INFO,
    ]
