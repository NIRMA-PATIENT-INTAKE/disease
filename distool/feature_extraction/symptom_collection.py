from typing import Dict, List

from distool.feature_extraction.symptom import Symptom

"""
    Represents all supported symptoms.

"""

SYMPTOM_ENTITY_LABEL_ID: str = "label"
SYMPTOM_ENTITY_LABEL_VALUE: str = "SYMPTOM"
SYMPTOM_ENTITY_PATTERN_ID: str = "pattern"
SYMPTOM_PATTERN_LEMMA_ID: str = "LEMMA"

# TODO: Need to generate all symptoms in proper way!
SYMPTOMS: List[Symptom] = [
    Symptom("температура", ({SYMPTOM_PATTERN_LEMMA_ID: "температура"},)),
    Symptom("недомогание", ({SYMPTOM_PATTERN_LEMMA_ID: "недомогание"},)),
    Symptom(
        "болеть голова",
        ({SYMPTOM_PATTERN_LEMMA_ID: "болеть"}, {SYMPTOM_PATTERN_LEMMA_ID: "голова"}),
    ),
]

NAME_TO_SYMPTOM: Dict[str, Symptom] = {symptom.id_name: symptom for symptom in SYMPTOMS}


def _create_spacy_model_patterns() -> List[Dict]:
    result_patterns = []
    for symptom in SYMPTOMS:
        pattern_record = {
            SYMPTOM_ENTITY_LABEL_ID: SYMPTOM_ENTITY_LABEL_VALUE,
            SYMPTOM_ENTITY_PATTERN_ID: list(symptom.patterns),
        }
        result_patterns.append(pattern_record)

    return result_patterns


SYMPTOMS_SPACY_MODEL_PATTERNS = _create_spacy_model_patterns()
