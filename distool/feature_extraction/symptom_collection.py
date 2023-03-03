import json
import os
from typing import Dict, List

from distool.feature_extraction.symptom import Symptom

"""
    Represents all supported symptoms.

"""

SYMPTOM_ENTITY_LABEL_ID: str = "label"
SYMPTOM_ENTITY_LABEL_VALUE: str = "SYMPTOM"
SYMPTOM_ENTITY_PATTERN_ID: str = "pattern"
SYMPTOM_ENTITY_ID: str = "id"

SYMPTOMS_FILE_PATH: str = os.path.join(
    os.path.dirname(__file__), "../data/symptoms.json"
)
SYMPTOMS_FILE_TOP_KEY = "symptoms"
SYMPTOM_ID_NAME_KEY = "id_name"
SYMPTOM_PATTERNS_KEY = "patterns"


def _build_symptoms_from_file() -> List[Symptom]:
    f = open(SYMPTOMS_FILE_PATH)
    symptom_data = json.load(f)

    symptoms = []
    for symptom_item in symptom_data[SYMPTOMS_FILE_TOP_KEY]:
        id_name = symptom_item[SYMPTOM_ID_NAME_KEY]
        patterns = symptom_item[SYMPTOM_PATTERNS_KEY]
        symptoms.append(Symptom(id_name, patterns))
    return symptoms


SYMPTOMS: List[Symptom] = _build_symptoms_from_file()

NAME_TO_SYMPTOM: Dict[str, Symptom] = {symptom.id_name: symptom for symptom in SYMPTOMS}


def _create_spacy_model_patterns() -> List[Dict]:
    result_patterns = []
    for symptom in SYMPTOMS:
        for pattern in symptom.patterns:
            pattern_record = {
                SYMPTOM_ENTITY_LABEL_ID: SYMPTOM_ENTITY_LABEL_VALUE,
                SYMPTOM_ENTITY_PATTERN_ID: pattern,
                SYMPTOM_ENTITY_ID: SYMPTOM_ENTITY_LABEL_VALUE,
            }
            result_patterns.append(pattern_record)

    return result_patterns


SYMPTOMS_SPACY_MODEL_PATTERNS = _create_spacy_model_patterns()
