import json
import os
from pathlib import Path
from typing import Dict, List

from distool.feature_extraction.symptom import Symptom

"""
    Represents all supported symptoms.

"""

BASE_DIR = Path(__file__).parent.parent


class SymptomCollection:
    SYMPTOM_ENTITY_LABEL_ID: str = "label"
    SYMPTOM_ENTITY_LABEL_VALUE: str = "SYMPTOM"
    SYMPTOM_ENTITY_PATTERN_ID: str = "pattern"
    SYMPTOM_ENTITY_ID: str = "id"

    SYMPTOMS_FILE_PATH: str = BASE_DIR / "data/symptoms.json"
    SYMPTOMS_FILE_TOP_KEY: str = "symptoms"
    SYMPTOM_ID_NAME_KEY: str = "id_name"
    SYMPTOM_PATTERNS_KEY: str = "patterns"

    _symptoms: List[Symptom] = None
    _name_to_symptom_dict: Dict[str, Symptom] = None
    _symptoms_spacy_model_patterns: List[Dict] = None

    @classmethod
    def get_symptoms(cls):
        if cls._symptoms is None:
            cls._symptoms = cls._build_symptoms_from_file()

        return cls._symptoms

    @classmethod
    def _build_symptoms_from_file(cls) -> List[Symptom]:
        f = open(cls.SYMPTOMS_FILE_PATH)
        symptom_data = json.load(f)

        symptoms = []
        for symptom_item in symptom_data[SymptomCollection.SYMPTOMS_FILE_TOP_KEY]:
            id_name = symptom_item[SymptomCollection.SYMPTOM_ID_NAME_KEY]
            patterns = symptom_item[SymptomCollection.SYMPTOM_PATTERNS_KEY]
            symptoms.append(Symptom(id_name, patterns))
        return symptoms

    @classmethod
    def get_name_to_symptom_dict(cls) -> Dict[str, Symptom]:
        if cls._name_to_symptom_dict is None:
            cls._name_to_symptom_dict = {
                symptom.id_name: symptom for symptom in cls.get_symptoms()
            }

        return cls._name_to_symptom_dict

    @classmethod
    def get_spacy_model_patterns(cls) -> List[Dict]:
        if cls._symptoms_spacy_model_patterns is None:
            cls._symptoms_spacy_model_patterns = cls._create_spacy_model_patterns()
        return cls._symptoms_spacy_model_patterns

    @classmethod
    def _create_spacy_model_patterns(cls) -> List[Dict]:
        result_patterns = []
        for symptom in cls.get_symptoms():
            for pattern in symptom.patterns:
                pattern_record = {
                    cls.SYMPTOM_ENTITY_LABEL_ID: cls.SYMPTOM_ENTITY_LABEL_VALUE,
                    cls.SYMPTOM_ENTITY_PATTERN_ID: pattern,
                    cls.SYMPTOM_ENTITY_ID: symptom.id_name,
                }
                result_patterns.append(pattern_record)

        return result_patterns
