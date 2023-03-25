import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import spacy
from spacy import Language
from spacy.tokens import Doc

from distool import SymptomExtractor
from distool.feature_extraction.symptom_collection import SymptomCollection

PATH_TO_SYMPTOM_DATASET = "../data/symptoms_datasets/symptoms_dataset.csv"
SYMPTOM_LEMMA_PATTERN_KEY = "LEMMA"


def main():
    symptoms_list: List[List[str]] = load_and_preprocess_new_symptoms_list()
    symptoms_rules: Dict = {}
    symptoms_rules: Dict = add_patterns_from_symptoms_list(
        symptoms_rules, symptoms_list
    )
    prepared_symptoms_rules: Dict[str, List[Dict]] = transform_symptom_rules_for_json(
        symptoms_rules
    )
    save_symptoms_rules_to_json(prepared_symptoms_rules)


def load_and_preprocess_new_symptoms_list() -> List[List[str]]:
    df = pd.read_csv(PATH_TO_SYMPTOM_DATASET)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df["Симптомы"].replace("", np.NaN, inplace=True)
    df.dropna(inplace=True)

    df["Симптомы"] = df["Симптомы"].map(lambda x: x.split(";"))
    return list(df["Симптомы"])


def add_patterns_from_symptoms_list(
    symptoms_rules: Dict[str, Dict], symptoms_list: List[List[str]]
) -> Dict[str, Dict]:
    spacy_model: Language = spacy.load(SymptomExtractor.SPACY_LANG_MODEL_NAME)

    for symptoms_item in symptoms_list:
        for symptom in symptoms_item:
            preprocessed_symptom: str = symptom.strip()
            symptom_doc: Doc = spacy_model(preprocessed_symptom)
            symptom_lemmas: List[str] = [token.lemma_ for token in symptom_doc]
            symptom_name_id: str = " ".join(symptom_lemmas)

            if symptom_name_id in symptoms_rules:
                symptom_rule: Dict = symptoms_rules[symptom_name_id]
            else:
                symptom_rule = {
                    SymptomCollection.SYMPTOM_ID_NAME_KEY: symptom_name_id,
                    SymptomCollection.SYMPTOM_PATTERNS_KEY: set(),
                }

            new_symptom_pattern: List[Tuple[str, str]] = []
            for symptom_token in symptom_doc:
                symptom_pattern_part: Tuple[str, str] = (
                    SYMPTOM_LEMMA_PATTERN_KEY,
                    symptom_token.lemma_,
                )
                new_symptom_pattern.append(symptom_pattern_part)

            symptom_rule[SymptomCollection.SYMPTOM_PATTERNS_KEY].add(
                tuple(new_symptom_pattern)
            )
            symptoms_rules[symptom_name_id] = symptom_rule

    return symptoms_rules


def transform_symptom_rules_for_json(
    symptoms_rules: Dict[str, Dict]
) -> Dict[str, List[Dict]]:
    for symptom_name_id, symptom_rule in symptoms_rules.items():
        new_patterns: List[List[Dict[str, str]]] = []
        for pattern in symptom_rule[SymptomCollection.SYMPTOM_PATTERNS_KEY]:
            new_pattern: List[Dict[str, str]] = []
            for pattern_part in pattern:
                new_pattern_part: Dict[str, str] = {pattern_part[0]: pattern_part[1]}
                new_pattern.append(new_pattern_part)
            new_patterns.append(new_pattern)
        symptom_rule[SymptomCollection.SYMPTOM_PATTERNS_KEY] = new_patterns

    symptoms_rules_for_json: Dict[str, List[Dict]] = {
        SymptomCollection.SYMPTOMS_FILE_TOP_KEY: list(symptoms_rules.values())
    }
    return symptoms_rules_for_json


def save_symptoms_rules_to_json(symptoms_rules_for_json: Dict[str, List[Dict]]) -> None:
    with open(SymptomCollection.SYMPTOMS_FILE_PATH, "w", encoding="utf8") as file:
        json.dump(symptoms_rules_for_json, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
