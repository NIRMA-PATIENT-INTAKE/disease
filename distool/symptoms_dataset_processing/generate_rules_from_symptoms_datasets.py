import json

import numpy as np
import pandas as pd
import spacy

from distool import SymptomExtractor
from distool.feature_extraction.symptom_collection import (
    SYMPTOM_ID_NAME_KEY,
    SYMPTOM_PATTERNS_KEY,
    SYMPTOMS_FILE_PATH,
    SYMPTOMS_FILE_TOP_KEY,
)

PATH_TO_SYMPTOM_DATASET = "../data/symptoms_datasets/symptoms_dataset.csv"
SYMPTOM_LEMMA_PATTERN_KEY = "LEMMA"


def main():
    symptoms_list = load_and_preprocess_symptoms_list()
    symptoms_rules = {}
    symptoms_rules = generate_patterns_from_symptoms_list(symptoms_rules, symptoms_list)
    prepared_symptoms_rules = prepare_symptom_rules_to_json(symptoms_rules)
    save_symptoms_rules_to_json(prepared_symptoms_rules)
    print(symptoms_rules)


def load_and_preprocess_symptoms_list():
    df = pd.read_csv(PATH_TO_SYMPTOM_DATASET)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df["Симптомы"].replace("", np.NaN, inplace=True)
    df.dropna(inplace=True)

    df["Симптомы"] = df["Симптомы"].map(lambda x: x.split("; "))
    return list(df["Симптомы"])


def generate_patterns_from_symptoms_list(symptoms_rules, symptoms_list):
    spacy_model = spacy.load(SymptomExtractor.SPACY_LANG_MODEL_NAME)

    for symptoms_item in symptoms_list:
        for symptom in symptoms_item:
            preprocessed_symptom = symptom.strip()
            symptoms_doc = spacy_model(preprocessed_symptom)
            symptom_lemmas = [token.lemma_ for token in symptoms_doc]
            symptom_name_id = " ".join(symptom_lemmas)

            if symptom_name_id in symptoms_rules:
                symptom_rule = symptoms_rules[symptom_name_id]
            else:
                symptom_rule = {
                    SYMPTOM_ID_NAME_KEY: symptom_name_id,
                    SYMPTOM_PATTERNS_KEY: set(),
                }

            new_symptom_pattern = []
            for token in symptoms_doc:
                symptom_pattern_part = (SYMPTOM_LEMMA_PATTERN_KEY, token.lemma_)
                new_symptom_pattern.append(symptom_pattern_part)

            symptom_rule[SYMPTOM_PATTERNS_KEY].add(tuple(new_symptom_pattern))
            symptoms_rules[symptom_name_id] = symptom_rule

    return symptoms_rules


def prepare_symptom_rules_to_json(symptoms_rules):
    for symptom_name_id, symptom_rule in symptoms_rules.items():
        new_patterns = []
        for pattern in symptom_rule[SYMPTOM_PATTERNS_KEY]:
            new_pattern = []
            for pattern_part in pattern:
                new_pattern_part = {pattern_part[0]: pattern_part[1]}
                new_pattern.append(new_pattern_part)
            new_patterns.append(new_pattern)
        symptom_rule[SYMPTOM_PATTERNS_KEY] = new_patterns

    symptoms_rules_for_json = {SYMPTOMS_FILE_TOP_KEY: list(symptoms_rules.values())}
    return symptoms_rules_for_json


def save_symptoms_rules_to_json(symptoms_rules_for_json):
    with open(SYMPTOMS_FILE_PATH, "w", encoding="utf8") as file:
        json.dump(symptoms_rules_for_json, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
