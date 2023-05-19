from typing import Union

import pandas as pd

from distool.feature_extraction.symptom_status import SymptomStatus

RESULT_LABELS = {
    "VALID": [
        (SymptomStatus.YES, SymptomStatus.YES),
        (SymptomStatus.NO, SymptomStatus.NO),
    ],
    "INVALID": [
        (SymptomStatus.NO, SymptomStatus.YES),
        (SymptomStatus.YES, SymptomStatus.NO),
    ],
    "VALIDATE_EXTRACTOR": [
        (SymptomStatus.YES, SymptomStatus.NO_INFO),
        (SymptomStatus.NO, SymptomStatus.NO_INFO),
    ],
    "VALIDATE_MARKER": [
        (SymptomStatus.NO_INFO, SymptomStatus.YES),
        (SymptomStatus.NO_INFO, SymptomStatus.NO),
    ],
    "UNDEFINED": [(SymptomStatus.NO_INFO, SymptomStatus.NO_INFO)],
}
COMPUTATION_COLUMN = list(set(RESULT_LABELS.keys()).difference(["UNDEFINED"]))


def validate_extractor(
    df_showcase: pd.DataFrame, symptom: str, as_tuple: bool = True
) -> Union[tuple, dict]:
    marked_symptom = df_showcase["marked_" + symptom]
    extracted_symptom = df_showcase["extractor_" + symptom]

    results = {}

    for key, compares in RESULT_LABELS.items():
        results[key] = sum(
            [
                ((marked_symptom == m_l.value) & (extracted_symptom == e_l.value)).sum()
                for m_l, e_l in compares
            ]
        )

    if as_tuple:
        return tuple(results.values())

    return results


def get_label_partion(df_in_symptoms: pd.DataFrame, label: str) -> float:
    label_columns = list(set(RESULT_LABELS.keys()).difference(["UNDEFINED"]))

    return df_in_symptoms[label].sum() / df_in_symptoms[label_columns].sum().sum()


def compute_metrics(df_showcase: pd.DataFrame, print_metrics: bool = True) -> None:
    symptoms = (
        df_showcase.columns[df_showcase.columns.str.startswith("marked")]
        .str.split("marked_")
        .str[1]
    )

    symptom2labels = {}

    for s in symptoms:
        symptom2labels[s] = validate_extractor(df_showcase, s, as_tuple=True)

    symptom_labels_df = pd.DataFrame(symptom2labels, index=list(RESULT_LABELS.keys())).T
    in_symptoms_df = symptom_labels_df[
        symptom_labels_df["UNDEFINED"] != symptom_labels_df.shape[0]
    ].sort_values("INVALID")

    if print_metrics:
        for label in COMPUTATION_COLUMN:
            print(label, "-", round(get_label_partion(in_symptoms_df, label), 2))

    return in_symptoms_df


if __name__ == "__main__":
    df_showcase = pd.read_csv("../../data/showcase.csv")
    compute_metrics(df_showcase)
