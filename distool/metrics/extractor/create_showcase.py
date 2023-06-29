import re
from pathlib import Path
from typing import List, Optional, Set, Tuple

import pandas as pd
import pymorphy2
from pandas import DataFrame

from distool.base.estimators import BaseTransformer
from distool.feature_extraction import DumbSymptomExtractor, SmartSymptomExtractor
from distool.feature_extraction.anamnesis import Anamnesis
from distool.feature_extraction.symptom_collection import SymptomCollection
from distool.feature_extraction.symptom_status import SymptomStatus

BASE_DIR = Path(__file__).parent.parent

PATH_TO_MARKED_DF = BASE_DIR / "../data/patients_symptoms_marked.csv"
PATH_TO_PATIENT_CASE_DF = BASE_DIR / "../data/patient_case.csv"
PATH_TO_SMART_SHOWCASE_DF = BASE_DIR / "../data/smart_showcase.csv"
PATH_TO_DUMB_SHOWCASE_DF = BASE_DIR / "../data/dumb_showcase.csv"

morph = pymorphy2.MorphAnalyzer()


def create_extractor_showcase(
    extractor: BaseTransformer, path_to_save_showcase: Optional[Path]
) -> DataFrame:
    marked_df = load_marked_df()
    marked_symptoms = get_marked_symptoms(marked_df)
    marked_symptom_ids = get_marked_symptom_ids(marked_symptoms)
    existed_symptoms_id = get_existed_symptoms_id()
    unknown_symptoms_ids = marked_symptom_ids.difference(existed_symptoms_id)
    print("Unknown symptoms in marked df:")
    print(unknown_symptoms_ids)

    # Need to convert marked dataset to format:
    # `ID - <list of marked symptom column> - <list of extractor symptom column>`
    df_showcase = pd.DataFrame(list(marked_df.index), columns=["case_id"])
    df_showcase = add_case_text_column_to_showcase(df_showcase)

    marked_symptom_column_labels = get_marked_symptom_column_labels(existed_symptoms_id)
    column_symptoms_marks = get_column_symptom_marks(
        marked_df, marked_symptom_column_labels
    )
    df_showcase.loc[:, marked_symptom_column_labels] = column_symptoms_marks

    transformed_symptoms = []
    for case in df_showcase.case:
        anamnesis = Anamnesis()
        anamnesis_list = extractor.transform(case, True)
        for new_anamnesis in anamnesis_list:
            anamnesis = anamnesis.update_symptoms_statuses_by_new_anamnesis(
                new_anamnesis
            )
        transformed_symptoms.append(anamnesis.get_marks(True))

    extractor_symptom_column_labels = get_extractor_symptom_column_labels(
        existed_symptoms_id
    )
    df_showcase.loc[:, extractor_symptom_column_labels] = transformed_symptoms

    print(df_showcase.sum())
    print(df_showcase.shape)
    if path_to_save_showcase:
        df_showcase.to_csv(path_to_save_showcase, index=False)

    return df_showcase


def load_marked_df() -> pd.DataFrame:
    marked_df = pd.read_csv(PATH_TO_MARKED_DF).set_index("Идентификатор")
    print("Marked df:")
    print(marked_df.head())
    return marked_df


def get_marked_symptoms(marked_df) -> Set[str]:
    marked_symptoms = set()
    for row_tuple in marked_df.itertuples(index=False, name=None):
        filtered_row = list(
            filter(lambda x: not pd.isna(x) and isinstance(x, str), row_tuple)
        )
        marked_symptoms.update(filtered_row)
    return marked_symptoms


def get_marked_symptom_ids(marked_symptoms: Set[str]) -> Set[str]:
    marked_symptom_ids = set()
    for marked_symptom in marked_symptoms:
        marked_symptom_ids.add(try_to_transform_symptom_to_id(marked_symptom))
    return marked_symptom_ids


def try_to_transform_symptom_to_id(symptom_str):
    symptom_id = symptom_str.lower()
    symptom_id = re.sub(r"(\<(/?[^>]+)>)", " ", symptom_id)
    symptom_id = symptom_id.replace("[^\w\s]", " ")
    symptom_id = re.sub(r"\d+", " ", symptom_id)
    symptom_id = lemmatize(symptom_id)
    symptom_id = " ".join(symptom_id)
    return symptom_id


def lemmatize(text):
    words = text.split()  # разбиваем текст на слова
    res = list()
    for word in words:
        p = morph.parse(word)[0]
        res.append(p.normal_form)

    return res


def get_existed_symptoms_id() -> List[str]:
    existed_symptoms = SymptomCollection.get_symptoms()
    existed_symptoms_id = list(map(lambda x: x.id_name, existed_symptoms))
    return existed_symptoms_id


def add_case_text_column_to_showcase(df_showcase: pd.DataFrame) -> pd.DataFrame:
    patient_df = pd.read_csv(PATH_TO_PATIENT_CASE_DF).dropna()
    patient_df.rename(columns={"id": "case_id"}, inplace=True)
    preprocess_case(patient_df)
    patient_df = patient_df.drop(columns=["name", "url"])
    return df_showcase.join(patient_df.set_index("case_id"), on="case_id")


def preprocess_case(df):
    df.case = df.case.str.lower()
    df.case = df.case.apply(lambda x: re.sub(r"(\<(/?[^>]+)>)", " ", x).strip())
    df.case = df.case.str.replace("[^\w\s.!?]", " ")
    df.case = df.case.str.replace("[.!?]", ".")
    df.case = df.case.str.replace("\s*\.\s*", ".")
    df.case = df.case.apply(lambda x: re.sub(r"\d+", " ", x))
    df.case = df.case.str.replace("\s+", " ")
    df.case = df.case.apply(lambda x: x.split("."))
    df.case = df.case.apply(lambda x: [lemmatize(i) for i in x])
    df.case = df.case.apply(lambda x: [" ".join(i) for i in x])


def get_marked_symptom_column_labels(existed_symptoms_id: List[str]) -> List[str]:
    marked_column_existed_symptoms_id = []
    for id in existed_symptoms_id:
        column_id = symptom_id_to_marked_column_name(id)
        marked_column_existed_symptoms_id.append(column_id)
    return marked_column_existed_symptoms_id


def symptom_id_to_marked_column_name(id):
    return "marked_" + id.replace(" ", "_")


def get_column_symptom_marks(
    marked_df: pd.DataFrame, marked_symptom_column_labels: List[str]
) -> List[List[int]]:
    zipped_rows = get_zipped_row_marked_df(marked_df)

    column_symptoms_marks = []
    for zipped_row in zipped_rows:
        column_symptoms_marks_row = [
            SymptomStatus.NO_INFO.value for i in marked_symptom_column_labels
        ]  # 3 - No Info status
        for (id, mark) in zipped_row:
            marked_column_name = symptom_id_to_marked_column_name(id)
            print(marked_column_name)
            if marked_column_name in marked_symptom_column_labels:
                print(mark)
                if int(mark) == 1:
                    column_symptoms_marks_row[
                        marked_symptom_column_labels.index(marked_column_name)
                    ] = SymptomStatus.YES.value
                elif int(mark) == -1:
                    column_symptoms_marks_row[
                        marked_symptom_column_labels.index(marked_column_name)
                    ] = SymptomStatus.NO.value
        column_symptoms_marks.append(column_symptoms_marks_row)
    return column_symptoms_marks


def get_zipped_row_marked_df(marked_df: pd.DataFrame) -> List[List[Tuple[str, int]]]:
    zipped_rows = []
    for row_tuple in marked_df.itertuples(index=False, name=None):
        filtered_row = list(filter(lambda x: not pd.isna(x), row_tuple))
        row_iter = iter(filtered_row)
        zipped_row = list(zip(row_iter, row_iter))
        zipped_rows.append(zipped_row)
    return zipped_rows


def get_extractor_symptom_column_labels(existed_symptoms_id: List[str]) -> List[str]:
    extractor_column_existed_symptoms_id = []
    for id in existed_symptoms_id:
        column_id = symptom_id_to_extractor_column_name(id)
        extractor_column_existed_symptoms_id.append(column_id)
    return extractor_column_existed_symptoms_id


def symptom_id_to_extractor_column_name(id):
    return "extractor_" + id.replace(" ", "_")


if __name__ == "__main__":
    create_extractor_showcase(SmartSymptomExtractor(), PATH_TO_SMART_SHOWCASE_DF)
    create_extractor_showcase(DumbSymptomExtractor(), PATH_TO_DUMB_SHOWCASE_DF)
