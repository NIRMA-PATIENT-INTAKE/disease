from typing import Dict, List

from spacy.tokens import Span

from disease.feature_extraction.symptom import Symptom
from disease.feature_extraction.symptom_collection import NAME_TO_SYMPTOM, SYMPTOMS
from disease.feature_extraction.symptom_status import SymptomStatus


def _create_symptoms_marks() -> Dict[Symptom, SymptomStatus]:
    return dict.fromkeys(SYMPTOMS, SymptomStatus.NO_INFO)


class Anamnesis:
    """
    Represents symptoms and it's statuses extracted from user messages.
    """

    def __init__(self):
        self._symptoms_marks: Dict[Symptom, SymptomStatus] = _create_symptoms_marks()

    def update_symptom_status_by_entity(self, entity: Span):
        entity_lemma = entity.lemma_
        symptom = NAME_TO_SYMPTOM.get(entity_lemma)
        if symptom:
            old_value = self._symptoms_marks[symptom]
            if (old_value == SymptomStatus.YES and entity._.negex == False) or (
                old_value == SymptomStatus.NO and entity._.negex == True
            ):
                self._symptoms_marks[symptom] = SymptomStatus.CONFUSED
            elif old_value == SymptomStatus.NO_INFO and entity._.negex == False:
                self._symptoms_marks[symptom] = SymptomStatus.YES
            elif old_value == SymptomStatus.NO_INFO and entity._.negex == True:
                self._symptoms_marks[symptom] = SymptomStatus.NO

    def update_symptoms_statuses_by_new_anamnesis(self, new_anamnesis: Span):
        if not isinstance(new_anamnesis, Anamnesis):
            raise ValueError(
                "Other anamnesis should be Anamnesis, but it is "
                + new_anamnesis.__class__
            )

        for (key, new_value) in new_anamnesis._symptoms_marks.items():
            old_value = self._symptoms_marks[key]
            if (old_value == SymptomStatus.YES and new_value == SymptomStatus.NO) or (
                old_value == SymptomStatus.NO and new_value == SymptomStatus.YES
            ):
                self._symptoms_marks[key] = SymptomStatus.CONFUSED
            elif (
                old_value == SymptomStatus.NO_INFO
                or old_value == SymptomStatus.CONFUSED
            ) and new_value != SymptomStatus.NO_INFO:
                self._symptoms_marks[key] = new_value
        return self

    def get_symptom_status(self, symptom_name: str) -> SymptomStatus:
        symptom = NAME_TO_SYMPTOM.get(symptom_name)
        return self._symptoms_marks.get(symptom)

    def __len__(self):
        return len(self._symptoms_marks)

    def reset(self):
        self._symptoms_marks: Dict[Symptom, SymptomStatus] = _create_symptoms_marks()

    def get_marks(self) -> List[SymptomStatus]:
        return list(self._symptoms_marks.values())
