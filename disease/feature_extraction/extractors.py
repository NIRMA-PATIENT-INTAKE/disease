from typing import Iterable

import spacy
from negspacy.negation import Negex
from spacy import Language
from spacy.tokens import Doc

from disease.base.estimators import BaseTransformer
from disease.feature_extraction.anamnesis import Anamnesis
from disease.feature_extraction.symptom_collection import (
    SYMPTOM_ENTITY_LABEL_VALUE,
    get_symptoms_patterns,
)


class SymptomExtractor(BaseTransformer):
    """Symptom based text vectorization"""

    SPACY_LANG_MODEL_NAME: str = "ru_core_news_md"
    NEGEX_EXTENSION_NAME: str = "negex"

    # Following list of words is auto-translated list of words from negspacy/termsets.py
    pseudo_negations = [
        "не дальше",
        "не в состоянии быть",
        "не уверен, что",
        "не уверен, что",
        "не обязательно",
        "без дальнейшего",
        "без затруднений",
        "без дальнейшего",
        "возможно, нет",
        "не только",
        "без увеличения",
        "без значительных изменений",
        "без изменений",
        "нет определенного изменения",
        "не распространяться",
        "не вызывать",
        "граммотрицательный",
        "не исключено",
        "не слить",
        "нет подозрительных изменений",
        "нет изменений в интервале",
        "нет значительного изменения интервала",
    ]

    preceding_negations = [
        "отсутствие",
        "отклонено",
        "отказано",
        "отрицает",
        "отрицающий",
        "нет признаков",
        "не",
        "не демонстрируют",
        "симптомы нетипичные",
        "сомневаюсь",
        "отрицательно",
        "нет",
        "против",
        "без",
        "не делает",
        "не было",
        "не является",
        "не являются",
        "не могу",
        "не мог",
        "никогда",
    ]

    following_negations = ["отклонено", "маловероятно", "не было", "не были", "нет"]

    termination = [
        "хотя",
        "помимо",
        "как есть",
        "помимо",
        "но",
        "кроме",
        "однако",
        "вовлекающий",
        "тем не менее",
        "все еще",
        "хотя",
        "который",
        "пока",
    ]

    russian_termset = {
        "pseudo_negations": pseudo_negations,
        "preceding_negations": preceding_negations,
        "following_negations": following_negations,
        "termination": termination,
    }

    def __init__(self):
        self._spacy_lang_model: Language = spacy.load(
            SymptomExtractor.SPACY_LANG_MODEL_NAME
        )
        ruler = self._spacy_lang_model.add_pipe("entity_ruler")

        ruler.add_patterns(get_symptoms_patterns())
        self._negex_model: Negex = Negex(
            nlp=self._spacy_lang_model,
            name="negotiation",
            neg_termset=SymptomExtractor.russian_termset,
            ent_types=[SYMPTOM_ENTITY_LABEL_VALUE],
            extension_name=SymptomExtractor.NEGEX_EXTENSION_NAME,
            chunk_prefix=[],
        )

    def fit(self, x: Iterable[str]):
        pass

    def transform(self, message: str) -> Anamnesis:
        model_doc: Doc = self._spacy_lang_model(message)
        negex_doc: Doc = self._negex_model(model_doc)

        anamnesis: Anamnesis = Anamnesis()
        symptom_entities = [
            entity
            for entity in negex_doc.ents
            if entity.label_ == SYMPTOM_ENTITY_LABEL_VALUE
        ]

        for entity in symptom_entities:
            anamnesis.update_symptom_status_by_entity(entity)

        return anamnesis
