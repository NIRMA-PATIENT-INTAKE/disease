from typing import Dict, Iterable, List

import spacy
from negspacy.negation import Negex
from spacy import Language
from spacy.tokens import Doc

from disease.base.estimators import BaseTransformer

NEGEX_ENTITY_LABEL: str = "SYMPTOM"


class SymptomExtractor(BaseTransformer):
    """Symptom based text vectorization"""

    SPACY_LANG_MODEL_NAME: str = "ru_core_news_md"
    NEGEX_EXTENSION_NAME: str = "negex"

    # Следующие списки слов - автоматически переведенные списки слов из negspacy/termsets.py
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

    # TODO: Построить нормальную модель рассматриваемых симптомов
    symptoms = ["температура", "недомогание", "болеть голова"]
    patterns = [
        {"label": NEGEX_ENTITY_LABEL, "pattern": [{"LEMMA": "температура"}]},
        {"label": NEGEX_ENTITY_LABEL, "pattern": [{"LEMMA": "недомогание"}]},
        {
            "label": NEGEX_ENTITY_LABEL,
            "pattern": [{"LEMMA": "болеть"}, {"LEMMA": "голова"}],
        },
    ]

    def __init__(self):
        self._spacy_lang_model: Language = spacy.load(
            SymptomExtractor.SPACY_LANG_MODEL_NAME
        )
        ruler = self._spacy_lang_model.add_pipe("entity_ruler")

        ruler.add_patterns(SymptomExtractor.patterns)
        self._negex_model: Negex = Negex(
            nlp=self._spacy_lang_model,
            name="negotiation",
            neg_termset=SymptomExtractor.russian_termset,
            ent_types=[NEGEX_ENTITY_LABEL],
            extension_name=SymptomExtractor.NEGEX_EXTENSION_NAME,
            chunk_prefix=[],
        )

    def fit(self, x: Iterable[str]):
        pass

    def transform(self, message: str) -> List[int]:
        model_doc: Doc = self._spacy_lang_model(message)
        # TODO: Может быть можно переписать это так, чтобы учитывались не ents,
        #  а симптомы - https://spacy.io/api/entityruler
        negex_doc: Doc = self._negex_model(model_doc)

        # TODO: Что будет если для одного сиптома сначала ставим -1, а потом 1.
        #  -> Нужно ставить 0 (типо не уверены), а для этого нужен отдельный датакласс
        message_disease: Dict[str, int] = dict.fromkeys(SymptomExtractor.symptoms, 0)

        # Проверяем есть ли в entities симптомы и проставляем соответствующие метки
        entities: Dict[str, bool] = {
            entity.lemma_: entity._.negex for entity in negex_doc.ents
        }
        if entities:
            for symptom in SymptomExtractor.symptoms:
                if symptom in entities:
                    if entities[symptom]:
                        message_disease[symptom] = -1
                    else:
                        message_disease[symptom] = 1

        return list(message_disease.values())
