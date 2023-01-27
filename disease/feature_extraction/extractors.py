from typing import Iterable, Dict, List, Set

from spacy import Language
from spacy.tokens import Doc
from spacy.tokens.token import Token

from disease.base.estimators import BaseTransformer
import spacy
from negspacy.negation import Negex


class SymptomExtractor(BaseTransformer):
    """ Symptom based text vectorization
    """
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

    following_negations = [
        "отклонено",
        "маловероятно",
        "не было",
        "не были",
        "нет"
    ]

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
        "termination": termination
    }

    # TODO: Построить нормальную модель рассматриваемых симптомов
    symptoms = ["температура", "недомогание", "болит голова"]

    def __init__(self):
        self._spacy_lang_model: Language = spacy.load(SymptomExtractor.SPACY_LANG_MODEL_NAME)
        self._negex_model: Negex = Negex(self._spacy_lang_model, "test", SymptomExtractor.russian_termset,
                                         list(), SymptomExtractor.NEGEX_EXTENSION_NAME, list())

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
        entities: Dict[str, bool] = dict(negex_doc.ents)
        if entities:
            for symptom in SymptomExtractor.symptoms:
                if symptom in entities:
                    if entities[symptom]:
                        message_disease[symptom] = 1
                    else:
                        message_disease[symptom] = -1

        # TODO: Support multiwords symptoms
        for token in model_doc:
            if token.lemma_ in SymptomExtractor.symptoms:
                ancestors: Set[Token] = set(token.ancestors)
                children: Set[Token] = set(token.children)
                if any(ancestor.lemma_ in SymptomExtractor.preceding_negations for ancestor in ancestors):
                    message_disease[token.lemma_] = -1
                elif any(child.lemma_ in SymptomExtractor.following_negations for child in children):
                    message_disease[token.lemma_] = -1
                else:
                    message_disease[token.lemma_] = 1

        return list(message_disease.values())
