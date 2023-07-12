from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, List, Union

import numpy as np
import spacy
from negspacy.negation import Negex
from spacy import Language
from spacy.tokens import Doc

from distool.base.estimators import BaseTransformer
from distool.feature_extraction.anamnesis import Anamnesis
from distool.feature_extraction.symptom_collection import SymptomCollection


class SmartSymptomExtractor(BaseTransformer):
    """
    A symptom extractor that uses SpaCy and Negex for symptom recognition and negation handling.

    This class is a specific implementation of the BaseTransformer.

    Attributes:
        SPACY_LANG_MODEL_NAME: The name of the SpaCy language model to use.
        NEGEX_EXTENSION_NAME: The name of the Negex extension.
        pseudo_negations: A list of phrases that are considered pseudo negations.
        preceding_negations: A list of phrases that are considered preceding negations.
        following_negations: A list of phrases that are considered following negations.
        termination: A list of phrases that are considered termination phrases.
        russian_termset: A dictionary containing the Russian termset for Negex.
    """

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

    def __init__(self) -> None:
        """Initializes a new instance of the SmartSymptomExtractor class."""
        self._spacy_lang_model: Language = spacy.load(
            SmartSymptomExtractor.SPACY_LANG_MODEL_NAME,
            disable=["tok2vec", "morphologizer", "attribute_ruler", "ner"],
        )

        ruler = self._spacy_lang_model.add_pipe(
            "entity_ruler", config={"validate": True}
        )
        ruler.add_patterns(SymptomCollection.get_spacy_model_patterns())

        negex_config = {
            "neg_termset": SmartSymptomExtractor.russian_termset,
            "ent_types": [SymptomCollection.SYMPTOM_ENTITY_LABEL_VALUE],
            "extension_name": SmartSymptomExtractor.NEGEX_EXTENSION_NAME,
            "chunk_prefix": [],
        }
        self._spacy_lang_model.add_pipe(
            factory_name="negex", name="negex", last=True, config=negex_config
        )

    def fit(self, x: Iterable[str]):
        """Fits the transformer according to the given training data.

        Args:
            x: Iterable over raw text data.
        """
        pass

    def _transform(self, message: str) -> Anamnesis:
        """Transforms a single message into an Anamnesis instance.

        Args:
            message: A string representing a user message.

        Returns:
            An Anamnesis instance.
        """
        model_doc: Doc = self._spacy_lang_model(message)
        return SmartSymptomExtractor._transform_inner(model_doc)

    def _transform_inner(doc: Doc) -> Anamnesis:
        anamnesis: Anamnesis = Anamnesis()
        symptom_entities = [
            entity
            for entity in doc.ents
            if entity.label_ == SymptomCollection.SYMPTOM_ENTITY_LABEL_VALUE
        ]

        for entity in symptom_entities:
            anamnesis.update_symptom_status_by_entity(entity)

        return anamnesis

    def transform(
        self, messages: List[str], as_anamnesis: bool = False
    ) -> Union[List[Anamnesis], np.array]:
        """Transforms a list of messages into a list of Anamnesis instances or a numpy array.

        Args:
            messages: A list of strings representing user messages.
            as_anamnesis: A boolean indicating whether to return the result as a list of Anamnesis instances. If False, the result is returned as a numpy array.

        Returns:
            A list of Anamnesis instances or a numpy array.
        """
        model_docs: List[Doc] = self._spacy_lang_model.pipe(messages)
        with ThreadPoolExecutor() as executor:
            features = list(
                executor.map(SmartSymptomExtractor._transform_inner, model_docs)
            )

        if not as_anamnesis:
            features = np.array([anamnesis.get_marks() for anamnesis in features])

        return features
