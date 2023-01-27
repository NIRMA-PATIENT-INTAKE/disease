import unittest

import spacy

from disease.feature_extraction import SymptomExtractor
from spacy.lang.ru import Russian


class SymptomExtractorTest(unittest.TestCase):

    def test_transform_1(self):
        message = "У меня нет температуры, но есть недомогание и не болит голова."
        disease_vector = SymptomExtractor().transform(message)
        self.assertEqual(disease_vector,
                         [-1, -1, 0],
                         "Extracted disease vector is incorrect")