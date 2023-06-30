"""A tool kit for russian diseases procession

.. moduleauthor:: Michil Egorov <egorov_michil@mail.ru>
"""

from distool.estimators.classifiers import BaseDiseaseClassifier
from distool.feature_extraction.dumb_extractor import DumbSymptomExtractor
from distool.feature_extraction.smart_extractor import SmartSymptomExtractor
from distool.interpretation.explainer import SymptomBasedExplainer
