"""Welcome to the documentation of distool, a Python package for disease diagnosis and urgency estimation
based on the symptoms extracted from the text.
 This package provides a comprehensive set of tools for symptom extraction,
 disease classification, and result interpretation.
 The documentation is organized into several sections,
  each dedicated to a specific part of the package.
  Explore the sections to understand how to use distool effectively for your disease diagnosis tasks.

.. moduleauthor:: Michil Egorov <egorov_michil@mail.ru>
"""

from distool.estimators.classifiers import BaseDiseaseClassifier
from distool.feature_extraction.dumb_extractor import DumbSymptomExtractor
from distool.feature_extraction.smart_extractor import SmartSymptomExtractor
from distool.interpretation.explainer import SymptomBasedExplainer
