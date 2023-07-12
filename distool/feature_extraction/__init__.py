"""
This page documents the classes used for extracting features from the text. It includes Anamnesis,
DumbSymptomExtractor, SmartSymptomExtractor, Symptom, SymptomCollection, and SymptomStatus classes.
These classes are used to extract symptoms from the text, represent them in a structured way,
and transform them into a format suitable for machine learning models.
"""

from distool.feature_extraction.dumb_extractor import DumbSymptomExtractor
from distool.feature_extraction.smart_extractor import SmartSymptomExtractor

__all__ = ["SmartSymptomExtractor", "DumbSymptomExtractor"]
