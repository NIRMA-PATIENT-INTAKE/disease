"""This page documents the classes used for disease and urgency classification.
It includes BaseDiseaseClassifier, DiseaseClassifier, FedotDiseaseClassifier, and UrgencyClassifier classes.
These classes are used to train and predict diseases and their urgency based on the extracted symptoms.
"""
from distool.estimators.classifiers import DiseaseClassifier, FedotDiseaseClassifier

__all__ = ["DiseaseClassifier", "FedotDiseaseClassifier"]
