from typing import Tuple

from attr import dataclass


@dataclass(eq=False, frozen=True)
class Symptom:
    """
    Represents symptom.

    Args:
        id_name - unique name of symptom
        patterns - list of ScaPy patterns for symptom recognitions
    """

    id_name: str
    patterns: Tuple

    def __eq__(self, other):
        return isinstance(other, Symptom) and self.id_name == other.id_name

    def __hash__(self):
        return hash(self.id_name)
