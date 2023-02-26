from enum import Enum


class SymptomStatus(Enum):
    """
    Represents the status of symptom for the user.
    """

    # When patient has symptom
    YES = 1
    # When patient has not symptom
    NO = 2
    # When we have no info about symptom
    NO_INFO = 3
    # When we confused of info about symptom
    CONFUSED = 4
