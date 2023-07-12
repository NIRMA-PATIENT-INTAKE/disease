from enum import Enum


class SymptomStatus(Enum):
    """
    Represents the status of a symptom for a user.

    Attributes:
        YES: The symptom is present.
        NO: The symptom is not present.
        NO_INFO: There is no information about the symptom.
        CONFUSED: The information about the symptom is confusing.
    """

    # When patient has symptom
    YES = 1
    # When patient has not symptom
    NO = 2
    # When we have no info about symptom
    NO_INFO = 3
    # When we confused of info about symptom
    CONFUSED = 4
