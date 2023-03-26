from dataclasses import dataclass


@dataclass
class Criteria:
    """
    Attributes:
        weight (float): Weight of the criterion.
        criteria_type (int): If the criterion is benefit or cost (1 or -1).
        preference_threshold (float): Preference threshold for the criterion.
        indifference_threshold (float): Indifference threshold for the criterion.
    """
    weight: float
    criteria_type: int
    preference_threshold: float = None
    indifference_threshold: float = None
