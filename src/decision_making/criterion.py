from dataclasses import dataclass
from typing import Optional


@dataclass
class Criterion:
    """
    Args:
        criteria_type (int): If the criterion is benefit or cost (1 or -1).
        weight (float): Weight of the criterion. Optional, default 1.
        name (str): Name of the criterion. Optional.
        preference_threshold (float): Preference threshold for the criterion. Optional.
        indifference_threshold (float): Indifference threshold for the criterion. Optional.

    """
    criteria_type: int
    weight: float = 1.
    name: str = ""
    preference_threshold: Optional[float] = None
    indifference_threshold: Optional[float] = None
