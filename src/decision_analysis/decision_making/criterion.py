from dataclasses import dataclass
from typing import Optional


@dataclass
class Criterion:
    """A class representing a criterion in the decision analysis.

    Attributes:
        type (int): If the criterion is benefit or cost (1 or -1).
        weight (float): Weight of the criterion. Optional, default 1.
        name (str): Name of the criterion. Optional.
        preference_threshold (float): Preference threshold for the criterion. Optional.
        indifference_threshold (float): Indifference threshold for the criterion. Optional.
        veto_threshold (float): Veto threshold for the criterion. Optional.
        value_function_characteristic_points_location (list[float]): Location of the characteristic points of the value
            function in percentage with respect to the minimum and maximum values of the data for that criterion.
            Optional.
    """
    type: int
    weight: float = 1.
    name: Optional[str] = None
    preference_threshold: Optional[float] = None
    indifference_threshold: Optional[float] = None
    veto_threshold: Optional[float] = None
    value_function_characteristic_points_location: Optional[list[float]] = None

    _default_name = 'g1'

    def __post_init__(self) -> None:
        """Updates the name of the criterion after initialization."""
        if self.name is None:
            self.name = Criterion._default_name
            Criterion._update_default_name()

    @staticmethod
    def _update_default_name() -> None:
        """Updates the default name of the criterion."""
        Criterion._default_name = f'g{int(Criterion._default_name[1:]) + 1}'

    def reset_default_name(self) -> None:
        """Resets the default name of the criterion."""
        self._default_name = 'g1'
