from dataclasses import dataclass
import functools
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
        value_function_characteristic_points (list[tuple[float, float]]): Characteristic points of the value function
            defined as a list of tuples (x, y) where `x` is the value of the criterion and y is the value of the value
            function. Linear interpolation is used to calculate the value of the value function at intermediate values.
    """
    type: int
    weight: float = 1.
    name: Optional[str] = None
    preference_threshold: Optional[float] = None
    indifference_threshold: Optional[float] = None
    veto_threshold: Optional[float] = None
    value_function_characteristic_points_location: Optional[list[float]] = None
    value_function_characteristic_points: Optional[list[tuple[float, float]]] = None

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

    @staticmethod
    def linear_interpolation(x: float, x1: float, x2: float, y1: float, y2: float) -> float:
        """Returns the linear interpolation at the point x."""
        return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

    @functools.cache
    def value_function(self, value: float) -> float:
        """Returns the value of the value function at the given value."""
        if self.value_function_characteristic_points is None:
            raise ValueError("The value function characteristic points must be defined")

        sorted_characteristic_points = sorted(self.value_function_characteristic_points)
        for i, (x, y) in enumerate(self.value_function_characteristic_points):
            if value >= x:
                continue
            if i == 0:
                return y

            previous_x, previous_y = self.value_function_characteristic_points[i - 1]
            return self.linear_interpolation(value, previous_x, x, previous_y, y)
