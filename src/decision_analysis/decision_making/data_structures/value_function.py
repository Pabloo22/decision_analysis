import pulp
from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class ValueFunction:
    """A class representing a value function.

    Attributes:
        characteristic_points_locations (list[float]): The x coordinates of the characteristic points of the value
            function. This list must contain at least 2 numbers. It is sorted automatically.
        characteristic_points_values (list[float]): The y coordinates of the characteristic points of the value
            function. This list must contain the same number of elements as the characteristic_points_locations list
            before calling the __call__ method.
    """

    def __init__(self,
                 characteristic_points_locations: list[float],
                 characteristic_points_values: Optional[list[float]] = None):
        self.characteristic_points_locations = characteristic_points_locations
        self.characteristic_points_values = characteristic_points_values

    @property
    def n_break_points(self) -> int:
        """Returns the number of break points of the value function."""
        return len(self.characteristic_points_locations) - 2

    @property
    def characteristic_points_locations(self) -> list[float]:
        """Returns the y coordinates of the characteristic points of the value function."""
        return self._characteristic_points_locations

    @characteristic_points_locations.setter
    def characteristic_points_locations(self, characteristic_points_locations: list[float]):
        """Sets the y coordinates of the characteristic points of the value function."""
        if len(characteristic_points_locations) < 2:
            raise ValueError("The value function must have at least 2 characteristic points")
        self._characteristic_points_locations = sorted(characteristic_points_locations)

    @property
    def characteristic_points_values(self) -> list[float]:
        """Returns the y coordinates of the characteristic points of the value function."""
        return self._characteristic_points_values

    @staticmethod
    def is_monotonic(characteristic_points_values: list[float]) -> bool:
        """Checks if the value function is monotonic."""

        ascending = characteristic_points_values[0] < characteristic_points_values[1]
        for i in range(1, len(characteristic_points_values) - 1):
            if ascending and characteristic_points_values[i] > characteristic_points_values[i + 1]:
                return False
            if not ascending and characteristic_points_values[i] < characteristic_points_values[i + 1]:
                return False

        return True

    @characteristic_points_values.setter
    def characteristic_points_values(self, characteristic_points_values: Union[list[float]]):
        """Sets the y coordinates of the characteristic points of the value function."""
        if characteristic_points_values is None:
            self._characteristic_points_values = None
            return

        if len(characteristic_points_values) != len(self.characteristic_points_locations):
            raise ValueError("The number of characteristic points values must be equal to the number of "
                             f"characteristic points locations. Got {len(characteristic_points_values)} and "
                             f"{len(self.characteristic_points_locations)} respectively.")

        if not self.is_monotonic(characteristic_points_values):
            raise ValueError("The value function must be monotonic")

        self._characteristic_points_values = characteristic_points_values

    @staticmethod
    def linear_interpolation(x: float,
                             x1: float,
                             x2: float,
                             y1: Union[float, pulp.LpVariable],
                             y2: Union[float, pulp.LpVariable]) -> Union[float, pulp.LpVariable]:
        """Returns the linear interpolation at the point x."""
        return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

    def __call__(self, value: float):
        """Returns the value of the value function at the given value."""
        if self.characteristic_points_values is None:
            raise ValueError("The characteristic points values must be set before calling the value function")

        characteristic_points = list(zip(self.characteristic_points_locations, self.characteristic_points_values))

        for i, (x, y) in enumerate(characteristic_points):
            if value > x:
                continue
            if i == 0:
                return y

            previous_x, previous_y = characteristic_points[i - 1]
            return self.linear_interpolation(value, previous_x, x, previous_y, y)

        return self.characteristic_points_values[-1]
