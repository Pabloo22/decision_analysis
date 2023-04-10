from dataclasses import dataclass
from typing import Optional, Union

from .value_function import ValueFunction


@dataclass
class Criterion:
    """A class representing a criterion in the decision analysis.

    Attributes:
        type (int): Type of the criterion. Can be either gain or cost. Their values are 1 and -1, respectively. Default
            is 1.
        weight (float): Weight of the criterion. Optional, default 1.
        name (str): Name of the criterion. Optional.
        preference_threshold (float): Preference threshold for the criterion. Optional.
        indifference_threshold (float): Indifference threshold for the criterion. Optional.
        veto_threshold (float): Veto threshold for the criterion. Optional.
    """
    type: Union[int, str]
    weight: float = 1.
    name: Optional[str] = None
    preference_threshold: Optional[float] = None
    indifference_threshold: Optional[float] = None
    veto_threshold: Optional[float] = None
    value_function: Optional[ValueFunction] = None

    _default_name = 'g1'

    def __post_init__(self) -> None:
        """Updates the name of the criterion after initialization."""
        if self.name is None:
            self.name = Criterion._default_name
            Criterion._update_default_name()

        if isinstance(self.type, str):
            mapper = {'gain': 1, 'cost': -1}
            if self.type not in mapper:
                raise ValueError(f'Invalid criterion type: {self.type}')
            self.type = mapper[self.type]

    def is_gain(self) -> bool:
        """Returns True if the criterion is a gain, False otherwise."""
        return self.type == 1

    @staticmethod
    def _update_default_name() -> None:
        """Updates the default name of the criterion."""
        Criterion._default_name = f'g{int(Criterion._default_name[1:]) + 1}'

    def reset_default_name(self) -> None:
        """Resets the default name of the criterion."""
        self._default_name = 'g1'
