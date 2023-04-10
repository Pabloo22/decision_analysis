from typing import Optional
import numpy as np
from dataclasses import dataclass

from .criterion import Criterion


@dataclass
class Dataset:
    """A dataset is a collection of data and criteria.

    We have decided to not make use of the Alternative class here, alternatives will be encoded as integers.
    However, it is possible to associate a name to each one.

    Attributes:
        data: A numpy array containing the data.
        criteria: A list of criteria.
        alternative_names: A list of alternative names. If not given, the alternatives will be named
            as a_1, a_2, a_3, ...
    """
    data: np.ndarray
    criteria: list[Criterion]
    alternative_names: Optional[list[str]] = None

    def __post_init__(self):
        if not isinstance(self.data, np.ndarray):
            raise TypeError("Data must be a numpy array")

        if self.data.shape[1] != len(self.criteria):
            raise ValueError("The number of columns in the data must be equal to the number of criteria")

        if self.alternative_names is not None:
            if len(self.alternative_names) != self.data.shape[0]:
                raise ValueError("The number of alternative names must be equal to the number of alternatives")
        else:
            self.alternative_names = [f'a_{i + 1}' for i in range(self.data.shape[0])]
