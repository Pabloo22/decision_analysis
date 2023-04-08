import numpy as np

from typing import Optional, Union

from src.decision_analysis.decision_making import Criterion


class Dataset:
    """A dataset is a collection of data and criteria.

    We have decided to not make use of the Alternative class here, alternatives will be encoded as integers.
    However, it is possible to associate a name to each one.
    """

    def __init__(self,
                 data: np.ndarray,
                 criteria: list[Criterion],
                 alternative_names: Optional[list[str]] = None):

        if not isinstance(data, np.ndarray):
            raise TypeError("Data must be a numpy array")

        if data.shape[1] != len(criteria):
            raise ValueError("The number of columns in the data must be equal to the number of criteria")

        if alternative_names is not None:
            if len(alternative_names) != data.shape[0]:
                raise ValueError("The number of alternative names must be equal to the number of alternatives")

            self.alternative_names = alternative_names
        else:
            self.alternative_names = [f'a_{i+1}' for i in range(data.shape[0])]

        self.data = data
        self.criteria = criteria
        self.preference_relations = []
        self.indifference_relations = []

    def add_preference(self, alternative_1: Union[int, str], alternative_2: Union[int, str]):
        if isinstance(alternative_1, str):
            alternative_1 = self.alternative_names.index(alternative_1)
        if isinstance(alternative_2, str):
            alternative_2 = self.alternative_names.index(alternative_2)
        self.preference_relations.append((alternative_1, alternative_2))

    def add_indifference(self, alternative_1: Union[int, str], alternative_2: Union[int, str]):
        if isinstance(alternative_1, str):
            alternative_1 = self.alternative_names.index(alternative_1)
        if isinstance(alternative_2, str):
            alternative_2 = self.alternative_names.index(alternative_2)
        self.indifference_relations.append((alternative_1, alternative_2))




