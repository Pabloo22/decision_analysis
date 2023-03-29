from __future__ import annotations

import pandas as pd
from typing import Optional


class Alternative:
    """A class representing an alternative in the decision analysis.

    Attributes:
        name (str): The name of the alternative.
        evaluations (Dict[str, float]): A dictionary containing the evaluations of the alternative on each criterion.
    """
    _default_name = 'a1'

    def __init__(self, evaluations: dict[str, float], name: Optional[str] = None) -> None:
        """Initializes the Alternative object with the provided name and evaluations.

        Args:
            evaluations (Dict[str, float]): A dictionary containing the evaluations of the alternative on each
                criterion.
            name (str): The name of the alternative.
        """
        self.evaluations = evaluations
        self.name = name if name is not None else Alternative._default_name
        if name is None:
            Alternative._update_default_name()

    @staticmethod
    def _update_default_name() -> None:
        """Updates the default name of the alternative."""
        Alternative._default_name = f'a{int(Alternative._default_name[1:]) + 1}'

    def get_evaluation(self, criterion_name: str) -> float:
        """Gets the evaluation of the alternative on a specific criterion.

        Args:
            criterion_name (str): The name of the criterion.

        Returns:
            float: The evaluation of the alternative on the specified criterion.
        """
        return self.evaluations.get(criterion_name, 0.0)

    @staticmethod
    def dataframe_to_alternatives(df: pd.DataFrame) -> list[Alternative]:
        """
        Convert a pandas DataFrame to a list of Alternative objects.

        Args:
            df (pd.DataFrame): A pandas DataFrame where the index represents the alternative names and the columns represent the criteria names.

        Returns:
            List[Alternative]: A list of Alternative objects created from the DataFrame.
        """
        alternatives = []

        for index, row in df.iterrows():
            name = index if isinstance(index, str) else None
            evaluations = row.to_dict()
            alternative = Alternative(evaluations=evaluations, name=name)
            alternatives.append(alternative)

        return alternatives
