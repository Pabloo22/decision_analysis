import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Optional, Union, Sequence


class Ranking:
    """A class representing a ranking of alternatives.

    A ranking is a graph where each node is an alternative and each edge represents a preference relation between the
    alternatives. The ranking is represented as a matrix where the rows and columns are the alternatives and the value
    in the cell (i, j) is 1 if the alternative i is preferred to the alternative j, 0.5 if the alternatives are
    indifferent and 0 if the alternative j is preferred to the alternative i.

    Args:
        alternatives (int or list[str]): The number of alternatives or a list with the names of the alternatives. If
            the number of alternatives is given, the alternatives will be named as a_1, a_2, a_3, ...

    Attributes:
        alternative_names (list): List with the names of the alternatives.
        matrix (np.array): Matrix representing the ranking. The rows and columns are the alternatives and the
            value in the cell (i, j) is 1 if the alternative i is preferred to the alternative j, 0.5 if the
            alternatives are indifferent and 0 if the alternative j is preferred to the alternative i
    """

    def __init__(self, alternatives: Union[int, list[str]], matrix: Optional[np.ndarray] = None):
        if isinstance(alternatives, int):
            alternatives = [f'a_{i}' for i in range(1, alternatives + 1)]
        self.alternative_names = alternatives
        self.matrix = np.zeros((self.n_alternatives, self.n_alternatives)) if matrix is None else matrix

    @property
    def n_alternatives(self) -> int:
        return len(self.alternative_names)

    def create_matrix_from_ranking_dict(self, ranking: dict[str, int]):
        """Create a matrix from a list of preference relations.

        Args:
            ranking (dict): A dictionary with the position of each alternative in the ranking. The keys are the
                alternatives names and the values are the position in the ranking. The alternatives with the
                highest position are the most preferred alternatives.

        Returns:
            np.array: Matrix representing the ranking. The rows and columns are the alternatives and the
                value in the cell (i, j) is 1 if the alternative i is preferred to the alternative j, 0.5 if the
                alternatives are indifferent and 0 if the alternative j is preferred to the alternative i
        """
        matrix = np.zeros((self.n_alternatives, self.n_alternatives))
        for i in range(self.n_alternatives):
            for j in range(self.n_alternatives):
                if ranking[self.alternative_names[i]] < ranking[self.alternative_names[j]]:
                    matrix[i, j] = 1
                elif ranking[self.alternative_names[i]] == ranking[self.alternative_names[j]]:
                    matrix[i, j] = 0.5
        return matrix

    def get_preference_relations(self) -> list[tuple[str, str]]:
        """List with the preference relations in the ranking."""
        preference_relations = []
        for i in range(self.n_alternatives):
            for j in range(self.n_alternatives):
                if self.matrix[i, j] != 1:
                    continue
                preference_relations.append((self.alternative_names[i], self.alternative_names[j]))
        return preference_relations

    def get_indifference_relations(self) -> list[tuple[str, str]]:
        """List with the indifference relations in the ranking."""
        indifference_relations = []
        for i in range(self.n_alternatives):
            for j in range(self.n_alternatives):
                if self.matrix[i, j] != 0.5:
                    continue
                indifference_relations.append((self.alternative_names[i], self.alternative_names[j]))
        return indifference_relations

    def visualize(self, title: Optional[str] = None, seed: Optional[int] = None):
        """Visualize the ranking.

        Args:
            title (str): Title of the plot.
            seed (int): Seed for the random number generator.
        """

    def _get_alternative_index(self, alternative: Union[int, str]) -> int:
        """Get the index of an alternative.

        Args:
            alternative (int or str): The alternative.

        Returns:
            int: The index of the alternative.
        """
        if isinstance(alternative, str):
            return self.alternative_names.index(alternative)
        return alternative

    def add_preference(self, alternative_i: Union[int, str], alternative_j: Union[int, str]):
        """Add a preference relation between two alternatives.

        Args:
            alternative_i (int or str): The first alternative.
            alternative_j (int or str): The second alternative.
        """
        alternative_i = self._get_alternative_index(alternative_i)
        alternative_j = self._get_alternative_index(alternative_j)
        self.matrix[alternative_i, alternative_j] = 1
        self.matrix[alternative_j, alternative_i] = 0

    def add_indifference(self, alternative_i: Union[int, str], alternative_j: Union[int, str]):
        """Add an indifference relation between two alternatives.

        Args:
            alternative_i (int or str): The first alternative.
            alternative_j (int or str): The second alternative.
        """
        alternative_i = self._get_alternative_index(alternative_i)
        alternative_j = self._get_alternative_index(alternative_j)
        self.matrix[alternative_i, alternative_j] = 0.5
        self.matrix[alternative_j, alternative_i] = 0.5

    def add_incomparability(self, alternative_i: Union[int, str], alternative_j: Union[int, str]):
        """Add an incomparability relation between two alternatives.

        Args:
            alternative_i (int or str): The first alternative.
            alternative_j (int or str): The second alternative.
        """
        alternative_i = self._get_alternative_index(alternative_i)
        alternative_j = self._get_alternative_index(alternative_j)
        self.matrix[alternative_i, alternative_j] = 0
        self.matrix[alternative_j, alternative_i] = 0

    def kendall_distance(self, other: 'Ranking') -> float:
        """Calculate the Kendall distance between two rankings.

        Args:
            other (Ranking): The other ranking.

        Returns:
            float: The Kendall distance.
        """
        return np.sum(np.abs(self.matrix - other.matrix)) / 2
    
    def kendall_tau(self, other: 'Ranking') -> float:
        """Calculate the Kendall tau between two rankings.

        Args:
            other (Ranking): The other ranking.

        Returns:
            float: The Kendall tau.
        """
        return 1 - 4 * self.kendall_distance(other) / (self.n_alternatives * (self.n_alternatives - 1))
