import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import warnings

import enum
from typing import Optional, Union, NamedTuple, Sequence


class ComparisonType(enum.Enum):
    PREFERENCE = enum.auto()
    INDIFFERENCE = enum.auto()


class Comparison(NamedTuple):
    alternative_1: int
    alternative_2: int
    type: ComparisonType


class Ranking:
    """A class representing a ranking of alternatives.

    A ranking is a graph where each node is an alternative and each edge represents a preference relation between the
    alternatives. The ranking is represented as a matrix where the rows and columns are the alternatives and the value
    in the cell (i, j) is 1 if the alternative i is preferred to the alternative j, 0.5 if the alternatives are
    indifferent and 0 if the alternative i is not at leas as good as the alternative j. If i is not as good as j, and
    j is not as good as i, then Matrix[i, j] = 0 and Matrix[j, i] = 0, this is the way we encode incomparability.

    Args:
        alternatives (int or list[str]): The number of alternatives or a list with the names of the alternatives. If
            the number of alternatives is given, the alternatives will be named as a_1, a_2, a_3, ...

    Attributes:
        alternative_names (list): List with the names of the alternatives.
        matrix (np.array): Matrix representing the ranking. The rows and columns are the alternatives and the
            value in the cell (i, j) is 1 if the alternative i is preferred to the alternative j, 0.5 if the
            alternatives are indifferent and 0 if the alternative j is preferred to the alternative i
    """

    def __init__(self, matrix: Optional[np.ndarray] = None, alternatives: Optional[Union[int, list[str]]] = None):
        if alternatives is None and matrix is None:
            raise ValueError('Either the number of alternatives or the matrix must be given')

        if alternatives is None:
            alternatives = matrix.shape[0]

        if isinstance(alternatives, int):
            alternatives = [f'a_{i}' for i in range(1, alternatives + 1)]

        self.alternative_names = alternatives
        self.matrix = np.zeros((self.n_alternatives, self.n_alternatives)) if matrix is None else matrix

    @property
    def n_alternatives(self) -> int:
        return len(self.alternative_names)

    @staticmethod
    def from_dict(ranking: dict[str, int]) -> 'Ranking':
        """Creates a ranking from a list of preference relations.

        Args:
            ranking (dict): A dictionary with the position of each alternative in the ranking. The keys are the
                alternatives names and the values are the position in the ranking. The alternatives with the
                highest position are the most preferred alternatives.

        Returns:
            np.array: Matrix representing the ranking. The rows and columns are the alternatives and the
                value in the cell (i, j) is 1 if the alternative i is preferred to the alternative j, 0.5 if the
                alternatives are indifferent and 0 if the alternative j is preferred to the alternative i

        Example:
            >>> ranking = {'a_1': 1, 'a_2': 2, 'a_3': 3}
            >>> print(Ranking.from_dict(ranking).matrix)
            [[0. 1. 1.]
             [0. 0. 1.]
             [0. 0. 0.]]
        """
        n_alternatives = len(ranking)
        alternative_names = list(ranking.keys())
        matrix = np.zeros((n_alternatives, n_alternatives))
        for i in range(n_alternatives):
            for j in range(n_alternatives):
                if i == j:
                    continue
                if ranking[alternative_names[i]] < ranking[alternative_names[j]]:
                    matrix[i, j] = 1
                elif ranking[alternative_names[i]] == ranking[alternative_names[j]]:
                    matrix[i, j] = 0.5
        return Ranking(matrix, alternative_names)

    def add_comparisons(self, comparisons: Sequence[Comparison]):
        """Updates the matrix to represent the given comparisons."""
        for comparison in comparisons:
            if comparison.type == ComparisonType.PREFERENCE:
                self.add_preference(comparison.alternative_1, comparison.alternative_2)
            elif comparison.type == ComparisonType.INDIFFERENCE:
                self.add_indifference(comparison.alternative_1, comparison.alternative_2)

    def get_comparisons(self) -> list[Comparison]:
        """Returns the list of comparisons from the ranking.

        Indifference relations are only returned once, for example, if the alternatives 1 and 2 are indifferent, the
        comparison (1, 2) is returned, but not (2, 1).
        """
        comparisons = []
        for i in range(self.n_alternatives):
            for j in range(self.n_alternatives):
                if i == j:
                    continue
                if self.matrix[i, j] == 1:
                    comparisons.append(Comparison(i, j, ComparisonType.PREFERENCE))
                elif self.matrix[i, j] == 0.5 and i < j:
                    comparisons.append(Comparison(i, j, ComparisonType.INDIFFERENCE))
        return comparisons

    def remove_comparisons(self, comparisons: Sequence[Comparison]):
        """Removes a list of comparisons."""
        for comparison in comparisons:
            self.add_incomparability(comparison.alternative_1, comparison.alternative_2)

    def get_preference_relations(self) -> list[tuple[str, str]]:
        """Lists with the preference relations in the ranking."""
        preference_relations = []
        for i in range(self.n_alternatives):
            for j in range(self.n_alternatives):
                if self.matrix[i, j] != 1:
                    continue
                preference_relations.append((self.alternative_names[i], self.alternative_names[j]))
        return preference_relations

    def get_indifference_relations(self) -> list[tuple[str, str]]:
        """Lists with the indifference relations in the ranking."""
        indifference_relations = []
        for i in range(self.n_alternatives):
            for j in range(self.n_alternatives):
                if self.matrix[i, j] != 0.5:
                    continue
                indifference_relations.append((self.alternative_names[i], self.alternative_names[j]))
        return indifference_relations

    def visualize(self, title: Optional[str] = None, seed: Optional[int] = None, layout: str = 'graphviz'):
        """Visualizes the ranking.

        if receiving an ImportError, try to install the optional dependencies:
        https://pygraphviz.github.io/documentation/stable/install.html

        Args:
            title (str): Title of the plot.
            seed (int): Seed for the random number generator.
            layout (str): Method to use for the visualization. Can be 'dot' or 'spring'.
        """
        graph = nx.DiGraph()

        # Add nodes and edges to the graph
        for i, alternative in enumerate(self.alternative_names):
            graph.add_node(alternative)

        preference_relations = self.get_preference_relations()
        # Remove transitive edges
        graph_temp = nx.DiGraph()
        for (u, v) in preference_relations:
            graph_temp.add_edge(u, v)

        graph_reduced = nx.transitive_reduction(graph_temp)
        preference_relations = [(u, v) for (u, v) in graph_reduced.edges()]

        indifference_relations = self.get_indifference_relations()

        for (u, v) in preference_relations:
            graph.add_edge(u, v, type='preference')

        for (u, v) in indifference_relations:
            graph.add_edge(u, v, type='indifference')
            graph.add_edge(v, u, type='indifference')

        # Set positions of nodes
        if seed is not None:
            np.random.seed(seed)

        if layout == 'graphviz':
            try:
                pos = nx.drawing.nx_agraph.graphviz_layout(graph, prog='dot')
            except ImportError:
                warnings.warn('pygraphviz is not installed. Using spring layout instead. To install pygraphviz, '
                              'follow the instructions in '
                              'https://pygraphviz.github.io/documentation/stable/install.html')
                pos = nx.spring_layout(graph, seed=seed)
        elif layout == 'spring':
            pos = nx.spring_layout(graph, seed=seed)
        else:
            raise ValueError(f'Unknown method {layout}')

        # Draw nodes and labels
        nx.draw_networkx_nodes(graph, pos, node_color='skyblue', node_size=1000, alpha=0.5)
        nx.draw_networkx_labels(graph, pos, font_size=12, font_weight='bold')

        # Draw preference edges
        preference_edges = [(u, v) for (u, v, d) in graph.edges(data=True) if d['type'] == 'preference']
        nx.draw_networkx_edges(graph, pos, edgelist=preference_edges, edge_color='b',
                               arrowsize=20, node_size=2000, arrowstyle='->')

        # Draw indifference edges
        indifference_edges = [(u, v) for (u, v, d) in graph.edges(data=True) if d['type'] == 'indifference']
        nx.draw_networkx_edges(graph, pos, edgelist=indifference_edges, edge_color='g', style='dashed', arrowsize=20)

        # Set title and display the plot
        if title is not None:
            plt.title(title)
        plt.axis('off')
        plt.show()

    def _get_alternative_index(self, alternative: Union[int, str]) -> int:
        """Gets the index of an alternative.

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


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    ranking = Ranking(alternatives=4)
    ranking.add_preference(1, 0)
    ranking.add_preference(1, 2)
    ranking.add_preference(0, 3)
    ranking.add_preference(1, 3)
    ranking.visualize(title="Example Ranking", seed=42)
