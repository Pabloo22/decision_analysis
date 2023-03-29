import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

from src.decision_analysis.decision_making import Criterion

sns.set(style="white")


class Promethee:
    """PROMETHEE method for multi-criteria decision-making.

    Args:
        matrix_of_alternative_values (np.array): Matrix with the value for each criterion for each alternative.
        alternatives_names (list): List with the alternatives names.
        criteria (list): List with the criteria.

    Example:
        >>> matrix = np.array([[10, 18, 10], [15, 0, 20]])
        >>> alternatives = ['A', 'B']
        >>> criteria = [Criterion(weight=3, criteria_type=1, preference_threshold=0, indifference_threshold=10),
        ...             Criterion(weight=5, criteria_type=1, preference_threshold=20, indifference_threshold=10),
        ...             Criterion(weight=2, criteria_type=1, preference_threshold=5, indifference_threshold=2)]
        >>> promethee = Promethee(matrix, criteria, alternatives)
        >>> promethee.run()
        >>> print(promethee.comprehensiveness_matrix)
        [[0.  0.4]
         [0.2 0. ]]
    """

    def __init__(self,
                 matrix_of_alternative_values: np.ndarray,
                 criteria: list[Criterion],
                 alternatives_names: list[str] = None):
        self.matrix_of_alternatives_values = matrix_of_alternative_values
        self.alternative_names = alternatives_names
        self.criteria = criteria

        self.comprehensiveness_matrix = None
        self.positive_flow = None
        self.negative_flow = None
        self.net_flow = None

    def run(self):
        self.comprehensiveness_matrix = np.zeros((self.n_alternatives, self.n_alternatives))
        self.positive_flow = np.zeros(self.n_alternatives)
        self.negative_flow = np.zeros(self.n_alternatives)
        self.net_flow = np.zeros(self.n_alternatives)

        self._calculate_comprehensiveness_matrix()
        self._calculate_positive_flow()
        self._calculate_negative_flow()
        self._calculate_net_flow()

    @property
    def n_alternatives(self):
        return self.matrix_of_alternatives_values.shape[0]

    @property
    def n_criteria(self):
        return self.matrix_of_alternatives_values.shape[1]

    def _calculate_comprehensiveness_matrix(self):
        for i in range(self.n_alternatives):
            for j in range(self.n_alternatives):
                self.comprehensiveness_matrix[i, j] = self._calculate_comprehensiveness(i, j)

    def _calculate_comprehensiveness(self, i, j):
        if i == j:
            return 0
        else:
            return sum(self._calculate_comprehensiveness_criterion(i, j, k) / sum(c.weight for c in self.criteria)
                       for k in range(self.n_criteria))

    def _calculate_comprehensiveness_criterion(self, i, j, k):
        """Calculates the comprehensiveness for a given criterion.

        First, it calculates the difference between the value of the criterion for the alternatives i and j, keeping in
        mind that the criterion can be either a benefit or a cost, (that is why we multiply by the criteria type, to
        invert the sign if it is a cost). Then, it checks if the difference is smaller than the indifference threshold,
        in which case it returns 0. If the difference is larger than the preference threshold, it returns the weight of
        the criterion. Otherwise, it interpolates between the indifference and preference thresholds, and returns the
        comprehensiveness value.
        """
        criterion = self.criteria[k]
        diff = criterion.criteria_type * (self.matrix_of_alternatives_values[i, k] - self.matrix_of_alternatives_values[j, k])
        if diff < criterion.indifference_threshold:
            return 0
        elif diff > criterion.preference_threshold:
            return criterion.weight
        else:
            return criterion.weight * (diff - criterion.indifference_threshold) / (
                    criterion.preference_threshold - criterion.indifference_threshold)

    def _calculate_positive_flow(self):
        self.positive_flow = np.sum(self.comprehensiveness_matrix, axis=1)

    def _calculate_negative_flow(self):
        self.negative_flow = np.sum(self.comprehensiveness_matrix, axis=0)

    def _calculate_net_flow(self):
        self.net_flow = self.positive_flow - self.negative_flow

    def rank(self, method: str):
        if method == 'I':
            return self._rank_method_i()
        elif method == 'II':
            return self._rank_method_ii()
        else:
            raise ValueError('Invalid method, must be either "I" or "II"')

    def _rank_method_i(self):
        """Ranks the alternatives based on their positive and negative flows.
        a > b:
        - if a has a higher positive flow than b and a has a lower negative flow than b
        - if a has a higher positive flow than b and a and b have the same negative flow
        - if a and b have the same positive flow and a has a lower negative flow than b

        a R b:
        - if a has a higher positive flow than b and a has a higher negative flow than b
        - if a has a lower positive flow than b and a has a lower negative flow than b

        a = b:
        - if a and b have the same positive flow and a and b have the same negative flow
        """

        g = nx.DiGraph()

        g.add_nodes_from(self.alternative_names)

        ranking_matrix = np.zeros((len(self.alternative_names), len(self.alternative_names)))

        # First, we calculate the outranking relations (we only need to check the outranking relations)
        for i in range(len(self.alternative_names)):
            for j in range(len(self.alternative_names)):
                if i != j:
                    if self.positive_flow[i] > self.positive_flow[j] and self.negative_flow[i] < self.negative_flow[j] or \
                            self.positive_flow[i] > self.positive_flow[j] and self.negative_flow[i] == self.negative_flow[j] or \
                            self.positive_flow[i] == self.positive_flow[j] and self.negative_flow[i] < self.negative_flow[j]:
                        ranking_matrix[i, j] = 1  # Outranking

        # We create a dataframe to make it easier to create the edges
        ranking_df = pd.DataFrame(ranking_matrix, columns=self.alternative_names, index=self.alternative_names)

        # We add the edges to the graph
        g.add_edges_from(ranking_df[ranking_df == 1].stack().index.tolist())

        # We remove the transitive edges
        g = nx.transitive_reduction(g)

        nx.draw(g, with_labels=True, node_size=1000, node_color='lightblue', font_size=16, font_weight='bold',
                edgecolors='black', linewidths=2, alpha=0.9, width=2, font_color='black', arrowsize=20, arrowstyle='->')

    def _rank_method_ii(self):
        """Ranks the alternatives based on their net flow.
        """

        order = self.alternative_names[np.argsort(-self.net_flow)]

        g = nx.DiGraph()
        g.add_nodes_from(order)

        g.add_edges_from(([(order[i], order[i + 1]) for i in range(len(order) - 1)]))

        nx.draw(g, with_labels=True, node_size=1000, node_color='lightblue', font_size=16, font_weight='bold',
                edgecolors='black', linewidths=2, alpha=0.9, width=2, font_color='black', arrowsize=20, arrowstyle='->')

    @staticmethod
    def plot_criterion(criterion: Criterion):
        """Plots the criterion function.

        Args:
            criterion: The criterion to plot.
        """
        no_support = np.linspace(0, criterion.indifference_threshold, 100)
        partial_support = np.linspace(criterion.indifference_threshold, criterion.preference_threshold, 100)
        full_support = np.linspace(criterion.preference_threshold, criterion.preference_threshold + 10, 100)

        no_support_y = np.zeros(len(no_support))
        partial_support_y = criterion.weight * (partial_support - criterion.indifference_threshold) / (
                criterion.preference_threshold - criterion.indifference_threshold)
        full_support_y = np.ones(len(full_support)) * criterion.weight

        plt.plot(no_support, no_support_y, label='No support', color='red')
        plt.plot(partial_support, partial_support_y, label='Partial support', color='orange')
        plt.plot(full_support, full_support_y, label='Full support', color='green')
        plt.legend()
        plt.ylabel('$w_i\cdot\pi_i(a,b)$')
        plt.xlabel('$d_{i}(a,b)$' if criterion.criteria_type == 1 else '$d_{i}(b,a)$')
        plt.show()


if __name__ == '__main__':
    import doctest

    doctest.testmod()
