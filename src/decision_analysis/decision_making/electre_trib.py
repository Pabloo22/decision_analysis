import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.decision_analysis.decision_making import Criterion

sns.set(style="white")


class Electre_trib:
    """ELECTRE-TRI method for multi-criteria decision-making.

    Args:
        matrix (np.array): Matrix with the value for each criterion for each alternative.
        criteria (list): List with the criteria.
        boundaries_matrix (np.array): Matrix with the boundaries for each criterion for each alternative.
        boundaries_names (list): List with the boundaries names.
        alternatives (list): List with the alternatives names.
    """

    def __init__(self,
                 matrix: np.ndarray,
                 criteria: list[Criterion],
                 boundaries_matrix: np.ndarray,
                 boundaries_names: list[str] = None,
                 alternatives: list[str] = None):
        self.matrix = matrix
        self.alternatives = alternatives
        self.criteria = criteria
        self.boundaries_matrix = boundaries_matrix
        self.boundaries_names = boundaries_names

        self.weights = np.array([c.weight for c in self.criteria])
        self.sum_weights = np.sum(self.weights)
        self.weights = self.weights / self.sum_weights
        self.marginal_concordance_tensor_alt_bound = None
        self.marginal_concordance_tensor_bound_alt = None
        self.marginal_discordance_tensor_alt_bound = None
        self.marginal_discordance_tensor_bound_alt = None
        self.concordance_matrix = None
        self.outranking_credibility_matrix = None
        self.outranking_matrix = None
        self.relation_matrix = None

    @property
    def n_alternatives(self):
        return self.matrix.shape[0]

    @property
    def n_criteria(self):
        return self.matrix.shape[1]

    @property
    def n_boundaries(self):
        return self.boundaries_matrix.shape[0]

    def run(self):
        self.marginal_concordance_tensor_alt_bound = np.zeros((self.n_alternatives, self.n_boundaries, self.n_criteria))
        self.marginal_concordance_tensor_bound_alt = np.zeros((self.n_alternatives, self.n_boundaries, self.n_criteria))
        self.marginal_discordance_tensor_alt_bound = np.zeros((self.n_alternatives, self.n_boundaries, self.n_criteria))
        self.marginal_discordance_tensor_bound_alt = np.zeros((self.n_alternatives, self.n_boundaries, self.n_criteria))
        self.concordance_matrix = np.zeros((self.n_alternatives, self.n_boundaries))
        self.outranking_credibility_matrix = np.zeros((self.n_alternatives, self.n_boundaries))
        self.outranking_matrix = np.zeros((self.n_alternatives, self.n_boundaries))
        self.relation_matrix = np.zeros((self.n_alternatives, self.n_boundaries))

        self._calculate_marginal_concordance_tensors()
        self._calculate_marginal_discordance_tensors()
        self._calculate_concordance_matrix()
        self._calculate_outranking_credibility_matrix()
        self._calculate_outranking_matrix()
        self._calculate_relation_matrix()

    @staticmethod
    def _calculate_marginal_concordance(a: float, b: float, criteria: Criterion):
        if criteria.criteria_type == 1:
            if a - b >= -criteria.indifference_threshold:
                return 1
            elif a - b < -criteria.preference_threshold:
                return 0
            else:
                return (criteria.preference_threshold - (b - a)) / (
                        criteria.preference_threshold - criteria.indifference_threshold)
        else:
            if a - b <= criteria.indifference_threshold:
                return 1
            elif a - b > criteria.preference_threshold:
                return 0
            else:
                return (criteria.preference_threshold - (a - b)) / (
                        criteria.preference_threshold - criteria.indifference_threshold)

    def _calculate_marginal_concordance_tensors(self):
        for i in range(self.n_alternatives):
            for j in range(self.n_boundaries):
                for k in range(self.n_criteria):
                    self.marginal_concordance_tensor_alt_bound[i, j, k] = self._calculate_marginal_concordance(
                        self.matrix[i, k],
                        self.boundaries_matrix[j, k],
                        self.criteria[k]) * self.weights[k]
                    self.marginal_concordance_tensor_bound_alt[i, j, k] = self._calculate_marginal_concordance(
                        self.boundaries_matrix[j, k],
                        self.matrix[i, k],
                        self.criteria[k]) * self.weights[k]

    @staticmethod
    def _calculate_marginal_discordance(a: float, b: float, criteria: Criterion):
        if criteria.criteria_type == 1:
            if a - b <= -criteria.veto_threshold:
                return 1
            elif a - b >= -criteria.preference_threshold:
                return 0
            else:
                return ((b - a) - criteria.preference_threshold) / (
                        criteria.veto_threshold - criteria.preference_threshold)
        else:
            if a - b >= criteria.veto_threshold:
                return 1
            elif a - b <= criteria.preference_threshold:
                return 0
            else:
                return (criteria.veto_threshold - (a - b)) / (
                        criteria.veto_threshold - criteria.preference_threshold)

    def _calculate_marginal_discordance_tensors(self):
        for i in range(self.n_alternatives):
            for j in range(self.n_boundaries):
                for k in range(self.n_criteria):
                    self.marginal_discordance_tensor_alt_bound[i, j, k] = self._calculate_marginal_discordance(
                        self.matrix[i, k],
                        self.boundaries_matrix[j, k],
                        self.criteria[k]) * self.weights[k]
                    self.marginal_discordance_tensor_bound_alt[i, j, k] = self._calculate_marginal_discordance(
                        self.boundaries_matrix[j, k],
                        self.matrix[i, k],
                        self.criteria[k]) * self.weights[k]

    def _calculate_concordance_matrix(self):
        pass

    def _calculate_outranking_credibility(self, a: int, b: int):
        outranking_credibility = self.concordance_matrix[a, b]
        pass

    def _calculate_outranking_credibility_matrix(self):
        for i in range(self.n_alternatives):
            for j in range(self.n_boundaries):
                self.outranking_credibility_matrix[i, j] = self._calculate_outranking_credibility(i, j)

    def _calculate_outranking_matrix(self):
        pass

    def _calculate_relation_matrix(self):
        pass

    def optimistic_ranking(self):
        pass

    def pessimistic_ranking(self):
        pass

    @staticmethod
    def plot_marginal_concordance(criterion: Criterion):
        """Plots the marginal concordance for the criterion.

        Args:
            criterion: The criterion to plot.
        """
        g_i_b = 10 + criterion.preference_threshold  # g_i(b) example value, only for plotting

        b_preference_a = np.linspace(0, g_i_b - criterion.preference_threshold,
                                     100) if criterion.criteria_type == 1 else \
            np.linspace(g_i_b + criterion.preference_threshold, g_i_b + criterion.preference_threshold + 10,
                        100)  # a P_i b
        b_weak_preference_a = np.linspace(g_i_b - criterion.preference_threshold,
                                          g_i_b - criterion.indifference_threshold,
                                          100) if criterion.criteria_type == 1 else \
            np.linspace(g_i_b + criterion.indifference_threshold, g_i_b + criterion.preference_threshold,
                        100)  # b Q_i a
        b_indifference_a = np.linspace(g_i_b - criterion.indifference_threshold,
                                       g_i_b + criterion.indifference_threshold,
                                       100) if criterion.criteria_type == 1 else \
            np.linspace(g_i_b - criterion.indifference_threshold, g_i_b + criterion.indifference_threshold,
                        100)  # a I_i b
        b_weak_dispreference_a = np.linspace(g_i_b + criterion.indifference_threshold,
                                             g_i_b + criterion.preference_threshold,
                                             100) if criterion.criteria_type == 1 else \
            np.linspace(g_i_b - criterion.preference_threshold, g_i_b - criterion.indifference_threshold,
                        100)  # a Q_i b
        b_dispreference_a = np.linspace(g_i_b + criterion.preference_threshold,
                                        g_i_b + criterion.preference_threshold + 10,
                                        100) if criterion.criteria_type == 1 else \
            np.linspace(0, g_i_b - criterion.preference_threshold, 100)  # a P_i b

        b_preference_a_y = np.zeros(len(b_preference_a))
        b_weak_preference_a_y = criterion.weight * (
                criterion.preference_threshold - criterion.criteria_type * (g_i_b - b_weak_preference_a)) / (
                                        criterion.preference_threshold - criterion.indifference_threshold)
        b_indifference_a_y = np.ones(len(b_indifference_a)) * criterion.weight
        b_weak_dispreference_a_y = np.ones(len(b_weak_dispreference_a)) * criterion.weight
        b_dispreference_a_y = np.ones(len(b_dispreference_a)) * criterion.weight

        plt.plot(b_preference_a, b_preference_a_y, label='$b_h P_i a$', color='purple')
        plt.plot(b_weak_preference_a, b_weak_preference_a_y, label='$b_h Q_i a$', color='orange')
        plt.plot(b_indifference_a, b_indifference_a_y, label='$a I_i b_h$', color='yellowgreen')
        plt.plot(b_weak_dispreference_a, b_weak_dispreference_a_y, label='$a Q_i b_h$', color='green')
        plt.plot(b_dispreference_a, b_dispreference_a_y, label='$a P_i b_h$', color='aquamarine')
        plt.legend()
        plt.ylabel('$w_{i}\cdot c_i(a,b_h)$')
        plt.xlabel('$g_i(b)$')
        plt.show()

    @staticmethod
    def plot_marginal_discordance(criterion: Criterion):
        """Plots the marginal discordance for the criterion.

        Args:
            criterion: The criterion to plot.
        """
        g_i_b = 10 + criterion.veto_threshold  # g_i(b) example value, only for plotting

        b_veto_a = np.linspace(0, g_i_b - criterion.veto_threshold, 100) if criterion.criteria_type == 1 else \
            np.linspace(g_i_b + criterion.veto_threshold, g_i_b + criterion.veto_threshold + 10,
                        100)  # a worse than b by at least v_i

        b_partial_veto_a = np.linspace(g_i_b - criterion.veto_threshold, g_i_b - criterion.preference_threshold,
                                       100) if criterion.criteria_type == 1 else \
            np.linspace(g_i_b + criterion.preference_threshold, g_i_b + criterion.veto_threshold,
                        100)  # a worse than b by at least p_i but less than v_i

        b_no_veto_a = np.linspace(g_i_b - criterion.preference_threshold, g_i_b + criterion.preference_threshold,
                                  100) if criterion.criteria_type == 1 else \
            np.linspace(g_i_b - criterion.preference_threshold, g_i_b + criterion.preference_threshold,
                        100)  # a as good as b or worse than b by at most p_i

        b_veto_a_y = np.zeros(len(b_veto_a))
        b_partial_veto_a_y = criterion.weight * (
                criterion.veto_threshold - criterion.criteria_type * (g_i_b - b_partial_veto_a)) / (
                                     criterion.veto_threshold - criterion.preference_threshold)
        b_no_veto_a_y = np.ones(len(b_no_veto_a)) * criterion.weight

        plt.plot(b_veto_a, b_veto_a_y, color='purple')
        plt.plot(b_partial_veto_a, b_partial_veto_a_y, color='orange')
        plt.plot(b_no_veto_a, b_no_veto_a_y, color='yellowgreen')
        plt.ylabel('$w_{i}\cdot D_i(a,b_h)$')
        plt.xlabel('$g_i(b)$')

        plt.show()

    def plot_boundary_profiles(self):
        for i in range(self.boundaries_matrix.shape[0]):
            plt.plot(self.boundaries_matrix[i, :], range(self.boundaries_matrix.shape[1]), marker='o',
                     label=f'Class {i + 1}' if self.boundaries_names is None else self.boundaries_names[i])

        plt.legend()
        plt.show()
