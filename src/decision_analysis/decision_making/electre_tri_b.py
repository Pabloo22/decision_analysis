import numpy as np
import matplotlib.pyplot as plt
import functools


from src.decision_analysis.decision_making import Criterion, Alternative


class ElectreTriB:
    """Implements the ELECTRE TRI-B method for sorting alternatives into classes.

    Attributes:
        criteria (List[Criterion]): List of criteria to be considered.
        boundaries (List[Alternative]): List of profiles representing class boundaries.
        credibility_threshold (float): Credibility threshold for validating outranking. Optional, default 0.5.
    """

    def __init__(self,
                 criteria: list[Criterion],
                 alternatives: list[Alternative],
                 boundaries: list[Alternative],
                 credibility_threshold: float = 0.5):
        self.criteria = criteria
        self.alternatives = alternatives
        self.boundaries = boundaries
        self.credibility_threshold = credibility_threshold

        self.weights = np.array([c.weight for c in self.criteria])
        self.sum_weights = np.sum(self.weights)
        self.weights = self.weights / self.sum_weights

        self.initialized = False

        self.marginal_concordance_tensor_alt_bound = None
        self.marginal_concordance_tensor_bound_alt = None

        self.discordance_tensor_alt_bound = None
        self.discordance_tensor_bound_alt = None

        self.comprehensive_concordance_matrix_alt_bound = None
        self.comprehensive_concordance_matrix_bound_alt = None

        self.outranking_credibility_matrix_alt_bound = None
        self.outranking_credibility_matrix_bound_alt = None

        self.outranking_matrix = None
        self.relation_matrix = None

        self.optimistic_classes = None
        self.pessimistic_classes = None

    @property
    def n_alternatives(self):
        return len(self.alternatives)

    @property
    def n_boundaries(self):
        return len(self.boundaries)

    @property
    def n_criteria(self):
        return len(self.criteria)

    def run(self):
        """Runs the ELECTRE TRI-B method."""

        self.marginal_concordance_tensor_alt_bound = np.zeros((self.n_alternatives,
                                                               self.n_boundaries,
                                                               self.n_criteria))
        self.marginal_concordance_tensor_bound_alt = np.zeros((self.n_alternatives,
                                                               self.n_boundaries,
                                                               self.n_criteria))

        self.discordance_tensor_alt_bound = np.zeros((self.n_alternatives,
                                                      self.n_boundaries,
                                                      self.n_criteria))
        self.discordance_tensor_bound_alt = np.zeros((self.n_alternatives,
                                                      self.n_boundaries,
                                                      self.n_criteria))

        self.comprehensive_concordance_matrix_alt_bound = np.zeros((self.n_alternatives, self.n_boundaries))
        self.comprehensive_concordance_matrix_bound_alt = np.zeros((self.n_alternatives, self.n_boundaries))

        self.outranking_credibility_matrix_alt_bound = np.zeros((self.n_alternatives, self.n_boundaries))
        self.outranking_credibility_matrix_bound_alt = np.zeros((self.n_alternatives, self.n_boundaries))

        self.outranking_matrix = np.zeros((self.n_alternatives, self.n_boundaries))
        self.relation_matrix = np.zeros((self.n_alternatives, self.n_boundaries))

        self.optimistic_classes = np.zeros((self.n_alternatives,), dtype=int)
        self.pessimistic_classes = np.zeros((self.n_alternatives,), dtype=int)

        self._calculate_marginal_concordance_tensors()
        self._calculate_marginal_discordance_tensors()

        self._calculate_comprehensive_concordance_matrices()

        self._calculate_outranking_credibility_matrices()
        self._calculate_outranking_matrix()
        self.optimistic_classes, self.pessimistic_classes = self.calculate_class_assignments(self.outranking_matrix)

        self.initialized = True

    def _calculate_marginal_concordance_tensors(self):
        """Calculates the concordance matrices."""
        for i, alternative in enumerate(self.alternatives):
            for j, bound in enumerate(self.boundaries):
                for k, criterion in enumerate(self.criteria):
                    self.marginal_concordance_tensor_alt_bound[i, j, k] = \
                        self.get_marginal_concordance(alternative, bound, criterion)

                    self.marginal_concordance_tensor_bound_alt[i, j, k] = \
                        self.get_marginal_concordance(bound, alternative, criterion)

    def _calculate_marginal_discordance_tensors(self):
        """Calculates the discordance matrix."""
        for i, alternative in enumerate(self.alternatives):
            for j, bound in enumerate(self.boundaries):
                for k, criterion in enumerate(self.criteria):
                    self.discordance_tensor_alt_bound[i, j, k] = \
                        self.get_marginal_discordance(alternative, bound, criterion)

                    self.discordance_tensor_bound_alt[i, j, k] = \
                        self.get_marginal_discordance(bound, alternative, criterion)

    def _calculate_comprehensive_concordance_matrices(self):
        """Calculates the comprehensive concordance matrix."""
        # for i in range(self.n_alternatives):
        #     for j in range(self.n_boundaries):
        #         self.comprehensive_concordance_matrix[i, j] = np.sum(
        #             self.weights * self.marginal_concordance_tensor_alt_bound[i, j, :])
        self.comprehensive_concordance_matrix_alt_bound = np.sum(
            self.weights * self.marginal_concordance_tensor_alt_bound, axis=2)
        self.comprehensive_concordance_matrix_bound_alt = np.sum(
            self.weights * self.marginal_concordance_tensor_bound_alt, axis=2)

    def _calculate_outranking_credibility_matrices(self):
        """Calculates the outranking credibility matrix."""
        # for i, alternative in enumerate(self.alternatives):
        #     for j, boundary in enumerate(self.boundaries):
        #         self.outranking_credibility_matrix[i, j] = self._calculate_outranking_credibility(i, j)
        self.outranking_credibility_matrix_alt_bound = np.fromfunction(
            np.vectorize(self._calculate_outranking_credibility),
            (self.n_alternatives, self.n_boundaries))

        self.outranking_credibility_matrix_bound_alt = np.fromfunction(
            np.vectorize(functools.partial(self._calculate_outranking_credibility, alt_bound=False)),
            (self.n_alternatives, self.n_boundaries))

    def _calculate_outranking_matrix(self):
        """Calculates the outranking matrix.

        The outranking matrix is a matrix, where the value of an element is 1 if the alternative
        outranks the boundary, 0.5 if the alternative is indifferent to the boundary, 0 if the alternative
        is outranked by the boundary, and -1 if the alternative is incomparable to the boundary.
        """
        for i, a in enumerate(self.alternatives):
            for j, b in enumerate(self.boundaries):
                a_is_at_least_as_good_as_b = (self.outranking_credibility_matrix_alt_bound[i, j] >=
                                              self.credibility_threshold)
                b_is_at_least_as_good_as_a = (self.outranking_credibility_matrix_bound_alt[i, j] >=
                                              self.credibility_threshold)

                a_is_preferred_to_b = a_is_at_least_as_good_as_b and not b_is_at_least_as_good_as_a
                if a_is_preferred_to_b:
                    self.outranking_matrix[i, j] = 1
                    continue

                b_is_preferred_to_a = b_is_at_least_as_good_as_a and not a_is_at_least_as_good_as_b
                if b_is_preferred_to_a:
                    self.outranking_matrix[i, j] = 0
                    continue

                a_is_indifferent_to_b = a_is_at_least_as_good_as_b and b_is_at_least_as_good_as_a
                if a_is_indifferent_to_b:
                    self.outranking_matrix[i, j] = 0.5
                    continue

                self.outranking_matrix[i, j] = -1

    @staticmethod
    def calculate_class_assignments(outranking_credibility_matrix_alt_bound: np.ndarray
                                    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculates the class assignments.

        The lower the class assignment, the lower the preference of the alternative.

        Args:
            outranking_credibility_matrix_alt_bound: The outranking credibility matrix. Relations
                are encoded as follows:
                    * 1: a outranks b
                    * 0.5: a is indifferent to b
                    * 0: a is outranked by b
                    * -1: a is incomparable to b
                where a is an alternative and b is a boundary. The matrix is indexed as follows:
                outranking_credibility_matrix_alt_bound[alternative, boundary].

        Returns:
            Optimistic and pessimistic class assignments.
        """
        return (ElectreTriB._optimistic_class_assignment(outranking_credibility_matrix_alt_bound),
                ElectreTriB._pessimistic_class_assignment(outranking_credibility_matrix_alt_bound))

    @staticmethod
    def _optimistic_class_assignment(outranking_credibility_matrix_alt_bound: np.ndarray
                                     ) -> np.ndarray:
        """Calculates the optimistic class assignment.

        1. Start from the worst profile
        2. find the first profile bh: b_h > a
        3. select class h

        The lower the class assignment, the lower the preference of the alternative.
        """
        n_alternatives = outranking_credibility_matrix_alt_bound.shape[0]
        optimistic_class_assignment = np.zeros(n_alternatives, dtype=int)
        for i, a in enumerate(outranking_credibility_matrix_alt_bound):
            optimistic_class_assignment[i] = n_alternatives
            for h, relation in enumerate(a):
                if relation == 0:  # relations = {-1: '?', 0: '<', 1: '>', 0.5: '='}
                    optimistic_class_assignment[i] = h
                    break
        return optimistic_class_assignment

    @staticmethod
    def _pessimistic_class_assignment(outranking_credibility_matrix_alt_bound: np.ndarray
                                      ) -> np.ndarray:
        """Calculates the pessimistic class assignment.
        
        1. Start from the best profile
        2. find the first profile bh: a S b_h
        3. select class h + 1

        The lower the class assignment, the lower the preference of the alternative.
        """
        n_alternatives = outranking_credibility_matrix_alt_bound.shape[0]
        n_boundaries = outranking_credibility_matrix_alt_bound.shape[1]
        pessimistic_class_assignment = np.zeros(n_alternatives, dtype=int)
        for i, a in enumerate(outranking_credibility_matrix_alt_bound):
            pessimistic_class_assignment[i] = 1
            for h, relation in enumerate(reversed(a)):
                a_is_at_least_as_good_as_b = relation >= 0.5
                if a_is_at_least_as_good_as_b:  # relations = {-1: '?', 0: '<', 1: '>', 0.5: '='}
                    pessimistic_class_assignment[i] = n_boundaries - h
                    break
        return pessimistic_class_assignment

    def _calculate_outranking_credibility(self,
                                          alternative_idx: int,
                                          boundary_idx: int,
                                          alt_bound: bool = True) -> float:
        """Outranking credibility σ aggregates the comprehensive concordance and marginal discordances"""

        alternative_idx = int(alternative_idx)
        boundary_idx = int(boundary_idx)
        if alt_bound:
            comprehensive_concordance = self.comprehensive_concordance_matrix_alt_bound[alternative_idx,
                                                                                        boundary_idx]
        else:
            comprehensive_concordance = self.comprehensive_concordance_matrix_bound_alt[alternative_idx,
                                                                                        boundary_idx]
        outranking_credibility = comprehensive_concordance

        # F = {j = 1,…, n: Dj(a, bh) > C(a, bh)}
        for j, _ in enumerate(self.criteria):
            if alt_bound:
                dj_a_b = self.discordance_tensor_alt_bound[alternative_idx, boundary_idx, j]
            else:
                dj_a_b = self.discordance_tensor_bound_alt[alternative_idx, boundary_idx, j]
            if dj_a_b > comprehensive_concordance:
                outranking_credibility *= (1 - dj_a_b) / (1 - comprehensive_concordance)

        return outranking_credibility

    @staticmethod
    def get_marginal_concordance(a: Alternative,
                                 b: Alternative,
                                 criterion: Criterion) -> float:
        """Calculates the marginal concordance between one alternative and one profile boundary.
        
        Args:
            a (Alternative): Alternative to compare or profile boundary.
            b (Alternative): Alternative to compare or profile boundary.
            criterion (Criterion): Criterion to compare.
        """
        a_value = a.get_evaluation(criterion.name)
        b_value = b.get_evaluation(criterion.name)

        if criterion.criteria_type == 1:
            if a_value - b_value >= -criterion.indifference_threshold:
                return 1.
            elif a_value - b_value < -criterion.preference_threshold:
                return 0.
            else:
                return (criterion.preference_threshold - (b_value - a_value)) / (
                        criterion.preference_threshold - criterion.indifference_threshold)
        else:
            if a_value - b_value <= criterion.indifference_threshold:
                return 1.
            elif a_value - b_value > criterion.preference_threshold:
                return 0.
            else:
                return (criterion.preference_threshold - (a_value - b_value)) / (
                        criterion.preference_threshold - criterion.indifference_threshold)

    @staticmethod
    def get_marginal_discordance(a: Alternative, b: Alternative, criterion: Criterion) -> float:
        a_value = a.get_evaluation(criterion.name)
        b_value = b.get_evaluation(criterion.name)
        a_value = a_value
        if criterion.criteria_type == 1:
            if a_value - b_value <= -criterion.veto_threshold:
                return 1.
            elif a_value - b_value >= -criterion.preference_threshold:
                return 0.
            else:
                return ((b_value - a_value) - criterion.preference_threshold) / (
                        criterion.veto_threshold - criterion.preference_threshold)
        else:
            if a_value - b_value >= criterion.veto_threshold:
                return 1.
            elif a_value - b_value <= criterion.preference_threshold:
                return 0.
            else:
                return (criterion.veto_threshold - (a_value - b_value)) / (
                        criterion.veto_threshold - criterion.preference_threshold)

    def _is_alt_bound(self, a: Alternative, b: Alternative) -> bool:
        """Checks if an alternative is a profile boundary."""
        b_in_boundaries = b in self.boundaries
        if b not in self.alternatives and not b_in_boundaries:
            raise ValueError(f"Alternative {b} is not in the alternatives or boundaries.")
        if a not in self.alternatives and a not in self.boundaries:
            raise ValueError(f"Alternative {a} is not in the alternatives or boundaries.")

        return b_in_boundaries

    def get_comprehensive_concordance(self, a: Alternative, b: Alternative):
        """Calculates the comprehensive concordance between an alternative and a boundary.

        Args:
            a (Alternative): Alternative/Profile boundary to be compared.
            b (Alternative): Alternative/Profile boundary to be compared.

        Returns:
            float: Comprehensive concordance between the alternative and the boundary.
        """
        if self.comprehensive_concordance_matrix_alt_bound is None:
            self.run()

        if self._is_alt_bound(a, b):
            return self.comprehensive_concordance_matrix_alt_bound[self.alternatives.index(a),
                                                                   self.boundaries.index(b)]

        return self.comprehensive_concordance_matrix_bound_alt[self.alternatives.index(b),
                                                               self.boundaries.index(a)]

    def get_outranking_credibility(self, a: Alternative, b: Alternative) -> str:
        """Returns the outranking credibility of an alternative for a boundary."""

        if not self.initialized:
            self.run()

        if self._is_alt_bound(a, b):
            return self.outranking_credibility_matrix_alt_bound[self.alternatives.index(a),
                                                                self.boundaries.index(b)]

        return self.outranking_credibility_matrix_bound_alt[self.alternatives.index(b),
                                                            self.boundaries.index(a)]

    def get_outranking_relation(self, alternative: Alternative, boundary: Alternative) -> str:
        """Returns the outranking relation between an alternative and a boundary."""
        if not self.initialized:
            self.run()

        if not self._is_alt_bound(alternative, boundary):
            raise ValueError(f"Argument 'alternative' must be an alternative and 'boundary' a boundary.")

        relations = {-1: '?', 0: '<', 1: '>', 0.5: '='}

        relation = self.outranking_matrix[self.alternatives.index(alternative),
                                          self.boundaries.index(boundary)]
        return relations[relation]

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
        plt.ylabel(r'$w_{i}\cdot c_i(a,b_h)$')
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
                        100)  # 'a' worse than 'b' by at least v_i

        b_partial_veto_a = np.linspace(g_i_b - criterion.veto_threshold, g_i_b - criterion.preference_threshold,
                                       100) if criterion.criteria_type == 1 else \
            np.linspace(g_i_b + criterion.preference_threshold, g_i_b + criterion.veto_threshold,
                        100)  # 'a' worse than 'b' by at least p_i but less than v_i

        b_no_veto_a = np.linspace(g_i_b - criterion.preference_threshold, g_i_b + criterion.preference_threshold,
                                  100) if criterion.criteria_type == 1 else \
            np.linspace(g_i_b - criterion.preference_threshold, g_i_b + criterion.preference_threshold,
                        100)  # 'a' as good as 'b' or worse than b by at most p_i

        b_veto_a_y = np.zeros(len(b_veto_a))
        b_partial_veto_a_y = criterion.weight * (
                criterion.veto_threshold - criterion.criteria_type * (g_i_b - b_partial_veto_a)) / (
                                     criterion.veto_threshold - criterion.preference_threshold)
        b_no_veto_a_y = np.ones(len(b_no_veto_a)) * criterion.weight

        plt.plot(b_veto_a, b_veto_a_y, color='purple')
        plt.plot(b_partial_veto_a, b_partial_veto_a_y, color='orange')
        plt.plot(b_no_veto_a, b_no_veto_a_y, color='yellowgreen')
        plt.ylabel(r'$w_{i}\cdot D_i(a,b_h)$')
        plt.xlabel('$g_i(b)$')

        plt.show()

    def plot_boundary_profiles(self, use_names=False):
        for i, boundary in enumerate(self.boundaries):
            plt.plot(boundary.evaluations.values(), range(len(boundary.evaluations)), marker='o',
                     label=f'Class {i +1}' if not use_names else boundary.name)
        plt.legend()
        plt.show()
