from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from src.decision_analysis.decision_making import Criterion, Alternative


class ElectreTriB:
    """Implements the ELECTRE TRI-B method for sorting alternatives into classes.

    Attributes:
        criteria (List[Criterion]): List of criteria to be considered.
        boundaries (List[Alternative]): List of profiles representing class boundaries.
        cutting_level (float): Credibility threshold for validating outranking. Optional, default 0.5.
    """

    def __init__(self,
                 criteria: list[Criterion],
                 alternatives: list[Alternative],
                 profiles: list[Alternative],
                 cutting_level: float = 0.5):
        self.criteria = criteria
        self.alternatives = alternatives
        self.boundaries = profiles
        self.cutting_level = cutting_level

        self.weights = np.array([c.weight for c in self.criteria])
        self.sum_weights = np.sum(self.weights)
        self.weights = self.weights / self.sum_weights

        self.marginal_concordance_tensor_alt_bound = None
        self.marginal_concordance_tensor_bound_alt = None

        self.marginal_discordance_tensor_alt_bound = None
        self.marginal_discordance_tensor_bound_alt = None

        self.comprehensive_concordance_matrix = None

        self.outranking_credibility_matrix = None
        self.outranking_matrix = None
        self.relation_matrix = None

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

        self.marginal_discordance_tensor_alt_bound = np.zeros((self.n_alternatives,
                                                               self.n_boundaries,
                                                               self.n_criteria))
        self.marginal_discordance_tensor_bound_alt = np.zeros((self.n_alternatives,
                                                               self.n_boundaries,
                                                               self.n_criteria))

        self.comprehensive_concordance_matrix = np.zeros((self.n_alternatives, self.n_boundaries))
        self.outranking_credibility_matrix = np.zeros((self.n_alternatives, self.n_boundaries))

        self.outranking_matrix = np.zeros((self.n_alternatives, self.n_boundaries))
        self.relation_matrix = np.zeros((self.n_alternatives, self.n_boundaries))

        self._calculate_marginal_concordance_tensors()
        self._calculate_marginal_discordance_tensors()

        self._calculate_comprehensive_concordance_matrix()

        self._calculate_outranking_credibility_matrix()
        self._calculate_outranking_matrix()
        self._calculate_relation_matrix()

    def _calculate_marginal_concordance_tensors(self):
        """Calculates the concordance matrices."""
        for i, alternative in enumerate(self.alternatives):
            for j, bound in enumerate(self.boundaries):
                for k, criterion in enumerate(self.criteria):
                    self.marginal_concordance_tensor_alt_bound[i, j, k] = \
                        self.calculate_marginal_concordance(alternative, bound, criterion)

                    self.marginal_concordance_tensor_bound_alt[i, j, k] = \
                        self.calculate_marginal_concordance(bound, alternative, criterion)

    def _calculate_marginal_discordance_tensors(self):
        """Calculates the discordance matrix."""
        for i, alternative in enumerate(self.alternatives):
            for j, bound in enumerate(self.boundaries):
                for k, criterion in enumerate(self.criteria):
                    self.marginal_discordance_tensor_alt_bound[i, j, k] = \
                        self.calculate_marginal_discordance(alternative, bound, criterion)

                    self.marginal_discordance_tensor_bound_alt[i, j, k] = \
                        self.calculate_marginal_discordance(bound, alternative, criterion)

    def _calculate_comprehensive_concordance_matrix(self):
        """Calculates the comprehensive concordance matrix."""
        for i in range(self.n_alternatives):
            for j in range(self.n_boundaries):
                self.comprehensive_concordance_matrix[i, j] = np.sum(
                    self.weights * self.marginal_concordance_tensor_alt_bound[i, j, :])
    def _calculate_outranking_matrix(self):
        """Calculates the outranking matrix."""
        for i in range(self.n_alternatives):
            for j in range(self.n_boundaries):
                self.outranking_matrix[i, j] = self.comprehensive_concordance_matrix[i, j] - self.discordance_matrix[i, j]

    def calculate_marginal_concordance(self,
                                       a: Alternative,
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
                return 1
            elif a_value - b_value < -criterion.preference_threshold:
                return 0
            else:
                return (criterion.preference_threshold - (b_value - a_value)) / (
                        criterion.preference_threshold - criterion.indifference_threshold)
        else:
            if a_value - b_value <= criterion.indifference_threshold:
                return 1
            elif a_value - b_value > criterion.preference_threshold:
                return 0
            else:
                return (criterion.preference_threshold - (a_value - b_value)) / (
                        criterion.preference_threshold - criterion.indifference_threshold)

    @staticmethod
    def calculate_marginal_discordance(a: Alternative, b: Alternative, criterion: Criterion):
        a_value = a.get_evaluation(criterion.name)
        b_value = b.get_evaluation(criterion.name)
        a_value = a_value
        if criterion.criteria_type == 1:
            if a_value - b_value <= -criterion.veto_threshold:
                return 1
            elif a_value - b_value >= -criterion.preference_threshold:
                return 0
            else:
                return ((b_value - a_value) - criterion.preference_threshold) / (
                        criterion.veto_threshold - criterion.preference_threshold)
        else:
            if a_value - b_value >= criterion.veto_threshold:
                return 1
            elif a_value - b_value <= criterion.preference_threshold:
                return 0
            else:
                return (criterion.veto_threshold - (a_value - b_value)) / (
                        criterion.veto_threshold - criterion.preference_threshold)

    def compute_concordance(self, alternative: Alternative, profile: Alternative) -> float:
        """Computes the concordance index for an alternative and a profile.

        Args:
            alternative (Alternative): Alternative to compare.
            profile (Alternative): Profile to compare.

        Returns:
            float: The concordance index.
            """
        concordance = 0
        total_weight = 0

        for criterion in self.criteria:
            key = criterion.name
            alternative_value = alternative.get_evaluation(key)
            profile_value = profile.get_evaluation(key)
            if alternative_value * criterion.criteria_type >= profile_value * criterion.criteria_type:
                concordance += criterion.weight
            total_weight += criterion.weight

        return concordance / total_weight

    @staticmethod
    def compute_discordance(alternative: Alternative, profile: Alternative, criterion: Criterion) -> float:
        """Computes the discordance index for an alternative, a profile, and a criterion.

        Args:
            alternative (Alternative): Alternative to compare.
            profile (Alternative):
            criterion (Criterion): The criterion to calculate discordance.

        Returns:
            float: The discordance index.
        """
        key = criterion.name
        alternative_value = alternative.get_evaluation(key)
        profile_value = profile.get_evaluation(key)
        diff = abs(alternative_value - profile_value)

        if diff <= criterion.indifference_threshold:
            return 0
        elif criterion.preference_threshold is not None and diff < criterion.preference_threshold:
            return (diff - criterion.indifference_threshold) / (
                        criterion.preference_threshold - criterion.indifference_threshold)
        else:
            return 1
