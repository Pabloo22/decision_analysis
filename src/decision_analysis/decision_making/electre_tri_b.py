from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from src.decision_analysis.decision_making import Criterion, Alternative


class ElectreTriB:
    """Implements the ELECTRE TRI-B method for sorting alternatives into classes.

    Attributes:
        criteria (List[Criterion]): List of criteria to be considered.
        profiles (List[Alternative]): List of profiles representing class boundaries.
        cutting_level (float): Credibility threshold for validating outranking. Optional, default 0.5.
    """

    def __init__(self,
                 criteria: list[Criterion],
                 alternatives: list[Alternative],
                 profiles: list[Alternative],
                 cutting_level: float = 0.5):
        self.criteria = criteria
        self.alternatives = alternatives
        self.profiles = profiles
        self.cutting_level = cutting_level

        self.concordance_matrix = None
        self.discordance_matrix = None
        self.outranking_matrix = None
        self.outranking_credibility_matrix = None
        self.relation_matrix = None

    @property
    def n_alternatives(self):
        return len(self.alternatives)

    @property
    def n_profiles(self):
        return len(self.profiles)

    def run(self):
        """Runs the ELECTRE TRI-B method."""
        self.concordance_matrix = np.zeros((self.n_alternatives, self.n_profiles))
        self.discordance_matrix = np.zeros((self.n_alternatives, self.n_profiles))
        self.outranking_matrix = np.zeros((self.n_alternatives, self.n_profiles))
        self.outranking_credibility_matrix = np.zeros((self.n_alternatives, self.n_profiles))
        self.relation_matrix = np.zeros((self.n_alternatives, self.n_alternatives))

        self._calculate_concordance_matrix()
        self._calculate_discordance_matrix()
        self._calculate_outranking_matrix()
        self._calculate_outranking_credibility_matrix()
        self._calculate_relation_matrix()

    def _calculate_concordance_matrix(self):
        """Calculates the concordance matrix."""
        for i, alternative in enumerate(self.alternatives):
            for j, profile in enumerate(self.profiles):
                self.concordance_matrix[i, j] = self.compute_concordance(alternative, profile)

    def _calculate_discordance_matrix(self):
        """Calculates the discordance matrix."""
        for i, alternative in enumerate(self.alternatives):
            for j, profile in enumerate(self.profiles):
                for criterion in self.criteria:
                    self.discordance_matrix[i, j] += self.compute_discordance(alternative, profile, criterion)

    def _calculate_outranking_matrix(self):
        """Calculates the outranking matrix."""
        for i in range(self.n_alternatives):
            for j in range(self.n_profiles):
                self.outranking_matrix[i, j] = self.concordance_matrix[i, j] - self.discordance_matrix[i, j]

    @staticmethod
    def calculate_marginal_concordance(a: Alternative, b: Alternative, criterion: Criterion):
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
