import pulp

from src.decision_analysis.decision_making import Dataset, Ranking, Comparison


class UTA:
    def __init__(self, dataset: Dataset, comparisons: Comparison):
        self.dataset = dataset

        self.find_value_functions_model = pulp.LpProblem("Find value functions", pulp.LpMinimize)
        self.find_inconsistency_model = pulp.LpProblem("Find inconsistency", pulp.LpMinimize)
        self.comparisons = comparisons
        self.preference_info_ranking = Ranking(alternatives=dataset.alternative_names)
        self.preference_info_ranking.add_comparisons(comparisons)

    def find_minimal_inconsistent_subset(self, ranking: Ranking) -> list[Comparison]:
        """Finds a minimal subset of constraints that need to be removed to restore consistency.

        Args:
            ranking: The ranking that we want to check for consistency. It does not need to be complete. If the
                comparison between two alternatives 'i' and 'j' is not present in the ranking, the value of the
                matrix at position (i, j) and (j, i) is 0.

        Returns:
            A list of Comparison objects that need to be removed to restore consistency.
        """

    def _get_inconsistent_comparisons(self) -> list[Comparison]:
        """Creates the inconsistent comparisons based on the results of the model.

        Returns:
            A list of Comparison objects that need to be removed to restore consistency.
        """

    def update_value_functions(self) -> None:
        """Updates the value functions of the dataset based on the results of the model.

        The value functions are the values of the objective function for each alternative.
        """

    def _add_general_constraints(self, model: pulp.LpProblem) -> None:
        """Adds the general constraints to the model.

        The general constraints are the ones that are common to both ordinal regression problems:
            - Normalization constraint.
            - Monotonicity constraint.
            - Non-negativity constraint.

        Args:
            model: The model to which the constraints are added.
        """
