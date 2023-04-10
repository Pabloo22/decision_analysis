import pulp

from src.decision_analysis.decision_making import Dataset, Ranking, Comparison, ComparisonType


class UTA:
    """Class that implements the UTA method.

    It uses linear programming in order to optimally infer additive value functions, so that these functions are
    as consistent as possible with the global decision-maker’s preferences.

    The optimal value functions are found by inferring the value of some specific breakpoints of the value
    functions. These breakpoints are defined when creating the criteria of the dataset by setting
    the characteristic_points_locations attribute of the value function of each criterion.

    This class also implements the method to find the minimal inconsistent subset of comparisons.

    Attributes:
        dataset: The dataset that we want to analyze.
        comparisons: The comparisons made by the decision maker.
    """
    def __init__(self, dataset: Dataset, comparisons: Comparison, epsilon: float = 0.0001):
        self.dataset = dataset

        self.value_functions_prob = pulp.LpProblem("Find value functions", pulp.LpMinimize)
        self.inconsistency_prob = pulp.LpProblem("Find inconsistency", pulp.LpMinimize)
        self.comparisons = comparisons
        self.preference_info_ranking = Ranking(alternatives=dataset.alternative_names)
        self.preference_info_ranking.add_comparisons(comparisons)
        self.epsilon = epsilon

        self._value_functions_prob_variables = {}
        self._inconsistency_prob_variables = {}

    def _check_value_function_locations(self):
        """Check value_function.characteristic_points_locations and compare if they are within the range of the
        dataset"""
        for i, criterion in enumerate(self.dataset.criteria):
            min_dataset_i = self.dataset.data[:, i].min()
            max_dataset_i = self.dataset.data[:, i].max()
            for location in criterion.value_function.characteristic_points_locations:
                if location < min_dataset_i or location > max_dataset_i:
                    raise ValueError(f"{location} is not within the range of the dataset for criterion "
                                     f"{self.dataset.criteria[i].name}")
            if min_dataset_i not in criterion.value_function.characteristic_points_locations:
                raise ValueError(f"{min_dataset_i} is not in the dataset for criterion "
                                 f"{self.dataset.criteria[i].name}")
            if max_dataset_i not in criterion.value_function.characteristic_points_locations:
                raise ValueError(f"{max_dataset_i} is not in the dataset for criterion "
                                 f"{self.dataset.criteria[i].name}")

    @property
    def n_alternatives(self):
        return len(self.dataset.data.shape[0])

    @property
    def n_criteria(self):
        return len(self.dataset.data.shape[1])

    def find_minimal_inconsistent_subset(self) -> list[Comparison]:
        """Finds a minimal subset of constraints that need to be removed to restore consistency.

        The objective function consist on minimizing the sum of the binary variables that represent the
        comparisons between alternatives. We call these variables v_i. The index i is the position of the
        comparison in the list of self.comparisons. The value of the variable is 1 if the comparison
        is inconsistent and 0 otherwise.

        The specific constraints of this model can be represented as follows:
        U(a_1) > U(a_2) - v_1 if a_1 is preferred to a_2 by the decision maker. This is represented by:
        U(a_2) - U(a_1) - v_1 <= epsilon

        U(a_1) is the comprehensive value of alternative a_1. It is the sum of the value functions of all
        the criteria:

        U(a_1) = sum_{i=1}^{n} u_i(g_i(a_1)) where u_i is the value function of criterion i and g_i is the
        performance function of criterion (the value of the alternative in the criterion).

        Returns:
            A list of Comparison objects that need to be removed to restore consistency.
        """
        inconsistency_vars = pulp.LpVariable.dicts("inconsistency", range(len(self.comparisons)), cat="Binary")
        self._inconsistency_prob_variables.update(inconsistency_vars)

        # Objective function
        self.inconsistency_prob += pulp.lpSum(inconsistency_vars)

        # Constraints for inconsistent comparisons
        for idx, comparison in enumerate(self.comparisons):
            alternative_i = comparison.alternative_1
            alternative_j = comparison.alternative_2
            U_ai = self._get_comprehensive_value_equation(alternative_i)
            U_aj = self._get_comprehensive_value_equation(alternative_j)

            # Only "<=" comparisons are allowed
            if comparison.type == ComparisonType.PREFERENCE:
                self.inconsistency_prob += U_aj - U_ai - inconsistency_vars[idx] <= self.epsilon
            elif comparison.type == ComparisonType.INDIFFERENCE:
                self.inconsistency_prob += U_ai - U_aj - inconsistency_vars[idx] <= self.epsilon
                self.inconsistency_prob += U_aj - U_ai - inconsistency_vars[idx] <= self.epsilon

        self.inconsistency_prob.solve(pulp.GLPK())

        return self._get_inconsistent_comparisons()

    def _get_inconsistent_comparisons(self) -> list[Comparison]:
        """Creates the inconsistent comparisons based on the results of the inconsistency model.

        If the binary variable that represents the comparison between two alternatives is 1, then the comparison
        is inconsistent.

        This is an auxiliary function for the method find_minimal_inconsistent_subset that is
        called after solving the inconsistency model.

        Returns:
            A list of Comparison objects that need to be removed to restore consistency.
        """
        inconsistent_comparisons = []
        for idx, var in enumerate(self.inconsistency_prob.variables()):
            if var.varValue == 1:
                inconsistent_comparisons.append(self.comparisons[idx])
        return inconsistent_comparisons

    def update_value_functions(self) -> None:
        """Updates the value functions of the dataset based on the results of the model.

        The value functions are the values of the objective function for each alternative.
        """
        for criterion in self.dataset.criteria:
            values = [self.value_functions_prob.variables()[i].varValue
                      for i in range(len(self.dataset.alternative_names))]
            criterion.value_function.characteristic_points_values = values

    def _get_comprehensive_value_equation(self, alternative_idx) -> pulp.LpVariable:
        """Creates the comprehensive value equation.

        The comprehensive value equation is the objective function of the model. It is the sum of the
        value functions of all the criteria.

        Args:
            alternative_idx: The index of the alternative that we want to evaluate.
        """

    def _add_general_constraints(self, prob: pulp.LpProblem) -> None:
        """Adds the general constraints to the model.

        The general constraints are the ones that are common to both ordinal regression problems:
            * Normalization constraint:
                sum_{i=1}^{n} u_i(g_i(beta)) = 1 where beta is the best value of the dataset for criterion i.
            * Monotonicity constraint:
                u_i(x_j) <= u_i(x_{j+1}) where x_j is the value_function.characteristic_points_locations[j]
                of dataset.criteria[i].

            * Non-negativity constraint: All variables must be non-negative.

        This constraints will be used to find the value functions and the minimal inconsistent subset.

        Note that this method must be called after adding the value function into the model.

        Args:
            prob: The problem in which the constraints are added.
        """

    def get_comprehensive_values(self) -> dict[str, float]:
        """Gets the comprehensive values of the alternatives.

        They are computed as the sum of the value functions of each criterion.

        Returns:
            A dictionary with the comprehensive values of the alternatives.
        """
        comprehensive_values = {}
        for alternative in self.dataset.alternative_names:
            U_a = sum(self.dataset.criteria[k].value_function(self.dataset.data[alternative, k]) for k in
                      range(len(self.dataset.criteria)))
            comprehensive_values[alternative] = U_a
        return comprehensive_values

    def create_ranking(self) -> Ranking:
        """Creates a ranking based on the value functions of the alternatives.

        Returns:
            A ranking of the alternatives.
        """
        comprehensive_values = self.get_comprehensive_values()
        sorted_comprehensive_values = sorted(comprehensive_values.items(), key=lambda x: x[1], reverse=True)
        ranking_dict = {}
        rank = 1
        last_value = -1
        for alternative, value in sorted_comprehensive_values:
            ranking_dict[alternative] = rank
            if value == last_value:
                continue
            rank += 1
            last_value = value

        return Ranking.from_dict(ranking_dict)

    @staticmethod
    def print_model_results(prob) -> None:
        """Prints the results of the value functions model.

        The results are the value functions of the criteria.
        """
        print(f"Status: {pulp.LpStatus[prob.status]}")
        print("Variables:")
        for var in prob.variables():
            print(f"{var.name} = {var.varValue}")
        print(f"Objective function value: {pulp.value(prob.objective)}")
