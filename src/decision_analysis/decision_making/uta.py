import pulp
import numpy as np

from decision_analysis.decision_making import Dataset, Ranking, Comparison, ComparisonType, ValueFunction
from decision_analysis.decision_making.data_structures import Criterion


class UTA:
    """Class that implements the UTA method.

    It uses linear programming in order to optimally infer additive value functions, so that these functions are
    as consistent as possible with the global decision-maker’s preferences.

    The optimal value functions are found by inferring the value of some specific breakpoints of the value
    functions. These breakpoints are defined when creating the criteria of the dataset by setting
    the characteristic_points_locations attribute of the value function for each criterion.

    This class also implements the method to find the minimal inconsistent subset of comparisons.

    Attributes:
        dataset: The dataset that we want to analyze.
        comparisons: The comparisons made by the decision maker.
    """

    def __init__(self, dataset: Dataset, comparisons: list[Comparison]):
        self.dataset = dataset

        self.prob = pulp.LpProblem("LinearProblem", pulp.LpMinimize)

        self.comparisons = comparisons

        self._epsilon = 1e-4
        self._prob_variables = {}

        self._min_values = []
        self._max_values = []

        for i, criterion in enumerate(self.dataset.criteria):
            self._min_values.append(self.dataset.data[:, i].min())
            self._max_values.append(self.dataset.data[:, i].max())

        self._check_value_function_locations()

    def _check_value_function_locations(self) -> None:
        """Check value_function.characteristic_points_locations and compare if they are within the range of the
        dataset.

        Raises:
            ValueError: If the characteristic points locations are not within the range of the dataset.
        """
        for i, criterion in enumerate(self.dataset.criteria):
            if self._min_values[i] not in criterion.value_function.characteristic_points_locations:
                raise ValueError(f"{self._min_values[i]} is not in the characteristic points locations of "
                                 f"{criterion.name}")
            if self._max_values[i] not in criterion.value_function.characteristic_points_locations:
                raise ValueError(f"{self._max_values[i]} is not in the characteristic points locations of "
                                 f"{criterion.name}")

            for location in criterion.value_function.characteristic_points_locations:
                if location < self._min_values[i] or location > self._max_values[i]:
                    raise ValueError(f"{location} is not within the range of the dataset for criterion "
                                     f"{self.dataset.criteria[i].name}")

    @property
    def n_alternatives(self) -> int:
        return len(self.dataset.data.shape[0])

    @property
    def n_criteria(self) -> int:
        return len(self.dataset.data.shape[1])

    def solve(self) -> None:
        """Finds a minimal subset of constraints that need to be removed to restore consistency.

        The objective function consist of minimizing the sum of the binary variables that represent the
        comparisons between alternatives. We call these variables v_i. The index i is the position of the
        comparison in the list of self.comparisons. The value of the variable is 1 if the comparison
        is inconsistent and 0 otherwise.

        The specific constraints of this model can be represented as follows:
        U(a_1) > U(a_2) - v_1 if a_1 is preferred to a_2 by the decision maker. This is represented by:
        U(a_1) >= U(a_2) - v_1 + epsilon
        U(a_2) - U(a_1) - v_1 <= -epsilon

        U(a_1) is the comprehensive value of alternative a_1. It is the sum of the value functions of all
        the criteria:

        U(a_1) = sum_{i=1}^{n} u_i(g_i(a_1)) where u_i is the value function of criterion i and g_i is the
        performance function of criterion (the value of the alternative in the criterion).

        Returns:
            A list of Comparison objects that need to be removed to restore consistency.
        """
        inconsistency_vars = [pulp.LpVariable(f"v_{i}", cat="Binary")
                              for i in range(1, len(self.comparisons) + 1)]

        # Add the variables to the dictionary
        self._prob_variables = {var.name: var for var in inconsistency_vars}
        for i, criterion in enumerate(self.dataset.criteria, start=1):
            for location in criterion.value_function.characteristic_points_locations:
                self._prob_variables[f"u_{i}({location})"] = pulp.LpVariable(f"u_{i}({location})",
                                                                             lowBound=0,
                                                                             upBound=1)

        # Objective function
        self.prob += pulp.lpSum(inconsistency_vars)

        # Constraints for inconsistent comparisons
        for idx, comparison in enumerate(self.comparisons, start=1):
            alternative_i = comparison.alternative_1
            alternative_j = comparison.alternative_2
            U_ai = self._get_comprehensive_value_equation(alternative_i, self._prob_variables)
            U_aj = self._get_comprehensive_value_equation(alternative_j, self._prob_variables)

            # Only "<=" comparisons are allowed
            if comparison.type == ComparisonType.PREFERENCE:
                self.prob += U_aj - U_ai - inconsistency_vars[idx - 1] <= -self._epsilon
            elif comparison.type == ComparisonType.INDIFFERENCE:
                self.prob += U_ai - U_aj - inconsistency_vars[idx - 1] <= 0
                self.prob += U_aj - U_ai - inconsistency_vars[idx - 1] <= 0
            else:
                raise ValueError(f"Comparison type {comparison.type} is not supported.")

        self._add_general_constraints(self.prob, self._prob_variables)

        self.prob.solve(pulp.GLPK(msg=False))

    def get_inconsistent_comparisons(self) -> list[Comparison]:
        """Creates the inconsistent comparisons based on the results of the problem.

        If the binary variable that represents the comparison between two alternatives is 1, then the comparison
        is inconsistent.

        This is an auxiliary function for the method find_minimal_inconsistent_subset that is
        called after solving the inconsistency model.

        Returns:
            A list of Comparison objects that need to be removed to restore consistency.
        """
        inconsistent_comparisons = []
        v_variables = [variable for (name, variable) in self._prob_variables.items()
                       if name.startswith("v_")]
        for i, variable in enumerate(v_variables):
            if variable.varValue == 1:
                inconsistent_comparisons.append(self.comparisons[i])

        return inconsistent_comparisons

    def update_value_functions(self) -> None:
        """Updates the value functions of the dataset based on the results of the model.

        The value functions are the values of the objective function for each alternative.
        """
        for i, criterion in enumerate(self.dataset.criteria, start=1):
            values = [self._prob_variables[f"u_{i}({location})"].varValue
                      for location in criterion.value_function.characteristic_points_locations]
            criterion.value_function.characteristic_points_values = values

    def _get_comprehensive_value_equation(self,
                                          alternative_idx: int,
                                          variables: dict[str, pulp.LpVariable]) -> pulp.LpVariable:
        """Creates the comprehensive value equation.

        The comprehensive value equation is the objective function of the model. It is the sum of the
        value functions of all the criteria.

        Args:
            alternative_idx: The index of the alternative that we want to evaluate.
        """
        affine_expressions = []
        for i, criterion in enumerate(self.dataset.criteria, start=1):
            locations = criterion.value_function.characteristic_points_locations
            values = [variables[f"u_{i}({location})"] for location in locations]
            characteristic_points = list(zip(locations, values))

            x = self.dataset.data[alternative_idx, i - 1]

            affine_expressions.append(ValueFunction.piecewise_linear_interpolation(x, characteristic_points))

        return pulp.lpSum(affine_expressions)

    def _add_general_constraints(self, prob: pulp.LpProblem, variables: dict[str, pulp.LpVariable]) -> None:
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
        norm_constraints = []
        for i, criterion in enumerate(self.dataset.criteria, start=1):
            locations = criterion.value_function.characteristic_points_locations

            # Normalization constraint
            if criterion.is_gain():
                norm_constraints.append(variables[f"u_{i}({max(locations)})"])
                prob += variables[f"u_{i}({min(locations)})"] == 0
            else:
                norm_constraints.append(variables[f"u_{i}({min(locations)})"])
                prob += variables[f"u_{i}({max(locations)})"] == 0

            # Monotonicity constraint
            for j in range(len(locations) - 1):
                if criterion.is_gain():
                    prob += variables[f"u_{i}({locations[j + 1]})"] >= variables[f"u_{i}({locations[j]})"]
                else:
                    prob += variables[f"u_{i}({locations[j + 1]})"] <= variables[f"u_{i}({locations[j]})"]

        prob += pulp.lpSum(norm_constraints) == 1

        # Non-negativity constraint
        for var in variables.values():
            prob += var >= 0

    def get_comprehensive_values(self) -> dict[str, float]:
        """Gets the comprehensive values of the alternatives.

        They are computed as the sum of the value functions of each criterion.

        Returns:
            A dictionary with the comprehensive values of the alternatives.
        """
        comprehensive_values = {}
        for i, alternative in enumerate(self.dataset.alternative_names):
            U_a = sum(self.dataset.criteria[k].value_function(self.dataset.data[i, k]) for k in
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


if __name__ == "__main__":
    data = np.array([[5, 0, 10],  # B
                     [10, 5, 15],  # E
                     [0, 10, 12.5]])  # I
    criteria = [Criterion(name='g1', type=1, value_function=ValueFunction([0, 5, 10])),
                Criterion(name='g2', type=1, value_function=ValueFunction([0, 5, 10])),
                Criterion(name='g3', type=-1, value_function=ValueFunction([10, 12.5, 15]))]
    dataset = Dataset(data, criteria, ['B', 'E', 'I'])
    idx_dict = {'B': 0, 'E': 1, 'I': 2}

    comparisons = [Comparison(idx_dict['E'], idx_dict['B'], ComparisonType.PREFERENCE),
                   # Comparison(idx_dict['E'], idx_dict['I'], ComparisonType.PREFERENCE),
                   Comparison(idx_dict['B'], idx_dict['I'], ComparisonType.PREFERENCE)]
    """min v_{E,B} + v_{B,I} 
    s.t.
    U(E) > U(B) - v_{E,B} => U(E) - U(B) + v_{E,B} >= epsilon =>
    => u_1(10) - u_1(5) + u_2(5) - u_2(0) + u_3(15) - u_3(10) - v_{E,B} >= epsilon
    U(B) > U(I) - v_{B,I} => U(B) - U(I) + v_{B,I} >= epsilon =>
    => u_1(5) - u_1(0) + u_2(0) - u_2(10) + u_3(10) - u_3(12.5) - v_{B,I} >= epsilon
    sum_{i=1}^{n} u_i(g_i(beta)) = 1 for all i => u_1(10) + u_2(10) + u_3(10) = 1
    u_i(g_i(alpha)) = 0 for all i => 
    1.- u_1(0) = 0
    2.- u_2(0) = 0
    3.- u_3(15) = 0
    u_i(x_j) <= u_i(x_{j+1}) for all i, j if g_i is a gain criterion =>
    1.- u_1(5) - u_1(0) >= 0
    2.- u_1(10) - u_1(5) >= 0
    3.- u_2(5) - u_2(0) >= 0
    4.- u_2(10) - u_2(5) >= 0
    u_i(x_j) >= u_i(x_{j+1}) for all i, j if g_i is a cost criterion =>
    1.- u_3(10) - u_3(12.5) >= 0
    2.- u_3(12.5) - u_3(15) >= 0
    u_i(x_j) >= 0 for all i, j
    """

    uta = UTA(dataset, comparisons)
    uta.solve()
    UTA.print_model_results(uta.prob)

    # --------------------------------------

    def get_simple_example_dataset():
        alternatives = ["X", "Y", "Z"]
        criteria = [
            Criterion(type=1, name="g_1", value_function=ValueFunction([0, 10])),
            Criterion(type=1, name="g_2", value_function=ValueFunction([0, 10])),
        ]
        data = np.array([[10, 0],
                         [0, 10],
                         [5, 5]])

        dataset = Dataset(data, criteria, alternatives)
        return dataset


    def get_ranking1():
        ranking_dict = {"X": 1, "Y": 2, "Z": 3}
        return Ranking.from_dict(ranking_dict)


    def get_ranking2():
        ranking_dict = {"X": 1, "Y": 3, "Z": 2}
        return Ranking.from_dict(ranking_dict)


    def simple_example():
        dataset = get_simple_example_dataset()
        ranking = get_ranking1()
        comparisons = ranking.get_comparisons()
        uta = UTA(dataset, comparisons)
        uta.solve()
        UTA.print_model_results(uta.prob)

    print("\nSimple example")
    print("---")
    simple_example()
