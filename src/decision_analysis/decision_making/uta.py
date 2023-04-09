from pulp import LpProblem, LpVariable, LpMinimize, LpConstraint, lpSum, GLPK
import pulp as pl
import numpy as np

from src.decision_analysis.decision_making import Dataset


class UTA:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

        self._model = pl.LpProblem("UTA", pl.LpMinimize)

        # POSIBLE MEJORA: Es una cuestión de estilo, pero creo que usar los atributos de dataset
        # directamente hace que el código sea más conciso, legible y evita errores como el que he
        # corregido: `self.alternatives = [criterion.name for criterion in self.criteria]` en lugar
        # de `self.alternatives = dataset.alternative_names`.
        self.data = dataset.data
        self.criteria = dataset.criteria
        self.criteria_types = [criterion.type for criterion in self.criteria]
        self.alternatives = dataset.alternative_names
        self.preference_relations = dataset.preference_relations
        self.indifference_relations = dataset.indifference_relations
        self.n_criteria = len(self.criteria)
        self.n_alternatives = len(self.alternatives)

        # POSIBLE MEJORA: Creo que es mucho más sencillo usar np.min y np.max, y no tener que hacer
        # un if para cada criterio. Luego, al crear las constraints, se puede meter un if para
        # ver si el criterio es de tipo -1 o 1, y así crear la constraint adecuada.
        self.min_values = np.array(
            [min(self.data[:, i]) if self.criteria_types[i] == 1 else max(self.data[:, i]) for i in
             range(self.n_criteria)])
        self.max_values = np.array(
            [max(self.data[:, i]) if self.criteria_types[i] == 1 else min(self.data[:, i]) for i in
             range(self.n_criteria)])
        self.value_function_characteristic_points_locations = [criterion.value_function_characteristic_points_location
                                                               for criterion in self.criteria]
        self.splices_list = [
            [self.min_values[i]] + list((np.array(self.value_function_characteristic_points_locations) - self.min_values[i])
                                    / (self.max_values[i] - self.min_values[i])) +
            [self.max_values[i]] for i in range(self.n_criteria)]

        # POSIBLES MEJORAS:
        # Sugerencia de GPT-4:
        # Inconsistent array dimensions in self.splices_list:
        # The expression (np.array(self.value_function_characteristic_points_locations) - self.min_values[i])
        # / (self.max_values[i] - self.min_values[i]) might have unintended consequences due to the
        # dimensions of the arrays. You should loop through the criteria and calculate the corresponding
        # values, like this:
        # self.splices_list = [
        #     [self.min_values[i]] +
        #     [(self.value_function_characteristic_points_locations[i][j] - self.min_values[i]) / (
        #                 self.max_values[i] - self.min_values[i])
        #      for j in range(len(self.value_function_characteristic_points_locations[i]))] +
        #     [self.max_values[i]] for i in range(self.n_criteria)
        # ]
        # Sugerencia mía:
        # No usar list comprehensions para cosas tan largas, es más fácil de leer y de depurar si se hace
        # con un bucle for.
        # POSIBLE MEJORA 2:
        # Si el atributo no es de utilidad al usuario (nosotros), hacerlos privados.
        # Evitar crear atributos si solo se van a usar en un método.

        self.n_splices = [len(self.splices_list[i]) for i in range(self.n_criteria)]
        self.u_splices_list = [
            [pl.LpVariable(f"u_{i + 1}({self.splices_list[i][j]})", lowBound=0, cat="Continuous") for j in
             range(self.n_splices[i])]
            for i in range(self.n_criteria)]

        self.overestimation = {
            self.alternatives[i]: pl.LpVariable(f"σ_plus_{self.alternatives[i]}", lowBound=0, cat="Continuous") for
            i in range(self.n_alternatives)}
        self.underestimation = {
            self.alternatives[i]: pl.LpVariable(f"σ_minus_{self.alternatives[i]}", lowBound=0, cat="Continuous")
            for i in range(self.n_alternatives)}

    def update_criteria_value_functions(self) -> float:
        """Updates the `value_function` of each criterion by solving the ordinal regression problem.
        
        We wish to find an additive value function that: reproduces the DM’s pairwise comparisons, is normalized
        to the interval between 0 and 1, and employs monotonic marginal value functions. The value function
        is represented by a set of characteristic points, which are the values of the function at certain
        points. By doing linear interpolation between these points, we can obtain the value of the function
        at any point.

        Returns:
            The value of the objective function of the LP problem which corresponds to the sum of over-
            and underestimation errors. If the value is 0, the problem has a feasible solution.
        """

    def find_minimal_inconsistent_subset(self) -> list[tuple[str, str]]:
        """Finds a minimal subset of constraints that need to be removed to restore consistency.

        This method solves a linear programming problem similar to the one used to solve the ordinal
        regression problem to find the value functions. The main difference is that the objective function is now to
        minimize the sum of binary variables which are 1 if the corresponding preference relation has to be removed
        from the preference relations to obtain a consistent set, and 0 otherwise. These binary variables substitute
        the overestimation and underestimation variables used to solve the mentioned ordinal regression problem.

        Returns:
            A list of tuples containing the names of the alternatives that are involved in the preference relations
            that have to be removed to obtain a consistent set of constraints. If the list is empty, the set of
            constraints is consistent.
        """

    def _add_objective_function(self):
        self._model += pl.lpSum(
            [self.overestimation[self.alternatives[i]] + self.underestimation[self.alternatives[i]] for i in
             range(self.n_alternatives)])

    def _add_constraints(self):
        for alt1, alt2 in self.preference_relations:
            self._model += pl.lpSum([self._get_u_value(alt1, i) - self._get_u_value(alt2, i) for i in range(self.n_criteria)]) \
                           - self.overestimation[alt1] + self.underestimation[alt1] + self.overestimation[alt2] - self.underestimation[
                         alt2] >= 1e-6

        for alt1, alt2 in self.indifference_relations:
            self._model += pl.lpSum([self._get_u_value(alt1, i) - self._get_u_value(alt2, i) for i in range(self.n_criteria)]) \
                           - self.overestimation[alt1] + self.underestimation[alt1] + self.overestimation[alt2] - self.underestimation[alt2] == 0

    @staticmethod
    def _interpolate(x, x0, x1, y0, y1):
        return (y0 * (x1 - x) + y1 * (x - x0)) / (x1 - x0)

    def _get_u_value(self, alternative, criteria):
        for i in range(self.n_splices[criteria] - 1):
            if self.splices_list[criteria][i] <= self.data[alternative, criteria] <= self.splices_list[criteria][i + 1]:
                return self._interpolate(self.data[alternative, criteria], self.splices_list[criteria][i],
                                         self.splices_list[criteria][i + 1],
                                         self.u_splices_list[criteria][i], self.u_splices_list[criteria][i + 1])

    def _solve(self):
        self._model.solve(pl.GLPK())
        self.solution = {v.name: v.varValue for v in self._model.variables()}
