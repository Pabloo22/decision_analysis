from pulp import LpProblem, LpVariable, LpMinimize, LpConstraint, lpSum, GLPK
import pulp as pl
import numpy as np

from src.decision_analysis.decision_making import Dataset


class UTA:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

        self.data = dataset.data
        self.criteria = dataset.criteria
        self.criteria_types = [criterion.type for criterion in self.criteria]
        self.alternatives = [criterion.name for criterion in self.criteria]
        self.preference_relations = dataset.preference_relations
        self.indifference_relations = dataset.indifference_relations
        self.n_criteria = len(self.criteria)
        self.n_alternatives = len(self.alternatives)
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
        self.model = pl.LpProblem("UTA", pl.LpMinimize)
        self.solution = None

    def run(self):
        self._add_objective_function()
        self._add_constraints()
        self._solve()
        return self.solution

    def _add_objective_function(self):
        self.model += pl.lpSum(
            [self.overestimation[self.alternatives[i]] + self.underestimation[self.alternatives[i]] for i in
             range(self.n_alternatives)])

    def _add_constraints(self):
        for alt1, alt2 in self.preference_relations:
            self.model += pl.lpSum([self._get_u_value(alt1, i) - self._get_u_value(alt2, i) for i in range(self.n_criteria)]) \
                     - self.overestimation[alt1] + self.underestimation[alt1] + self.overestimation[alt2] - self.underestimation[
                         alt2] >= 1e-6

        for alt1, alt2 in self.indifference_relations:
            self.model += pl.lpSum([self._get_u_value(alt1, i) - self._get_u_value(alt2, i) for i in range(self.n_criteria)]) \
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
        self.model.solve(pl.GLPK())
        self.solution = {v.name: v.varValue for v in self.model.variables()}
