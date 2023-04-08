from pulp import LpProblem, LpVariable, LpMinimize, LpConstraint, lpSum, GLPK
import pulp

from src.decision_analysis.decision_making import Dataset


class UTA:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.lp_model = None
        self.solution = None

    def run(self):
        self._create_lp_model()
        self._solve()
        return self.solution

    def _create_lp_model(self):
        model = pulp.LpProblem("UTA", pulp.LpMinimize)
        n_alternatives, n_criteria = self.dataset.data.shape

        # Create weight and preference difference variables
        weights = [pulp.LpVariable(f'w_{i}', lowBound=0) for i in range(n_criteria)]
        pref_diff_vars = [pulp.LpVariable(f'p_{i}_{j}', lowBound=0)
                          for i in range(n_alternatives)
                          for j in range(n_alternatives) if i != j]

        # Objective function
        model += pulp.lpSum(pref_diff_vars)

        # Constraints
        for alt1, alt2 in self.dataset.preference_relations:
            for i in range(n_criteria):
                pref_diff = self.dataset.criteria[i].type * (self.dataset.data[alt1, i] - self.dataset.data[alt2, i])
                model += pulp.LpConstraint(weights[i] * pref_diff >= pref_diff_vars[alt1 * n_alternatives + alt2])

        for alt1, alt2 in self.dataset.indifference_relations:
            for i in range(n_criteria):
                pref_diff = self.dataset.criteria[i].type * (self.dataset.data[alt1, i] - self.dataset.data[alt2, i])
                model += pulp.LpConstraint(weights[i] * pref_diff == 0)

        # Normalization constraint
        model += pulp.lpSum(weights) == 1

        self.lp_model = model

    def _solve(self):
        if self.lp_model is None:
            self._create_lp_model()

        self.lp_model.solve(pulp.GLPK())
        self.solution = {v.name: v.varValue for v in self.lp_model.variables()}
