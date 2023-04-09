import pulp
from src.decision_analysis.decision_making import ValueFunction


def test_linear_interpolation():
    pulp_vars = {0: pulp.LpVariable("u_1(0)", lowBound=0),
                 5: pulp.LpVariable("u_1(1)", lowBound=0),
                 10: pulp.LpVariable("u_1(10)", lowBound=0)}

    assert ValueFunction.linear_interpolation(0, 0, 5, pulp_vars[0], pulp_vars[5]) == pulp_vars[0]
    assert ValueFunction.linear_interpolation(5, 0, 5, pulp_vars[0], pulp_vars[5]) == pulp_vars[5]
    assert ValueFunction.linear_interpolation(2.5, 0, 5, pulp_vars[0], pulp_vars[5]) == 0.5 * pulp_vars[0] + 0.5 * \
           pulp_vars[5]
    assert ValueFunction.linear_interpolation(2.5, 0, 10, pulp_vars[0], pulp_vars[10]) == 0.25 * pulp_vars[0] + 0.75 * \
           pulp_vars[10]
    assert ValueFunction.linear_interpolation(7.5, 0, 10, pulp_vars[0], pulp_vars[10]) == 0.75 * pulp_vars[0] + 0.25 * \
           pulp_vars[10]
    assert ValueFunction.linear_interpolation(7.5, 5, 10, pulp_vars[5], pulp_vars[10]) == 0.5 * pulp_vars[5] + 0.5 * \
           pulp_vars[10]