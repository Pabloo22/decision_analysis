import pulp
import pytest

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


def test_value_function_instantiation():
    vf = ValueFunction(characteristic_points_locations=[0, 5, 10])
    assert vf.characteristic_points_locations == [0, 5, 10]
    assert vf.characteristic_points_values is None


def test_value_function_n_break_points():
    vf = ValueFunction(characteristic_points_locations=[0, 5, 10])
    assert vf.n_break_points == 1


@pytest.mark.parametrize("value, expected", [
    (0, 0),
    (5, 5),
    (2.5, 2.5),
    (10, 10),
])
def test_value_function_call(value, expected):
    vf = ValueFunction(
        characteristic_points_locations=[0, 5, 10],
        characteristic_points_values=[0, 5, 10]
    )
    assert vf(value) == expected


def test_value_function_call_error():
    vf = ValueFunction(characteristic_points_locations=[0, 5, 10])
    with pytest.raises(ValueError,
                       match="The characteristic points values must be set before calling the value function"):
        vf(2.5)


@pytest.mark.parametrize("characteristic_points_values, expected", [
    ([1, 2, 3, 4, 5], True),
    ([5, 4, 3, 2, 1], True),
    ([3, 3, 3, 3, 3], True),
    ([1, 2, 3, 2, 1], False),
    ([1, 2, 3, 3, 4], True),
])
def test_is_monotonic(characteristic_points_values, expected):
    assert ValueFunction.is_monotonic(characteristic_points_values) == expected


if __name__ == "__main__":
    pytest.main()
