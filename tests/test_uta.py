import pulp
import numpy as np

from src.decision_analysis.decision_making import Criterion, UTA, Dataset


def test_pulp_glpk():
    """If this test fails, you need to install GLPK"""
    prob = pulp.LpProblem("test", pulp.LpMaximize)
    x = pulp.LpVariable("x", lowBound=0)
    y = pulp.LpVariable("y", lowBound=0)
    prob += 2*x + y
    prob += x + y <= 1
    prob += x <= 0.5
    prob.solve(pulp.GLPK())
    assert pulp.LpStatus[prob.status] == "Optimal"
    assert x.value() == 0.5
    assert y.value() == 0.5


def get_simple_example_dataset():
    alternatives = ["X", "Y", "Z"]
    criteria = [
        Criterion(type=1, name="g_1"),
        Criterion(type=1, name="g_2"),
    ]
    data = np.array([[10, 0],
                     [0, 10],
                     [5, 5]])

    dataset = Dataset(data, criteria, alternatives)
    dataset.add_preference(0, 1)
    dataset.add_preference(0, 2)

    return dataset


def test_simple_example():
    dataset = get_simple_example_dataset()
    uta = UTA(dataset)
    solution = uta.run()

    assert solution == {'w_0': 0.5, 'w_1': 0.5,
                        'p_0_1': 0.0, 'p_0_2': 0.0, 'p_1_0': 0.0, 'p_1_2': 0.0,
                        'p_2_0': 0.0, 'p_2_1': 0.0}

