import pulp
import pytest
import numpy as np

from src.decision_analysis.decision_making import Criterion, UTA, Dataset, Ranking


def test_pulp_glpk():
    """If this test fails, you need to install GLPK"""
    prob = pulp.LpProblem("test", pulp.LpMaximize)
    x = pulp.LpVariable("x", lowBound=0)
    y = pulp.LpVariable("y", lowBound=0)
    prob += 2 * x + y
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
    return dataset


def get_ranking1():
    ranking_dict = {"X": 1, "Y": 2, "Z": 3}
    return Ranking.from_dict(ranking_dict)


def get_ranking2():
    ranking_dict = {"X": 1, "Y": 3, "Z": 2}
    return Ranking.from_dict(ranking_dict)


@pytest.mark.xfail
def test_find_minimal_inconsistent_dataset_empty():
    dataset = get_simple_example_dataset()
    ranking = get_ranking1()
    uta = UTA(dataset, )
    assert uta.find_minimal_inconsistent_subset(ranking) == []
