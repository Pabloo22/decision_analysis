import numpy as np
import pytest

from src.decision_analysis.decision_making import Ranking


def ranking1():
    matrix = np.array([[0, 1, 1],
                       [0, 0, 1],
                       [0, 0, 0]])
    return Ranking(matrix=matrix, alternatives=['E', 'B', 'I'])


def ranking2():
    matrix = np.array([[0, 1, 1],
                       [0, 0, 0.5],
                       [0, 0.5, 0]])
    return Ranking(matrix=matrix, alternatives=['E', 'B', 'I'])


def ranking3():
    matrix = np.array([[0, 0, 0],
                       [1, 0, 0],
                       [1, 1, 0]])
    return Ranking(matrix=matrix, alternatives=['E', 'B', 'I'])


@pytest.mark.parametrize("ranking_dict, expected", [
    ({"E": 1, "B": 2, "I": 3}, ranking1()),
    ({"E": 1, "B": 2, "I": 2}, ranking2()),
    ({"E": 1, "B": 3, "I": 3}, ranking2()),
    ({"E": 3, "B": 2, "I": 1}, ranking3()),
])
def test_from_dict(ranking_dict, expected):
    assert np.allclose(Ranking.from_dict(ranking_dict).matrix, expected.matrix)


@pytest.mark.parametrize("ranking_a, ranking_b, expected", [
    (ranking1(), ranking1(), 0),
    (ranking1(), ranking2(), 0.5),
    (ranking2(), ranking1(), 0.5),
    (ranking1(), ranking3(), 3),
    (ranking3(), ranking1(), 3),
])
def test_kendall_distance(ranking_a, ranking_b, expected):
    assert ranking_a.kendall_distance(ranking_b) == expected


@pytest.mark.parametrize("ranking_a, ranking_b, expected", [
    (ranking1(), ranking1(), 1),
    (ranking1(), ranking2(), pytest.approx(2 / 3)),
    (ranking2(), ranking1(), pytest.approx(2 / 3)),
    (ranking1(), ranking3(), -1),
    (ranking3(), ranking1(), -1),
])
def test_kendall_tau(ranking_a, ranking_b, expected):
    assert ranking_a.kendall_tau(ranking_b) == expected


if __name__ == "__main__":
    pytest.main()
