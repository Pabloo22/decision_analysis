import numpy as np
import pytest

from src.decision_analysis.decision_making import Ranking, Comparison, ComparisonType


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


def test_ranking_from_dict2():
    ranking_dict = {'x': 1, 'y': 2, 'z': 2}
    ranking = Ranking.from_dict(ranking_dict)
    assert ranking.n_alternatives == 3
    assert ranking.alternative_names == ['x', 'y', 'z']
    expected_matrix = np.array([[0, 1, 1],
                                [0, 0, 0.5],
                                [0, 0.5, 0]])
    assert np.array_equal(ranking.matrix, expected_matrix)


def test_ranking_from_comparisons():
    comparisons = [
        Comparison(0, 1, ComparisonType.PREFERENCE),
        Comparison(1, 2, ComparisonType.INDIFFERENCE),
    ]
    ranking = Ranking(alternatives=3)
    ranking.add_comparisons(comparisons)
    expected_matrix = np.array([[0, 1, 0], [0, 0, 0.5], [0, 0.5, 0]])
    assert np.array_equal(ranking.matrix, expected_matrix)


def test_ranking_get_comparisons():
    matrix = np.array([[0, 1, 0],
                       [0, 0, 0.5],
                       [0, 0.5, 0]])
    ranking = Ranking(matrix=matrix)
    comparisons = ranking.get_comparisons()
    expected_comparisons = [
        Comparison(0, 1, ComparisonType.PREFERENCE),
        Comparison(1, 2, ComparisonType.INDIFFERENCE),
    ]
    assert comparisons == expected_comparisons


def test_ranking_remove_comparisons():
    matrix = np.array([[0, 1, 0], [0, 0, 0.5], [0, 0.5, 0]])
    ranking = Ranking(matrix=matrix)
    ranking.remove_comparisons([
        Comparison(0, 1, ComparisonType.PREFERENCE),
        Comparison(1, 2, ComparisonType.INDIFFERENCE),
    ])
    assert np.array_equal(ranking.matrix, np.zeros((3, 3)))


def test_init_without_arguments():
    with pytest.raises(ValueError):
        Ranking()


def test_ranking_init_with_alternatives():
    ranking = Ranking(alternatives=3)
    assert ranking.n_alternatives == 3
    assert ranking.alternative_names == ['a_1', 'a_2', 'a_3']
    assert np.array_equal(ranking.matrix, np.zeros((3, 3)))


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
