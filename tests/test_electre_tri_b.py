import pytest
import numpy as np

from src.decision_analysis.decision_making import Criterion, Alternative, ElectreTriB


def _electre_exercise_1():
    # Instantiate Criterion objects
    g1 = Criterion(criteria_type=1,
                   weight=2,
                   preference_threshold=50,
                   indifference_threshold=10,
                   veto_threshold=100,
                   name='g1')
    g2 = Criterion(criteria_type=-1,
                   weight=3,
                   preference_threshold=10,
                   indifference_threshold=0,
                   veto_threshold=20,
                   name='g2')

    # Create a list of criteria
    criteria = [g1, g2]

    # Instantiate Alternative objects
    a = Alternative({"g1": 145, "g2": 40})
    e = Alternative({"g1": 240, "g2": 20})
    bt = Alternative({"g1": 150, "g2": 15})

    # Instantiate ElectreTriB object
    electre = ElectreTriB(criteria, [a, e], profiles=[bt], credibility_threshold=0.6)

    return electre


def test_calculate_marginal_concordance():
    electre = _electre_exercise_1()
    a, e = electre.alternatives
    g1, g2 = electre.criteria
    bt = electre.boundaries[0]

    assert electre.get_marginal_concordance(a, bt, g1) == 1
    assert electre.get_marginal_concordance(a, bt, g2) == 0
    assert electre.get_marginal_concordance(e, bt, g1) == 1
    assert electre.get_marginal_concordance(e, bt, g2) == 0.5
    assert electre.get_marginal_concordance(bt, a, g1) == 1
    assert electre.get_marginal_concordance(bt, a, g2) == 1
    assert electre.get_marginal_concordance(bt, e, g1) == 0
    assert electre.get_marginal_concordance(bt, e, g2) == 1


def test_calculate_marginal_discordance():
    electre = _electre_exercise_1()
    a, e = electre.alternatives
    g1, g2 = electre.criteria
    bt = electre.boundaries[0]

    assert electre.get_marginal_discordance(a, bt, g1) == 0
    assert electre.get_marginal_discordance(a, bt, g2) == 1
    assert electre.get_marginal_discordance(e, bt, g1) == 0
    assert electre.get_marginal_discordance(e, bt, g2) == 0
    assert electre.get_marginal_discordance(bt, a, g1) == 0
    assert electre.get_marginal_discordance(bt, a, g2) == 0
    assert electre.get_marginal_discordance(bt, e, g1) == 0.8
    assert electre.get_marginal_discordance(bt, e, g2) == 0

    electre.run()
    assert electre.discordance_tensor_alt_bound[0, 0, 0] == 0
    assert electre.discordance_tensor_alt_bound[0, 0, 1] == 1
    assert electre.discordance_tensor_alt_bound[1, 0, 0] == 0
    assert electre.discordance_tensor_alt_bound[1, 0, 1] == 0
    assert electre.discordance_tensor_bound_alt[0, 0, 0] == 0
    assert electre.discordance_tensor_bound_alt[0, 0, 1] == 0
    assert electre.discordance_tensor_bound_alt[1, 0, 0] == 0.8
    assert electre.discordance_tensor_bound_alt[1, 0, 1] == 0


def test__is_alt_bound():
    electre = _electre_exercise_1()
    a, e = electre.alternatives
    bt = electre.boundaries[0]

    assert electre._is_alt_bound(a, bt) is True
    assert electre._is_alt_bound(e, bt) is True
    assert electre._is_alt_bound(bt, a) is False
    assert electre._is_alt_bound(bt, e) is False

    c = Alternative({"g1": 100, "g2": 100})
    with pytest.raises(ValueError):
        electre._is_alt_bound(c, c)
    with pytest.raises(ValueError):
        electre._is_alt_bound(c, bt)
    with pytest.raises(ValueError):
        electre._is_alt_bound(bt, c)


def test_get_comprehensive_concordance():
    electre = _electre_exercise_1()
    a, e = electre.alternatives
    bt = electre.boundaries[0]

    assert electre.get_comprehensive_concordance(a, bt) == 0.4
    assert electre.get_comprehensive_concordance(e, bt) == 0.7
    assert electre.get_comprehensive_concordance(bt, a) == 1
    assert electre.get_comprehensive_concordance(bt, e) == 0.6

    assert electre.comprehensive_concordance_matrix_alt_bound[0, 0] == 0.4
    assert electre.comprehensive_concordance_matrix_alt_bound[1, 0] == 0.7
    assert electre.comprehensive_concordance_matrix_bound_alt[0, 0] == 1
    assert electre.comprehensive_concordance_matrix_bound_alt[1, 0] == 0.6


def test_get_outranking_credibility():
    electre = _electre_exercise_1()
    a, e = electre.alternatives
    bt = electre.boundaries[0]

    assert electre.get_outranking_credibility(a, bt) == 0
    assert electre.get_outranking_credibility(e, bt) == 0.7
    assert electre.get_outranking_credibility(bt, a) == 1
    assert electre.get_outranking_credibility(bt, e) == pytest.approx(0.3)


def test_get_outranking_relation():
    electre = _electre_exercise_1()
    a, e = electre.alternatives
    bt = electre.boundaries[0]

    assert electre.get_outranking_relation(a, bt) == '<'
    assert electre.get_outranking_relation(e, bt) == '>'


def test_calculate_class_assignments():
    # relations = {-1: '?', 0: '<', 1: '>', 0.5: '='}
    outranking_matrix = np.array(
        [[1, 1, 0, 0, 0],
         [1, -1, 0, 0, 0],
         [1, 1, 0.5, 0.5, 0],
         [1, -1, -1, -1, 0],
         [1, 1, 1, 1, 0],
         [0.5, 0, 0, 0, 0],
         [1, 1, 0.5, 0, 0],
         [1, 1, 0, 0, 0],
         [1, -1, -1, 0, 0],
         [1, 1, 0.5, 0.5, 0]]
    )

    optimistic_class_assignments, pessimistic_class_assignments = \
        ElectreTriB.calculate_class_assignments(outranking_matrix)

    assert np.array_equal(optimistic_class_assignments, np.array([2, 2, 4, 4, 4, 1, 3, 2, 3, 4], dtype=int))
    assert np.array_equal(pessimistic_class_assignments, np.array([2, 1, 4, 1, 4, 1, 3, 2, 1, 4], dtype=int))


if __name__ == "__main__":
    pytest.main()
