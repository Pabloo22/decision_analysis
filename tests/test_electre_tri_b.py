import pytest

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
    electre = ElectreTriB(criteria, [a, e], profiles=[bt], cutting_level=0.6)

    return electre


def test_calculate_marginal_concordance():
    electre = _electre_exercise_1()
    a, e = electre.alternatives
    g1, g2 = electre.criteria
    bt = electre.boundaries[0]

    assert electre.calculate_marginal_concordance(a, bt, g1) == 1
    assert electre.calculate_marginal_concordance(a, bt, g2) == 0
    assert electre.calculate_marginal_concordance(e, bt, g1) == 1
    assert electre.calculate_marginal_concordance(e, bt, g2) == 0.5
    assert electre.calculate_marginal_concordance(bt, a, g1) == 1
    assert electre.calculate_marginal_concordance(bt, a, g2) == 1
    assert electre.calculate_marginal_concordance(bt, e, g1) == 0
    assert electre.calculate_marginal_concordance(bt, e, g2) == 1


def test_calculate_marginal_discordance():
    electre = _electre_exercise_1()
    a, e = electre.alternatives
    g1, g2 = electre.criteria
    bt = electre.boundaries[0]

    assert electre.calculate_marginal_discordance(a, bt, g1) == 0
    assert electre.calculate_marginal_discordance(a, bt, g2) == 1
    assert electre.calculate_marginal_discordance(e, bt, g1) == 0
    assert electre.calculate_marginal_discordance(e, bt, g2) == 0
    assert electre.calculate_marginal_discordance(bt, a, g1) == 0
    assert electre.calculate_marginal_discordance(bt, a, g2) == 0
    assert electre.calculate_marginal_discordance(bt, e, g1) == 0.8
    assert electre.calculate_marginal_discordance(bt, e, g2) == 0


# def test_electre_tri_b():
#     # Instantiate Criterion objects
#     g1 = Criterion(criteria_type=1, weight=2, preference_threshold=50, indifference_threshold=10, veto_threshold=100)
#     g2 = Criterion(criteria_type=-1, weight=3, preference_threshold=10, indifference_threshold=0, veto_threshold=20)
#
#     # Create a list of criteria
#     criteria = [g1, g2]
#
#     # Instantiate Alternative objects
#     a = Alternative({"g1": 145, "g2": 40})
#     e = Alternative({"g1": 240, "g2": 20})
#     bt = Alternative({"g1": 150, "g2": 15})
#
#     # Instantiate ElectreTriB object
#     electre = ElectreTriB(criteria, [a, e, bt])
#
#     # Compute indices and outranking relations for pairs (a, bt), (bt, a), (e, bt), (bt, e)
#     pairs = [(a, bt), (bt, a), (e, bt), (bt, e)]
#     credibility_threshold = 0.6
#
#     for pair in pairs:
#         alternative1, alternative2 = pair
#         electre.compute_marginal_concordance_and_discordance(alternative1, alternative2)
#         electre.compute_comprehensive_concordance()
#         electre.compute_outranking_credibilities()
#         relation = electre.get_relation(credibility_threshold)
#         print(f"Relation between {alternative1.name} and {alternative2.name}: {relation}")


if __name__ == "__main__":
    pytest.main()
