import pytest
from decision_analysis.decision_making import srf
from src.decision_analysis.decision_making.srf import _preprocess_rank_dict, _calculate_weight


def test_preprocess_rank_dict():
    ranking = {
        1: 'g1',
        2: 'white_card',
        3: 'white_card',
        4: ['g6', 'g8'],
        5: 'g5',
        6: 'white_card',
        7: 'white_card',
        8: 'white_card',
        9: ['g2', 'g3', 'g7'],
        10: 'white_card',
        11: 'g4'
    }
    expected = {
        1: ['g1'],
        4: ['g6', 'g8'],
        5: ['g5'],
        9: ['g2', 'g3', 'g7'],
        11: ['g4']
    }
    assert _preprocess_rank_dict(ranking) == expected


def test_srf():
    ranking = {
        1: 'g1',
        2: 'white_card',
        3: 'white_card',
        4: ['g6', 'g8'],
        5: 'g5',
        6: 'white_card',
        7: 'white_card',
        8: 'white_card',
        9: ['g2', 'g3', 'g7'],
        10: 'white_card',
        11: 'g4'
    }
    weights = srf(ranking, 10)
    expected = {
        'g1': pytest.approx(0.021, abs=1e-3),
        'g2': pytest.approx(0.172, abs=1e-3),
        'g3': pytest.approx(0.172, abs=1e-3),
        'g4': pytest.approx(0.210, abs=1e-3),
        'g5': pytest.approx(0.097, abs=1e-3),
        'g6': pytest.approx(0.078, abs=1e-3),
        'g7': pytest.approx(0.172, abs=1e-3),
        'g8': pytest.approx(0.078, abs=1e-3)
    }
    assert weights == expected


if __name__ == "__main__":
    pytest.main()
