import pytest

from src.decision_analysis.decision_making import Criterion


def test_default_name():
    c1 = Criterion(1)
    c2 = Criterion(1)
    c3 = Criterion(1)
    assert c1.name == 'g1'
    assert c2.name == 'g2'
    assert c3.name == 'g3'


if __name__ == '__main__':
    pytest.main()
