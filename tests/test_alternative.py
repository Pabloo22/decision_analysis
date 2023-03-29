import pytest

from src.decision_analysis.decision_making import Alternative


def test_default_name():
    a1 = Alternative({})
    a2 = Alternative({})
    a3 = Alternative({})
    assert a1.name == 'a1'
    assert a2.name == 'a2'
    assert a3.name == 'a3'


if __name__ == '__main__':
    pytest.main()
