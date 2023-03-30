import pandas as pd
import pytest

from src.decision_analysis.decision_making import find_dominated_alternatives


def test_find_dominated_alternatives():
    data = {
        'g1': [1, 2, 3],  # a1, a2, a3
        'g2': [3, 2, 1],
        'g3': [1, 2, 3]
    }
    alternatives = ['Alternative 1', 'Alternative 2', 'Alternative 3']
    criteria_types = [1, -1, 1]
    df = pd.DataFrame(data, index=alternatives)
    expected_output = {'Alternative 2': ['Alternative 1'], 'Alternative 3': ['Alternative 1', 'Alternative 2']}
    result = find_dominated_alternatives(df, criteria_types)
    assert result == expected_output

    data = {
        'g1': [1, 2, 2],  # a1, a2, a3
        'g2': [3, 2, 1],
        'g3': [1, 2, 2]
    }
    df = pd.DataFrame(data, index=alternatives)
    expected_output = {'Alternative 2': ['Alternative 1'], 'Alternative 3': ['Alternative 1', 'Alternative 2']}
    result = find_dominated_alternatives(df, criteria_types)
    assert result == expected_output

    data = {
        'g1': [1, 2, 3],  # a1, a2, a3
        'g2': [3, 2, 1],
        'g3': [1, 2, 3]
    }
    df = pd.DataFrame(data, index=alternatives)
    expected_output = {}
    criteria_types = [1, 1, 1]
    result = find_dominated_alternatives(df, criteria_types)
    assert result == expected_output


if __name__ == '__main__':
    pytest.main()
