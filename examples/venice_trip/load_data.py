import pandas as pd

from src.decision_analysis.decision_making import Criterion, Alternative


def load_dataset() -> pd.DataFrame:
    data = {
        'Price': [366, 345, 234, 342, 351, 552, 864, 339, 411, 724, 434, 472],
        'Commodity': [5, 4, 2, 4, 4, 2, 3, 4, 3, 4, 4, 4],
        'Location': [2, 2, 2, 3, 1, 1, 1, 2, 2, 1, 1, 1],
        'Rating': [9.3, 7.5, 5.7, 9.5, 8.5, 7.8, 6.66, 7.3, 8.0, 9.44, 9.82, 9.84]
    }

    alternatives = ['a_1', 'a_2', 'a_3', 'a_4', 'a_5', 'a_6', 'a_7', 'a_8', 'a_9', 'a_10', 'a_11', 'a_12']

    df = pd.DataFrame(data, index=alternatives)

    return df


def load_criteria(version: int = 1) -> list[Criterion]:
    if version == 1:
        criteria = [
            Criterion(type=-1,
                      weight=0.357,
                      name='Price',
                      indifference_threshold=20,
                      preference_threshold=200,
                      veto_threshold=300),
            Criterion(type=1,
                      weight=0.214,
                      name='Commodity',
                      indifference_threshold=0,
                      preference_threshold=2,
                      veto_threshold=3),
            Criterion(type=-1,
                      weight=0.357,
                      name='Location',
                      indifference_threshold=0,
                      preference_threshold=1,
                      veto_threshold=2),
            Criterion(type=1,
                      weight=0.071,
                      name='Rating',
                      indifference_threshold=1,
                      preference_threshold=5,
                      veto_threshold=7)
        ]
    elif version == 2:
        criteria = [
            Criterion(type=-1,
                      weight=0.357,
                      name='Price',
                      indifference_threshold=20,
                      preference_threshold=60,
                      veto_threshold=300),
            Criterion(type=1,
                      weight=0.214,
                      name='Commodity',
                      indifference_threshold=0,
                      preference_threshold=2,
                      veto_threshold=3),
            Criterion(type=-1,
                      weight=0.357,
                      name='Location',
                      indifference_threshold=0,
                      preference_threshold=1,
                      veto_threshold=2),
            Criterion(type=1,
                      weight=0.071,
                      name='Rating',
                      indifference_threshold=1,
                      preference_threshold=5,
                      veto_threshold=7)
        ]
    else:
        raise ValueError('Invalid version')

    return criteria


def load_profile_boundaries():
    b1 = Alternative({'Price': 360, 'Commodity': 4, 'Location': 1, 'Rating': 8}, name='b1')
    b2 = Alternative({'Price': 430, 'Commodity': 3, 'Location': 2, 'Rating': 7}, name='b2')
    b3 = Alternative({'Price': 500, 'Commodity': 2, 'Location': 3, 'Rating': 5}, name='b3')

    boundaries = [b1, b2, b3]
    return boundaries
