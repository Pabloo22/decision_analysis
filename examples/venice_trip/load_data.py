import pandas as pd

from decision_analysis.decision_making import Alternative
from src.decision_analysis.decision_making.data_structures import Criterion
from src.decision_analysis.decision_making.data_structures import Dataset, Comparison, ComparisonType,ValueFunction


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
                      preference_threshold=2,
                      veto_threshold=2),
            Criterion(type=1,
                      weight=0.071,
                      name='Rating',
                      indifference_threshold=1,
                      preference_threshold=5,
                      veto_threshold=7)
        ]
    elif version == 3:
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
                      preference_threshold=2,
                      veto_threshold=2),
            Criterion(type=1,
                      weight=0.071,
                      name='Rating',
                      indifference_threshold=1,
                      preference_threshold=5,
                      veto_threshold=7)
        ]
    elif version == 4:
        criteria = [
            Criterion(type=-1,
                      name='Price',
                      value_function=ValueFunction([234, 864])),
            Criterion(type=1,
                      name='Commodity',
                      value_function=ValueFunction([2, 4, 5])),
            Criterion(type=-1,
                      name='Location',
                      value_function=ValueFunction([1, 3])),
            Criterion(type=1,
                      name='Rating',
                      value_function=ValueFunction([5.7, 7.08, 8.46, 9.84]))
        ]
    else:
        raise ValueError('Invalid version')

    return criteria


def load_profile_boundaries():
    b3 = Alternative({'Price': 360, 'Commodity': 4, 'Location': 1, 'Rating': 8}, name='b3')
    b2 = Alternative({'Price': 430, 'Commodity': 3, 'Location': 2, 'Rating': 7}, name='b2')
    b1 = Alternative({'Price': 500, 'Commodity': 2, 'Location': 3, 'Rating': 5}, name='b1')

    boundaries = [b1, b2, b3]
    return boundaries

def get_dataset(criteria_version=4) -> Dataset:
    df = load_dataset()
    data = df.values
    criteria = load_criteria(criteria_version)
    alternative_names = df.index
    dataset = Dataset(data=data, criteria=criteria, alternative_names=alternative_names)

    return dataset