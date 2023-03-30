import pandas as pd

from src.decision_analysis.decision_making import Criterion


def load_dataset() -> pd.DataFrame:
    data = {
        'Price (€)': [366, 345, 234, 342, 351, 552, 864, 339, 411, 724, 434, 472],
        'Commodity': [5, 4, 2, 4, 4, 2, 3, 4, 3, 4, 4, 4],
        'Location': [2, 2, 2, 3, 1, 1, 1, 2, 2, 1, 1, 1],
        'Rating': [9.3, 7.5, 5.7, 9.5, 8.5, 7.8, 6.66, 7.3, 8.0, 9.44, 9.82, 9.84]
    }

    alternatives = ['a_1', 'a_2', 'a_3', 'a_4', 'a_5', 'a_6', 'a_7', 'a_8', 'a_9', 'a_10', 'a_11', 'a_12']

    df = pd.DataFrame(data, index=alternatives)

    return df


def load_criteria() -> list[Criterion]:
    criteria = [
        Criterion(name='Price (€)', type=-1),
        Criterion(name='Commodity', type=1),
        Criterion(name='Location', type=-1),
        Criterion(name='Rating', type=1)
    ]

    return criteria
