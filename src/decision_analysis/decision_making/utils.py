import collections
import pandas as pd


def find_dominated_alternatives(df: pd.DataFrame, criteria_types: list[int]) -> dict[str, list[str]]:
    """Returns dictionary where the keys are the names of the alternatives and the values are
    lists of names of the alternatives that are dominated by the corresponding alternative.

    Given a pandas DataFrame, this function returns a dictionary where the keys are the
    names of the alternatives and the values are lists of names of the alternatives that
    are dominated by the corresponding alternative.

    An alternative is dominated by another one (is Pareto superior) if there is no other
    alternative b∈A such that:
    1. b is at least as good as a on all criteria gi, i=1,…,n, and
    2. b is strictly better than a for some criterion, gi, i∈{1,…,n}

    Args:
        df (pd.DataFrame): A DataFrame containing alternatives as indices and criteria as columns.
        criteria_types (list[int]): A list of integers where 1 means a benefit criterion and -1 means a cost
            criterion.
    """

    dominated = collections.defaultdict(list)
    criteria_types = pd.Series(criteria_types, index=df.columns)
    for i, a in enumerate(df.index):
        dominated_by_a = []
        for j, b in enumerate(df.index):
            if a == b:
                continue
            comparable_values_a = criteria_types * df.loc[a]
            comparable_values_b = criteria_types * df.loc[b]
            criteria_comparison = (comparable_values_a >= comparable_values_b).all()
            strictly_better = (comparable_values_a > comparable_values_b).any()

            if criteria_comparison and strictly_better:
                dominated_by_a.append(b)

        if dominated_by_a:
            dominated[a] = dominated_by_a

    return dominated
