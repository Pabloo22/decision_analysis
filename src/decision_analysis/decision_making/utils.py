import collections
import pandas as pd


def find_dominated_alternatives(df: pd.DataFrame, criteria_types: list[int]) -> dict[str, list[str]]:
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
