import collections

from load_data import load_dataset, load_criteria
from decision_analysis.decision_making import find_dominated_alternatives


def print_dominated_alternatives():
    df = load_dataset()
    criteria = load_criteria()
    criteria_types = [c.type for c in criteria]
    result = find_dominated_alternatives(df, criteria_types)
    dominated_dominating_dict = collections.defaultdict(set)
    for alternative, dominated in result.items():
        for d in dominated:
            dominated_dominating_dict[d].add(alternative)

    print('Dominated alternatives:')
    for alternative, dominated in dominated_dominating_dict.items():
        print(f'{alternative} is dominated by {dominated}')


if __name__ == '__main__':
    print_dominated_alternatives()
