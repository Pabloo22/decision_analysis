from load_data import load_dataset, load_criteria

from src.decision_analysis.decision_making import find_dominated_alternatives


def print_dominated_alternatives():
    df = load_dataset()
    criteria = load_criteria()
    criteria_types = [c.type for c in criteria]
    print('Dominated alternatives:')
    print(find_dominated_alternatives(df, criteria_types))


if __name__ == '__main__':
    print_dominated_alternatives()
