"""In this script, we will find some pairwise comparisons based on the idea that if the difference
in price is not too big, then an accomadation near Venice is preferable to one further away."""

from decision_analysis.decision_making import Comparison, ComparisonType

from load_data import load_dataset


def find_comparisons(price_difference: float = 25):

    dataset = load_dataset()

    for i, row in dataset.iterrows():
        for j, row2 in dataset.iterrows():
            if i >= j:
                continue
            small_price_difference = abs(row['Price'] - row2['Price']) < price_difference
            accommodation_nearer_venice = row['Location'] < row2['Location']
            if small_price_difference and accommodation_nearer_venice:
                print(f"${i}$ is preferred over ${j}$ (location)", end=', ')
                continue
            better_commodity = row['Commodity'] > row2['Commodity']
            same_location = row['Location'] == row2['Location']
            if small_price_difference and better_commodity and same_location:
                print(f"${i}$ is preferred over ${j}$ (commodity)", end=', ')
                continue


def get_comparisons():
    comparisons = [
        Comparison(4, 0, ComparisonType.PREFERENCE),  # Initial pairwise comparison
        Comparison(4, 1, ComparisonType.PREFERENCE),  # Initial pairwise comparison
        Comparison(4, 2, ComparisonType.PREFERENCE),  # Initial pairwise comparison
        Comparison(4, 3, ComparisonType.PREFERENCE),  # Initial pairwise comparison
        Comparison(0, 3, ComparisonType.PREFERENCE),  # Proximity to Venice and low price difference
        Comparison(1, 3, ComparisonType.PREFERENCE),  # Proximity to Venice and low price difference
        Comparison(4, 7, ComparisonType.PREFERENCE),  # Proximity to Venice and low price difference
        Comparison(10, 8, ComparisonType.PREFERENCE),  # Proximity to Venice and low price difference
        Comparison(0, 1, ComparisonType.PREFERENCE),  # Commodity, low price difference, same location
        Comparison(0, 6, ComparisonType.PREFERENCE),  # Least favorable option
        Comparison(1, 6, ComparisonType.PREFERENCE),  # Least favorable option
        Comparison(2, 6, ComparisonType.PREFERENCE),  # Least favorable option
        Comparison(3, 6, ComparisonType.PREFERENCE),  # Least favorable option
        Comparison(4, 6, ComparisonType.PREFERENCE),  # Least favorable option
        Comparison(5, 4, ComparisonType.PREFERENCE)  # Check inconsistency detection
    ]
    return comparisons


if __name__ == '__main__':
    find_comparisons()
