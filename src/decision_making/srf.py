from typing import Union


def _preprocess_rank_dict(ranking: dict[int, Union[str, list[str]]]) -> dict[int, list[str]]:
    """Deletes the white card alternatives from the rank dictionary and ensures that each rank has a list of
    alternatives."""
    for rank_, value in ranking.copy().items():
        if value == 'white_card':
            del ranking[rank_]
        elif not isinstance(value, list):
            ranking[rank_] = [value]
    return ranking


def _calculate_weight(rank: int, max_rank: int, ratio: int) -> float:
    return 1 + (ratio - 1) * (rank - 1) / (max_rank - 1)


def srf(ranking: dict[int, Union[str, list[str]]], ratio: int) -> dict[str, float]:
    """Calculates the Simos-Roy-Figueira weights for the given rank dictionary and ratio.

    Args:
        ranking (dict): Dictionary containing the alternatives and their ranks. The keys are rank and the
                          values are the alternatives (it can be a single alternative or a list of alternatives).
                          The white card alternative must be called 'white_card' or you can omit those ranks.
        ratio (int): The ratio used in the weight calculation.

    Returns:
        dict: A dictionary containing the weights for each alternative.

    Example:
    >>> ranking_dict = {1: 'g1', 2: 'white_card', 3: 'white_card', 4: ['g6', 'g8'], 5: 'g5', \
                     6: 'white_card', 7: 'white_card', 8: 'white_card', 9: ['g2', 'g3', 'g7'], \
                     10: 'white_card', 11: 'g4'}
    >>> weights_dict = srf(ranking_dict, 10)
    >>> sorted_weights = {key: round(weights_dict[key], 3) for key in sorted(weights_dict)}
    >>> sorted_weights
    {'g1': 0.021, 'g2': 0.172, 'g3': 0.172, 'g4': 0.21, 'g5': 0.097, 'g6': 0.078, 'g7': 0.172, 'g8': 0.078}
    """

    ranking = _preprocess_rank_dict(ranking)
    max_rank = max(ranking.keys())
    weights = dict()

    for rank in ranking.keys():
        for alternative in ranking[rank]:
            weights[alternative] = _calculate_weight(rank, max_rank, ratio)

    sum_weights = sum(weights.values())
    for alternative in weights:
        weights[alternative] /= sum_weights

    return weights


if __name__ == "__main__":
    import doctest

    doctest.testmod()


