from typing import Union


def srf(rank_dict: dict[int, Union[str, list[str]]], ratio: int) -> dict[str, float]:
    """Calculates the Simos-Roy-Figueira weights for the given rank dictionary and ratio.

    Args:
        rank_dict (dict): Dictionary containing the alternatives and their ranks. The keys are rank and the
                          values are the alternatives (it can be a single alternative or a list of alternatives).
                          The white card alternative must be called 'white_card' or you can omit those ranks.
        ratio (int): The ratio used in the weight calculation.

    Returns:
        dict: A dictionary containing the weights for each alternative.

    Example:
    >>> rank_dict = {1: 'g1', 2: 'white_card', 3: 'white_card', 4: ['g6', 'g8'], 5: 'g5', \
                     6: 'white_card', 7: 'white_card', 8: 'white_card', 9: ['g2', 'g3', 'g7'], \
                     10: 'white_card', 11: 'g4'}
    >>> weights = srf(rank_dict, 10)
    >>> sorted_weights = {k: round(weights[k], 3) for k in sorted(weights)}
    >>> sorted_weights
    {'g1': 0.021, 'g2': 0.172, 'g3': 0.172, 'g4': 0.21, 'g5': 0.097, 'g6': 0.078, 'g7': 0.172, 'g8': 0.078}
    """
    def preprocess_rank_dict(rank_dict_: dict[int, Union[str, list[str]]]) -> dict[int, list[str]]:
        for rank_ in rank_dict_.copy():
            if rank_dict_[rank_] == 'white_card':
                del rank_dict_[rank_]
            elif not isinstance(rank_dict_[rank_], list):
                rank_dict_[rank_] = [rank_dict_[rank_]]
        return rank_dict_

    def calculate_weight(rank_, r_v_, ratio_):
        return 1 + (ratio_ - 1) * (rank_ - 1) / (r_v_ - 1)

    rank_dict = preprocess_rank_dict(rank_dict)
    r_v = max(rank_dict.keys())
    weights = dict()

    for rank in rank_dict.keys():
        for alternative in rank_dict[rank]:
            weights[alternative] = calculate_weight(rank, r_v, ratio)

    sum_weights = sum(weights.values())
    for alternative in weights.keys():
        weights[alternative] /= sum_weights

    return weights


if __name__ == "__main__":
    import doctest

    doctest.testmod()


