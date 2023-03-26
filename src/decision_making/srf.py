import numpy as np


class Srf:
    """Simos-Roy-Figueira method for determining the criteria weights.

    Attributes:
        rank_dict (dict): Dictionary containing the alternatives and their ranks. The keys are rank and the values are
        the alternatives. The white card alternative must be called 'white_card'.

    Methods:
        get_weights(): Returns the weights for each criterion.
    """

    def __init__(self, rank_dict: dict, ratio: int):
        """
        Initializes the srf object.

        Args: rank_dict (dict): Dictionary containing the alternatives and their ranks. The keys are rank and the
        values are the alternatives (it can be a single alternative or a list of alternatives). The white card
        alternative must be called 'white_card' or you can omit those ranks.
        """
        self.rank_dict = rank_dict
        self.ratio = ratio
        self.r_v = max(self.rank_dict.keys())
        self.weights = dict()

        self._preprocess_rank_dict()

    def _preprocess_rank_dict(self):
        """Deletes the white card alternatives from the rank dictionary and ensures that each rank has a list of
        alternatives."""

        for rank in self.rank_dict:
            if self.rank_dict[rank] == 'white_card':
                del self.rank_dict[rank]
            elif not isinstance(self.rank_dict[rank], list):
                self.rank_dict[rank] = [self.rank_dict[rank]]

    def calculate_weights(self):
        for rank in self.rank_dict.keys():
            for alternative in self.rank_dict[rank]:
                self.weights[alternative] = self._calculate_weight(rank)

        sum_weights = sum(self.weights.values())
        for alternative in self.weights.keys():
            self.weights[alternative] /= sum_weights

    def _calculate_weight(self, rank):
        return 1 + (self.ratio - 1) * (rank - 1) / (self.r_v - 1)


if __name__ == "__main__":
    rank_dict = {1: 'g1', 2: 'white_card', 3: 'white_card', 4: ['g6', 'g8'], 5: 'g5', 6: 'white_card', 7: 'white_card',
                 8: 'white_card', 9: ['g2', 'g3', 'g7'], 10: 'white_card', 11: 'g4'}
    srf = Srf(rank_dict, 10)
    srf.calculate_weights()
    # print the dictionary sorted by key
    print({k: srf.weights[k] for k in sorted(srf.weights)})

