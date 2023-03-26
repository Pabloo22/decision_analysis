import numpy as np


class Promethee:
    """Promethee method for multi-criteria decision-making
    
    Attributes:
        matrix (np.array): Matrix with the value for each criterion for each alternative.
        alternatives (list): List with the alternatives names.
        criteria_types (list): If the criterion is benefit or cost (1 or -1).
        weights (np.array): Array with the weight for each criterion.
        preference_thresholds (np.array): Array containing the preference thresholds
        indifference_thresholds (np.array): Array containing the indifference thresholds

        
    Methods:
        rank(method): Ranks the alternatives based on their concordance and discordance indices, using the specified 
        method (either 'I' or 'II').
    """

    def __init__(self, matrix: np.ndarray, alternatives: list, criteria_types: list, weights: np.ndarray,
                 preference_thresholds: np.ndarray, indifference_thresholds: np.ndarray):
        """
        Initializes the promethee object.

        Args:
            matrix (np.array): Matrix with the value for each criterion for each alternative.
            alternatives (list): List with the alternatives names.
            criteria_types (list): If the criterion is benefit or cost (1 or -1).
            weights (np.array): Array with the weight for each criterion.
            preference_thresholds (np.array): Array containing the preference thresholds
            indifference_thresholds (np.array): Array containing the indifference thresholds
        """
        self.matrix = matrix
        self.alternatives = alternatives
        self.criteria_types = criteria_types
        self.weights = weights
        self.preference_thresholds = preference_thresholds
        self.indifference_thresholds = indifference_thresholds
        self.n_alternatives = matrix.shape[0]
        self.n_criteria = matrix.shape[1]
        self.comprehensiveness_matrix = np.zeros((self.n_alternatives, self.n_alternatives))
        self.positive_flow = np.zeros(self.n_alternatives)
        self.negative_flow = np.zeros(self.n_alternatives)
        self.net_flow = np.zeros(self.n_alternatives)

        self.__calculate_comprehensiveness_matrix()
        self.__calculate_positive_flow()
        self.__calculate_negative_flow()
        self.__calculate_net_flow()

    def __calculate_comprehensiveness_matrix(self):
        for i in range(self.n_alternatives):
            for j in range(self.n_alternatives):
                self.comprehensiveness_matrix[i, j] = self.__calculate_comprehensiveness(i, j)

    def __calculate_comprehensiveness(self, i, j):
        if i == j:
            return 0
        else:
            return sum(self.__calculate_comprehensiveness_criterion(i, j, k) / sum(self.weights) for k in
                       range(self.n_criteria))

    def __calculate_comprehensiveness_criterion(self, i, j, k):
        """Calculates the comprehensiveness for a given criterion.
        First, it calculates the difference between the value of the criterion for the alternatives i and j, keeping in
        mind that the criterion can be either a benefit or a cost, (that is why we multiply by the criteria type, to
        invert the sign if it is a cost). Then, it checks if the difference is smaller than the indifference threshold,
        in which case it returns 0. If the difference is larger than the preference threshold, it returns the weight of
        the criterion. Otherwise, it interpolates between the indifference and preference thresholds, and returns the
        comprehensiveness value.
        """
        print(i, j, k)
        diff = self.criteria_types[k] * (self.matrix[i, k] - self.matrix[j, k])
        if diff < self.indifference_thresholds[k]:
            return 0
        elif diff > self.preference_thresholds[k]:
            return self.weights[k]
        else:
            return self.weights[k] * (diff - self.indifference_thresholds[k]) / (
                    self.preference_thresholds[k] - self.indifference_thresholds[k])

    def __calculate_positive_flow(self):
        self.positive_flow = np.sum(self.comprehensiveness_matrix, axis=1)

    def __calculate_negative_flow(self):
        self.negative_flow = np.sum(self.comprehensiveness_matrix, axis=0)

    def __calculate_net_flow(self):
        self.net_flow = self.positive_flow - self.negative_flow

    def rank(self, method: str):
        if method == 'I':
            return self.__rank_method_I()
        elif method == 'II':
            return self.__rank_method_II()
        else:
            raise ValueError('Invalid method, must be either "I" or "II"')

    def __rank_method_I(self):
        """Ranks the alternatives based on their positive and negative flows.
        a > b:
        - if a has a higher positive flow than b and a has a lower negative flow than b
        - if a has a higher positive flow than b and a and b have the same negative flow
        - if a and b have the same positive flow and a has a lower negative flow than b

        a R b:
        - if a has a higher positive flow than b and a has a higher negative flow than b
        - if a has a lower positive flow than b and a has a lower negative flow than b

        a = b:
        - if a and b have the same positive flow and a and b have the same negative flow
        """
        # return self.alternatives[np.lexsort((self.negative_flow, self.positive_flow))[::-1]]
        pass

    def __rank_method_II(self):
        return self.alternatives[np.argsort(self.net_flow)[::-1]]


if __name__ == "__main__":
    matrix = np.array([[10, 18, 10], [15, 0, 20]])
    alternatives = ['A', 'B']
    criteria_types = [1, 1, 1]
    weights = np.array([3, 5, 2])
    preference_thresholds = np.array([0, 20, 5])
    indifference_thresholds = np.array([10, 10, 2])
    promethee = Promethee(matrix, alternatives, criteria_types, weights, preference_thresholds, indifference_thresholds)
    print(promethee.comprehensiveness_matrix)
