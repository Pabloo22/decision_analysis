from decision_analysis.decision_making.uta import UTA
from decision_analysis.decision_making.promethee import Promethee
from decision_analysis.decision_making import Ranking
from load_data import get_dataset, load_criteria
from comparisons import get_all_comparisons

import networkx as nx

if __name__ == '__main__':
    dataset = get_dataset()
    alternative_dict = {alternative: i for i, alternative in enumerate(dataset.alternative_names)}
    comparisons = get_all_comparisons()

    uta = UTA(dataset=dataset, comparisons=comparisons, epsilon=0.01)
    uta.solve()
    uta.update_value_functions()
    uta_ranking = uta.create_ranking()
    UTA.print_problem_results(uta.prob)

    matrix = dataset.data
    alternatives = dataset.alternative_names
    criteria = load_criteria(2)
    promethee = Promethee(matrix, criteria, alternatives)
    promethee.run()
    promethee_ii_graph = promethee.get_ranking_graph('II')
    promethee_ii_ranking = Ranking.from_dict(dict(zip(nx.to_dict_of_lists(promethee_ii_graph).keys(), range(1, len(alternatives) + 1))))

    uta_ranking.visualize("UTA Ranking", seed=42, layout='spring')
    promethee_ii_ranking.visualize("Promethee II Ranking", seed=42, layout='spring')

    print('Kendall tau:', uta_ranking.kendall_tau(promethee_ii_ranking))
    print('Kendall distance', uta_ranking.kendall_distance(promethee_ii_ranking))
