from decision_analysis.decision_making.uta import UTA
from decision_analysis.decision_making.promethee import Promethee
from decision_analysis.decision_making import Ranking
from load_data import get_dataset, load_criteria
from comparisons import get_comparisons

import networkx as nx

if __name__ == '__main__':
    dataset = get_dataset()
    alternative_dict = {alternative: i for i, alternative in enumerate(dataset.alternative_names)}
    comparisons = get_comparisons()

    uta = UTA(dataset=dataset, comparisons=comparisons)
    uta.solve()
    uta.update_value_functions()
    uta_ranking = uta.create_ranking()

    matrix = dataset.data
    alternatives = dataset.alternative_names
    criteria = load_criteria(2)
    promethee = Promethee(matrix, criteria, alternatives)
    promethee.run()
    promethee_ii_graph = promethee.get_ranking_graph('II')
    promethee_ii_ranking = Ranking(nx.to_numpy_array(promethee_ii_graph), alternatives)

    uta_ranking.visualize("UTA Ranking", seed=42, layout='spring')
    promethee_ii_ranking.visualize("Promethee II Ranking", seed=42, layout='spring')

    print('Kendall tau:', uta_ranking.kendall_tau(promethee_ii_ranking))
    print('Kendall distance', uta_ranking.kendall_distance(promethee_ii_ranking))
