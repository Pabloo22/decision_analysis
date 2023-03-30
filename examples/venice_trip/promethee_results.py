import networkx as nx
import matplotlib.pyplot as plt

from decision_analysis.decision_making import Promethee
from load_data import load_dataset, load_criteria


def draw_graph(graph, name):
    plt.figure(figsize=(10, 10))
    nx.draw(graph, with_labels=True, node_size=1000, node_color="lightblue", font_size=16, font_weight="bold",
            edgecolors="black", linewidths=2, alpha=0.9, width=2, font_color="black", arrowsize=20, arrowstyle="->")
    plt.savefig(name)


def promethee_results(version: int = 1):
    print(f'PROMETHEE version {version} results:')
    df = load_dataset()
    criteria = load_criteria(version=version)
    promethee = Promethee(matrix_of_alternative_values=df.values,
                          criteria=criteria,
                          alternatives_names=df.index)
    promethee.run()

    # Get the graphs
    g_i = promethee.get_ranking_graph('I')
    g_ii = promethee.get_ranking_graph('II')
    positive_flow = promethee.get_positive_ranking_graph()
    negative_flow = promethee.get_negative_ranking_graph()

    print(promethee.get_dict_of_lists('I'))
    print(promethee.get_dict_of_lists('II'))

    # Draw the graphs
    draw_graph(g_i, f'P_I_v{version}.png')
    draw_graph(g_ii, f'P_II_v{version}.png')
    draw_graph(positive_flow, f'P+_v{version}.png')
    draw_graph(negative_flow, f'P-v_{version}.png')



if __name__ == '__main__':
    promethee_results(1)
    promethee_results(2)
