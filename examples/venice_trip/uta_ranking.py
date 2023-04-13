import matplotlib.pyplot as plt
import pulp


from comparisons import get_comparisons
from decision_analysis.decision_making.uta import UTA
from load_data import get_dataset


def plot_value_functions(prob: pulp.LpProblem):
    variables = {v.name: v.varValue for v in prob.variables() if v.name.startswith('u_')}
    number_of_plots = max([int(v.split('_')[1].split('(')[0]) for v in variables.keys()])
    print(number_of_plots)
    x = [[float(v.split('(')[1][:-1]) for v in variables.keys() if v.startswith(f'u_{i}')]
         for i in range(1, number_of_plots + 1)]
    y = [[variables[v] for v in variables.keys() if v.startswith(f'u_{i}')] for i in range(1, number_of_plots + 1)]
    max_y = max([max(y[i]) for i in range(number_of_plots)])
    for u in range(1, number_of_plots + 1):
        plt.plot(x[u - 1], y[u - 1])
        plt.title(f'$u_{u}$')
        plt.xlabel('$x$')
        plt.ylabel('$u(x)$')
        plt.ylim(0, max_y)
        plt.xlim(x[u - 1][0], x[u - 1][-1])
        plt.show()


def main():
    dataset = get_dataset()
    comparisons = get_comparisons()

    uta = UTA(dataset=dataset, comparisons=comparisons)

    uta.solve()

    UTA.print_problem_results(uta.prob)
    # plot_value_functions(uta.prob)
    uta.update_value_functions()
    print(uta.get_comprehensive_values())
    uta_ranking = uta.create_ranking()
    uta_ranking.visualize("UTA Ranking", seed=42, layout='spring')


if __name__ == '__main__':
    main()
