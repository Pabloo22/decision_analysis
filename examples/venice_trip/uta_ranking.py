import matplotlib.pyplot as plt
import pulp


from comparisons import get_comparisons
from src.decision_analysis.decision_making.data_structures.dataset import Dataset
from src.decision_analysis.decision_making.uta import UTA
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
        plt.plot(x[u - 1], y[u - 1], label=f'u_{u}')
        plt.title(f'u_{u}')
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.ylim(0, max_y)
        plt.show()


if __name__ == '__main__':
    dataset = get_dataset()

    alternative_dict = {alternative: i for i, alternative in enumerate(dataset.alternative_names)}
    comparisons = get_comparisons()

    uta = UTA(dataset=dataset, comparisons=comparisons)

    uta.solve()

    UTA.print_model_results(uta.prob)
    plot_value_functions(uta.prob)
