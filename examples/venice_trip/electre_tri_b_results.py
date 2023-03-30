import pandas as pd

from decision_analysis.decision_making import ElectreTriB

from load_data import load_dataset, load_criteria, load_profile_boundaries, Alternative


def run_electre_tri_b():
    df = load_dataset()
    criteria = load_criteria(version=2)
    boundaries = load_profile_boundaries()

    alternatives = Alternative.dataframe_to_alternatives(df)

    electre_tri_b = ElectreTriB(criteria=criteria, alternatives=alternatives, boundaries=boundaries)

    electre_tri_b.run()

    pessimistic_class_assignments = electre_tri_b.pessimistic_classes
    optimistic_class_assignments = electre_tri_b.optimistic_classes

    print("Outranking matrix:")
    relations = {-1: '?', 0: '<', 1: '>', 0.5: '='}
    print(pd.DataFrame(electre_tri_b.outranking_matrix.copy(),
                       index=[alt.name for alt in electre_tri_b.alternatives],
                       columns=[b.name for b in electre_tri_b.boundaries]
                       ).applymap(lambda x: relations[x]))

    classes = ["Poor", "Fair", "Good", "Excellent"]
    print("alternative | pessimistic_class | optimistic_class")
    print("--------------------------------------------------")
    for i, alternative in enumerate(alternatives):
        pessimistic_class = classes[pessimistic_class_assignments[i]]
        optimistic_class = classes[optimistic_class_assignments[i]]
        print(f"{alternative.name}, {pessimistic_class}, {optimistic_class}")

    # print(electre_tri_b.outranking_matrix.map(relations))


if __name__ == '__main__':
    run_electre_tri_b()

