import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.decision_analysis.decision_making import Criterion

sns.set()

class Electre_trib:

    @staticmethod
    def plot_marginal_concordance(criterion: Criterion):
        """Plots the marginal concordance for the criterion.

        Args:
            criterion: The criterion to plot.
        """
        g_i_b = 10 + criterion.preference_threshold  # g_i(b) example value, only for plotting

        b_preference_a = np.linspace(0, g_i_b - criterion.preference_threshold,
                                     100) if criterion.criteria_type == 1 else \
            np.linspace(g_i_b + criterion.preference_threshold, g_i_b + criterion.preference_threshold + 10,
                        100)  # a P_i b
        b_weak_preference_a = np.linspace(g_i_b - criterion.preference_threshold,
                                          g_i_b - criterion.indifference_threshold,
                                          100) if criterion.criteria_type == 1 else \
            np.linspace(g_i_b + criterion.indifference_threshold, g_i_b + criterion.preference_threshold,
                        100)  # b Q_i a
        b_indifference_a = np.linspace(g_i_b - criterion.indifference_threshold,
                                       g_i_b + criterion.indifference_threshold,
                                       100) if criterion.criteria_type == 1 else \
            np.linspace(g_i_b - criterion.indifference_threshold, g_i_b + criterion.indifference_threshold,
                        100)  # a I_i b
        b_weak_dispreference_a = np.linspace(g_i_b + criterion.indifference_threshold,
                                             g_i_b + criterion.preference_threshold,
                                             100) if criterion.criteria_type == 1 else \
            np.linspace(g_i_b - criterion.preference_threshold, g_i_b - criterion.indifference_threshold,
                        100)  # a Q_i b
        b_dispreference_a = np.linspace(g_i_b + criterion.preference_threshold,
                                        g_i_b + criterion.preference_threshold + 10,
                                        100) if criterion.criteria_type == 1 else \
            np.linspace(0, g_i_b - criterion.preference_threshold, 100)  # a P_i b

        b_preference_a_y = np.zeros(len(b_preference_a))
        b_weak_preference_a_y = criterion.weight * (
                    criterion.preference_threshold - criterion.criteria_type * (g_i_b - b_weak_preference_a)) / (
                                            criterion.preference_threshold - criterion.indifference_threshold)
        b_indifference_a_y = np.ones(len(b_indifference_a)) * criterion.weight
        b_weak_dispreference_a_y = np.ones(len(b_weak_dispreference_a)) * criterion.weight
        b_dispreference_a_y = np.ones(len(b_dispreference_a)) * criterion.weight

        plt.plot(b_preference_a, b_preference_a_y, label='$b_h P_i a$', color='purple')
        plt.plot(b_weak_preference_a, b_weak_preference_a_y, label='$b_h Q_i a$', color='orange')
        plt.plot(b_indifference_a, b_indifference_a_y, label='$a I_i b_h$', color='yellowgreen')
        plt.plot(b_weak_dispreference_a, b_weak_dispreference_a_y, label='$a Q_i b_h$', color='green')
        plt.plot(b_dispreference_a, b_dispreference_a_y, label='$a P_i b_h$', color='aquamarine')
        plt.legend()
        plt.ylabel('$w_{i}\cdot c_i(a,b_h)$')
        plt.xlabel('$g_i(b)$')
        plt.show()

    @staticmethod
    def plot_marginal_discordance(criterion: Criterion):
        """Plots the marginal discordance for the criterion.

        Args:
            criterion: The criterion to plot.
        """
        g_i_b = 10 + criterion.veto_threshold  # g_i(b) example value, only for plotting

        b_veto_a = np.linspace(0, g_i_b - criterion.veto_threshold, 100) if criterion.criteria_type == 1 else \
            np.linspace(g_i_b + criterion.veto_threshold, g_i_b + criterion.veto_threshold + 10,
                        100)  # a worse than b by at least v_i

        b_partial_veto_a = np.linspace(g_i_b - criterion.veto_threshold, g_i_b - criterion.preference_threshold,
                                       100) if criterion.criteria_type == 1 else \
            np.linspace(g_i_b + criterion.preference_threshold, g_i_b + criterion.veto_threshold,
                        100)  # a worse than b by at least p_i but less than v_i

        b_no_veto_a = np.linspace(g_i_b - criterion.preference_threshold, g_i_b + criterion.preference_threshold,
                                  100) if criterion.criteria_type == 1 else \
            np.linspace(g_i_b - criterion.preference_threshold, g_i_b + criterion.preference_threshold,
                        100)  # a as good as b or worse than b by at most p_i

        b_veto_a_y = np.zeros(len(b_veto_a))
        b_partial_veto_a_y = criterion.weight * (
                    criterion.veto_threshold - criterion.criteria_type * (g_i_b - b_partial_veto_a)) / (
                                         criterion.veto_threshold - criterion.preference_threshold)
        b_no_veto_a_y = np.ones(len(b_no_veto_a)) * criterion.weight

        plt.plot(b_veto_a, b_veto_a_y, color='purple')
        plt.plot(b_partial_veto_a, b_partial_veto_a_y, color='orange')
        plt.plot(b_no_veto_a, b_no_veto_a_y, color='yellowgreen')
        plt.ylabel('$w_{i}\cdot D_i(a,b_h)$')
        plt.xlabel('$g_i(b)$')