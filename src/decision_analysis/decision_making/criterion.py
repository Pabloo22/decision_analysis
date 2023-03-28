from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import Optional


@dataclass
class Criterion:
    """
    Attributes:
        criteria_type (int): If the criterion is benefit or cost (1 or -1).
        weight (float): Weight of the criterion. Optional, default 1.
        name (str): Name of the criterion. Optional.
        preference_threshold (float): Preference threshold for the criterion. Optional.
        indifference_threshold (float): Indifference threshold for the criterion. Optional.
        veto_threshold (float): Veto threshold for the criterion. Optional.

    """
    criteria_type: int
    weight: float = 1.
    name: str = ""
    preference_threshold: Optional[float] = None
    indifference_threshold: Optional[float] = None
    veto_threshold: Optional[float] = None

    def plot_thresholds(self):
        if self.preference_threshold is None:
            self.preference_threshold = 0
        if self.indifference_threshold is None:
            self.indifference_threshold = 0
        if self.veto_threshold is None:
            self.veto_threshold = 0
        x = [self.indifference_threshold, self.preference_threshold, self.veto_threshold]
        y = [0, self.weight, 0]
        plt.plot(x, y)
        plt.title(self.name)
        plt.xlabel("Value")
        plt.ylabel("Weight")
        plt.show()


if __name__ == "__main__":
    c = Criterion(1, name="Criterion 1", preference_threshold=0.5, indifference_threshold=0.1, veto_threshold=0.9)
    c.plot_thresholds()
