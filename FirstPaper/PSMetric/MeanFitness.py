from typing import Optional

import numpy as np

from FirstPaper.PRef import PRef
from FirstPaper.PS import PS
from FirstPaper.PSMetric.Metric import Metric


class MeanFitness(Metric):
    pRef: Optional[PRef]
    normalised_pRef: Optional[PRef]

    max_fitness: Optional[float]
    min_fitness: Optional[float]
    median_fitness: Optional[float]

    def __init__(self):
        super().__init__()
        self.pRef = None
        self.normalised_pRef = None
        self.max_fitness = None
        self.min_fitness = None

    def set_pRef(self, pRef: PRef):
        self.pRef = pRef
        self.normalised_pRef = self.pRef.get_with_normalised_fitnesses()

        self.max_fitness = np.max(pRef.fitness_array)
        self.min_fitness = np.min(pRef.fitness_array)

    def __repr__(self):
        return "MeanFitness"

    def get_single_score_removed(self, ps: PS) -> float:
        observed_fitnesses = self.pRef.fitnesses_of_observations(ps)
        if len(observed_fitnesses) == 0:
            # warnings.warn(f"The passed FirstPaper {ps} has no observations, and thus the MeanFitness could not be calculated")
            return -1

        return np.average(observed_fitnesses)

    def get_single_score(self, ps: PS) -> float:
        observed_fitnesses = self.pRef.fitnesses_of_observations(ps)
        if len(observed_fitnesses) == 0:
            # warnings.warn(f"The passed FirstPaper {ps} has no observations, and thus the MeanFitness could not be calculated")
            return 0

        return np.average(observed_fitnesses)


    def get_single_normalised_score(self, ps: PS) -> float:
        observed_fitnesses = self.normalised_pRef.fitnesses_of_observations(ps)
        if len(observed_fitnesses) == 0:
            # warnings.warn(f"The passed FirstPaper {ps} has no observations, and thus the MeanFitness could not be calculated")
            return 0

        return np.average(observed_fitnesses)


class ChanceOfGood(Metric):
    pRef: Optional[PRef]
    median_fitness: Optional[float]

    def __init__(self):
        super().__init__()
        self.pRef = None
        self.median_fitness = None

    def set_pRef(self, pRef: PRef):
        self.pRef = pRef

        self.median_fitness = float(np.median(pRef.fitness_array))

    def __repr__(self):
        return "ChanceOfGood"

    def get_single_normalised_score(self, ps: PS) -> float:
        observations = self.pRef.fitnesses_of_observations(ps)
        if len(observations) == 0:
            return 0

        amount_which_are_better_than_median = sum([1 for observation in observations
                                                   if observation > self.median_fitness])

        return amount_which_are_better_than_median / len(observations)
