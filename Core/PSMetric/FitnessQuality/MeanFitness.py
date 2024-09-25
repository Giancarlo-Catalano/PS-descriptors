from typing import Optional

import numpy as np

from Core.PRef import PRef
from Core.PS import PS
from Core.PSMetric.Metric import Metric


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
            # warnings.warn(f"The passed Core {ps} has no observations, and thus the MeanFitness could not be calculated")
            return -1

        return np.average(observed_fitnesses)

    def get_single_score(self, ps: PS) -> float:
        assert(self.pRef is not None)  # which would mean that set_pRef has not been called
        observed_fitnesses = self.pRef.fitnesses_of_observations(ps)
        if len(observed_fitnesses) == 0:
            # warnings.warn(f"The passed Core {ps} has no observations, and thus the MeanFitness could not be calculated")
            return 0

        return np.average(observed_fitnesses)


    def get_single_normalised_score(self, ps: PS) -> float:
        observed_fitnesses = self.normalised_pRef.fitnesses_of_observations(ps)
        if len(observed_fitnesses) == 0:
            # warnings.warn(f"The passed Core {ps} has no observations, and thus the MeanFitness could not be calculated")
            return 0

        return np.average(observed_fitnesses)



class FitnessDelta(Metric):
    pRef: Optional[PRef]

    total_fitness: Optional[float]


    def __init__(self):
        self.pRef = None
        self.total_fitness = None
        super().__init__()

    def __repr__(self):
        return "FitnessDelta"


    def set_pRef(self, pRef: PRef):
        self.pRef = pRef
        self.total_fitness = np.sum(pRef.fitness_array)


    def get_mean_fitness_delta(self, ps: PS) -> float:
        fitnesses_when_present = self.pRef.fitnesses_of_observations(ps)
        if len(fitnesses_when_present) == 0:
            # no datapoints match the solution
            return 0 # TODO consider a better panic value
        if len(fitnesses_when_present) == self.pRef.sample_size:
            # all datapoints match the solution
            return 0

        sum_of_fitnesses_when_present = np.sum(fitnesses_when_present)
        sum_of_fitnesses_when_absent = self.total_fitness - sum_of_fitnesses_when_present
        mean_fitness_when_present = sum_of_fitnesses_when_present / len(fitnesses_when_present)
        mean_fitness_when_absent = sum_of_fitnesses_when_absent / (self.pRef.sample_size - len(fitnesses_when_present))

        return mean_fitness_when_present - mean_fitness_when_absent


    def get_single_score(self, ps: PS) -> float:
        return self.get_mean_fitness_delta(ps)


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
