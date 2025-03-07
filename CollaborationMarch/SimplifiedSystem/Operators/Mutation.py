import random

import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.variable import Real, get
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling

from Core.SearchSpace import SearchSpace


class GlobalPSUniformMutation(Mutation):
    search_space: SearchSpace
    single_point_probability: float

    def __init__(self, search_space: SearchSpace, prob=None):
        self.search_space = search_space
        self.single_point_probability = 1 / self.search_space.amount_of_parameters
        super().__init__(
            prob=0.9 if prob is None else prob)  # no idea what's supposed to be there, but it used to say 0.9 by default..

    def mutate_single_individual(self, x: np.ndarray) -> np.ndarray:
        result_values = x.copy()
        for index, _ in enumerate(result_values):
            if random.random() < self.single_point_probability:
                if random.random() < 0.5:
                    new_value = -1  # sets it to a *
                else:
                    new_value = random.randrange(self.search_space.cardinalities[index])
                result_values[index] = new_value

        return result_values

    def _do(self, problem, X, params=None, **kwargs):
        result_values = X.copy()
        for index, row in enumerate(result_values):
            result_values[index] = self.mutate_single_individual(row)

        return result_values
