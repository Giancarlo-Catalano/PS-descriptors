
import random

import numpy as np
from pymoo.core.mutation import Mutation
from pymoo.operators.sampling.rnd import FloatRandomSampling

from Core.FullSolution import FullSolution
from Core.SearchSpace import SearchSpace


class PolishLocalUniformPSMutation(Mutation):
    solution_to_explain: FullSolution
    single_point_probability: float

    def __init__(self, solution_to_explain: FullSolution, prob=None):
        self.solution_to_explain = solution_to_explain
        self.single_point_probability = 1 / len(solution_to_explain)
        super().__init__(
            prob=0.9 if prob is None else prob)  # no idea what's supposed to be there, but it used to say 0.9 by default..

    def mutate_single_individual(self, x: np.ndarray) -> np.ndarray:
        result_values : np.ndarray = x.copy()
        where_mutate = np.random.random(result_values.shape) < self.single_point_probability
        result_values = np.logical_xor(result_values, where_mutate)
        return result_values

    def _do(self, problem, X, params=None, **kwargs):
        result_values = X.copy()
        for index, row in enumerate(result_values):
            result_values[index] = self.mutate_single_individual(row)

        return result_values
