import random

import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.variable import Real, get
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling

from Core.SearchSpace import SearchSpace




class PSGeometricSampling(FloatRandomSampling):

    def generate_single_individual(self, n, xu) -> np.ndarray:
        result_values = np.full(shape=n, fill_value=-1)  # the stars
        chance_of_success = 0.79
        while random.random() < chance_of_success:
            var_index = random.randrange(n)
            new_value = random.randrange(int(xu[var_index])+1)
            result_values[var_index] = new_value
        return result_values

    def _do(self, problem, n_samples, **kwargs):
        n, (xl, xu) = problem.n_var, problem.bounds()
        return np.array([self.generate_single_individual(n, xu) for _ in range(n_samples)])



class PSUniformSampling(FloatRandomSampling):

    def generate_single_individual(self, n, xu) -> np.ndarray:
        return np.array([-1 if random.random() < 0.5 else random.randrange(cardinality+1) for cardinality in xu])

    def _do(self, problem, n_samples, **kwargs):
        n, (xl, xu) = problem.n_var, problem.bounds()
        return np.array([self.generate_single_individual(n, xu) for _ in range(n_samples)])


# ---------------------------------------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------------------------------------


class PSUniformMutation(Mutation):
    search_space: SearchSpace
    single_point_probability: float

    def __init__(self, search_space: SearchSpace, prob=None):
        self.search_space = search_space
        self.single_point_probability = 1 / self.search_space.amount_of_parameters
        super().__init__(prob=0.9 if prob is None else prob)  # no idea what's supposed to be there, but it used to say 0.9 by default..


    def mutate_single_individual(self, x: np.ndarray) -> np.ndarray:
        result_values = x.copy()
        for index, _ in enumerate(result_values):
            if random.random() < self.single_point_probability:
                if random.random() < 0.5:
                    new_value = -1
                else:
                    new_value = random.randrange(self.search_space.cardinalities[index])
                previous_value = result_values[index]
                result_values[index] = new_value

        return result_values

    def _do(self, problem, X, params=None, **kwargs):
        result_values = X.copy()
        for index, row in enumerate(result_values):
            result_values[index] = self.mutate_single_individual(row)

        return result_values


def ps_uniform_crossover(mother:np.ndarray, father:np.ndarray):
    daughter = mother.copy()
    son = father.copy()

    for index, _ in enumerate(daughter):
        if random.random() < 0.5:
            daughter[index], son[index] = son[index], daughter[index]

    return (daughter, son)


# ---------------------------------------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------------------------------------


class PSUniformCrossover(Crossover):

    def __init__(self,
                 n_offsprings=2,
                 **kwargs):
        super().__init__(2, n_offsprings, prob = 0.5, **kwargs)


    def _do(self, problem, X, **kwargs):
        _, n_matings, _ = X.shape


        children = np.array([ps_uniform_crossover(mother, father)
                             for mother, father in zip(X[0], X[1])])

        return np.swapaxes(children, 0, 1)


