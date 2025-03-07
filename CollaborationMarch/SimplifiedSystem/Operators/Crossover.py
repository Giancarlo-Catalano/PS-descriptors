import random

import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.variable import Real, get
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling

from Core.SearchSpace import SearchSpace


class GlobalPSUniformCrossover(Crossover):

    def __init__(self,
                 n_offsprings=2,
                 prob: float = 0.5,
                 **kwargs):
        super().__init__(2, n_offsprings, prob = prob, **kwargs)


    @classmethod
    def ps_uniform_crossover(self, mother: np.ndarray, father: np.ndarray):
        daughter = mother.copy()
        son = father.copy()

        for index, _ in enumerate(daughter):
            if random.random() < 0.5:
                daughter[index], son[index] = son[index], daughter[index]

        return (daughter, son)

    def _do(self, problem, X, **kwargs):
        _, n_matings, _ = X.shape

        children = np.array([self.ps_uniform_crossover(mother, father)
                             for mother, father in zip(X[0], X[1])])

        return np.swapaxes(children, 0, 1)



