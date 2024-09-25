import numpy as np

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.FullSolution import FullSolution
from Core.SearchSpace import SearchSpace


class WithLinkage(BenchmarkProblem):


    def __init__(self):
        super().__init__(SearchSpace([4 for i in range(19)]))

    def fitness_function(self, fs: FullSolution) -> float:
        v = fs.values
        univariate_additive = v[0]+2*v[1]+v[2]**2
        univariate_multiplicative = v[3]*v[4]*(v[5]**2)

        bivariate_mod = ((v[6]+v[7]) % 3) * 50
        bivariate_cond = v[8]**v[9] if v[8] < 3 and v[9] < 3 else v[8] * v[9]

        multivariate_mod = ((v[10]+v[11]+v[12]+v[13]) % 2) * 100
        multivariate_cond = (np.min(v[14:19]) + 1) ** 4

        return float(sum([univariate_additive,
                          univariate_multiplicative,
                          bivariate_mod,
                          bivariate_cond,
                          multivariate_mod,
                          multivariate_cond]))
