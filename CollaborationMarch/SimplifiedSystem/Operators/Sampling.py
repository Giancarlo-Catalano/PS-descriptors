import random

import numpy as np
from pymoo.operators.sampling.rnd import FloatRandomSampling


class LocalPSGeometricSampling(FloatRandomSampling):

    def generate_single_individual(self, n) -> np.ndarray:
        result_values = np.zeros(shape=n, dtype=bool)
        chance_of_success = 0.70
        while random.random() < chance_of_success:
            var_index = random.randrange(n)
            result_values[var_index] = True
        return result_values

    def _do(self, problem, n_samples, **kwargs):
        n = problem.n_var
        return np.array([self.generate_single_individual(n) for _ in range(n_samples)])



class GlobalPSGeometricSampling(FloatRandomSampling):

    def generate_single_individual(self, n, xu) -> np.ndarray:
        result_values = np.full(shape=n, fill_value=-1)  # the stars
        chance_of_success = 0.79
        while random.random() < chance_of_success:
            var_index = random.randrange(n)
            new_value = random.randrange(int(xu[var_index]) + 1)
            result_values[var_index] = new_value
        return result_values

    def _do(self, problem, n_samples, **kwargs):
        n, (xl, xu) = problem.n_var, problem.bounds()
        return np.array([self.generate_single_individual(n, xu) for _ in range(n_samples)])
