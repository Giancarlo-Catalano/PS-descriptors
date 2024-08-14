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



# mutation should be BitFlipMutation(...)
# crossover should be SimulatedBinaryCrossover(...)
# selection should be tournamentSelection
# crowding operator should be TODO