import random
from typing import Iterable, Any

import numpy as np
from pymoo.core.repair import Repair
from pymoo.operators.sampling.rnd import FloatRandomSampling

from Core.PS import PS, STAR
from PSMiners.PyMoo.CustomCrowding import PyMooCustomCrowding


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



class ObjectiveSpaceAvoidance(PyMooCustomCrowding):
    masks_to_avoid: list[np.ndarray]
    sigma_shared: float
    opt: Any


    @classmethod
    def ps_to_mask(cls, ps: PS) -> np.ndarray:
        return ps.values != STAR
    def __init__(self, to_avoid: Iterable[PS], sigma_shared: float = 0.5):
        super().__init__()

        self.masks_to_avoid = [self.ps_to_mask(ps) for ps in to_avoid]
        self.sigma_shared = sigma_shared
        self.opt = []

    def distance_metric(self, x: np.ndarray, mask: np.ndarray):
        overlap_count = np.sum(np.logical_and(x, mask), dtype=float)
        fixed_count = (np.sum(x) + np.sum(mask))/2

        if fixed_count < 1:
            return 1
        return 1 - (overlap_count / fixed_count)

    def is_too_close(self, x, mask) -> bool:
        return self.distance_metric(x, mask) < self.sigma_shared


    def get_crowding_score(self, x: np.ndarray) -> float:
        if len(self.masks_to_avoid) == 0:
            return 1
        amount_of_close = len([mask for mask in self.masks_to_avoid if self.is_too_close(x, mask)])
        return 1 - (amount_of_close / len(self.masks_to_avoid))


    def get_crowding_scores_of_front(self, all_F, n_remove, population, front_indexes) -> np.ndarray:
        scores = np.array([self.get_crowding_score(population[index].X) for index in front_indexes])

        self.opt = population[front_indexes]  # just to comply with Pymoo, ignore this
        return scores

# mutation should be BitFlipMutation(...)
# crossover should be SimulatedBinaryCrossover(...), probably set to 0
# selection should be tournamentSelection
# crowding operator should be UnexplainedCrowdingOperator



class ForceDifferenceMask(Repair):

    def _do(self, problem, Z, **kwargs):

        # assert(isinstance(problem, TMLocalRestrictedPymooProblem)) # including this requires a circular import
        difference_variables = problem.difference_variables

        def fix_row(row: np.ndarray):
            # we choose an item at random to activate
            to_activate  = random.choice(difference_variables)
            row[to_activate] = True

        which_need_fixing = problem.get_which_rows_satisfy_mask_constraint(Z)


        for row, needs_fixing in zip(Z, which_need_fixing):
            if needs_fixing:
                fix_row(row)

        return Z