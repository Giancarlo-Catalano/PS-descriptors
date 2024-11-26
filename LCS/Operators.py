import random
from typing import Iterable, Any, Optional

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



class ForceDifferenceMaskByActivatingOne(Repair):

    def add_one_where_needed(self, problem, Z, **kwargs):

        # assert(isinstance(problem, TMLocalRestrictedPymooProblem)) # including this requires a circular import
        difference_variables = problem.difference_variables

        def fix_row(row: np.ndarray):
            # we choose an item at random to activate
            to_activate  = random.choice(difference_variables)
            row[to_activate] = True

        which_rows_satisfy = problem.get_which_rows_satisfy_constraint(Z)


        for row, satisfied in zip(Z, which_rows_satisfy):
            if not satisfied:
                fix_row(row)

        return Z


    def _do_unsafe(self, problem, Z, **kwargs) -> (np.ndarray, bool):
        # this version has every difference variable turned on with a 50% chance.
        # The issue is that it is not guaranteed to make the solutions satisfy the constraint...
        # the second value in the tuple indicates whether Z was modified or not)

        which_rows_dont_satisfy = ~problem.get_which_rows_satisfy_constraint(Z)


        quantity_that_need_fixing = np.sum(which_rows_dont_satisfy)
        print(f"{quantity_that_need_fixing = }")
        if quantity_that_need_fixing < 1:
            return (Z, False)
        quantity_vars_in_difference = len(problem.difference_variables)

        # every difference variable has a 50/50 chance of being activated
        new_assignments = np.random.random((quantity_that_need_fixing, quantity_vars_in_difference)) > 0.5
        new_assignments[~np.any(new_assignments, axis=1)] = True

        Z[which_rows_dont_satisfy][:, problem.difference_variables] = new_assignments
        return (Z, True)

    def _do(self, problem, Z, **kwargs):
        ## why isn't this working?!!!
        unfeasible_rows = ~np.any(Z[:, problem.difference_variables], axis=1)
        cells_to_modify = np.ix_(unfeasible_rows, problem.difference_variables) # otherwise the assignment doesn't work
        Z[cells_to_modify] = True
        #Z[~np.any(Z[:, problem.difference_variables], axis=1)][:, problem.difference_variables] = True

        return Z
        # # simply does _do_unsafe until it is safe
        # while True:
        #     Z, was_modified = self._do_unsafe(problem, Z)
        #     if not was_modified:
        #         break
        #
        # return Z


class ForceDifferenceMaskByActivatingAll(Repair):

    def _do(self, problem, Z, **kwargs):

        Z[:, problem.difference_variables] = True
        return Z


class ForceDifferenceMarkByActivatingTheRightAmount(Repair):
    alpha: float
    beta: float



