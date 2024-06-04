import itertools
from typing import Optional

import numpy as np

import utils
from Core.PRef import PRef
from Core.PS import PS
from Core.PSMetric.Metric import Metric
from Core.custom_types import ArrayOfFloats


class Additivity(Metric):
    pRef: Optional[PRef]


    def __init__(self, which: int):
        super().__init__()
        self.pRef = None
        self.which = which

    def set_pRef(self, pRef: PRef):
        self.pRef = pRef

    def __repr__(self):
            return f"Additivity({self.which})"

    def get_fitnesses_split_by_error(self, ps: PS) -> (float, float, float):
        fixed_vars = ps.get_fixed_variable_positions()
        relevant_rows_in_pRef = self.pRef.full_solution_matrix[:, fixed_vars]
        correct_values = ps.values[fixed_vars]
        error_counts = np.sum(relevant_rows_in_pRef != correct_values, axis=1)

        def mean_with_amount_of_errors(error_count: int) -> np.ndarray:
            return np.mean(self.pRef.fitness_array[error_counts == error_count])

        return mean_with_amount_of_errors(0), mean_with_amount_of_errors(1), mean_with_amount_of_errors(2)


    def get_single_score(self, ps: PS) -> float:
        order = ps.fixed_count()
        if order == 0:
            return 0.0
        no_err, one_err, two_err = self.get_fitnesses_split_by_error(ps)

        if order == 1:
            return 0.0  # but maybe it should be no_err - one_err

        if np.isnan(no_err) or np.isnan(one_err) or np.isnan(two_err):
            return 0.0


        alpha = 2 * (no_err - one_err)
        beta = (no_err - two_err)

        var_a = alpha - beta
        var_b = abs(alpha - beta)
        var_c = abs(alpha) - abs(beta)
        var_d = abs(abs(alpha) - abs(beta))
        # the below formula could be in an abs(.), but that would promote hindrance effects I think
        return [var_a, var_b, var_c, var_d][self.which]  # it could be simplified





class MeanError(Metric):
    pRef: Optional[PRef]

    def __init__(self):
        super().__init__()
        self.pRef = None

    def __repr__(self):
        return "SimplePerturbation"
    def set_pRef(self, pRef: PRef):
        self.pRef = pRef


    def get_perturbation_items_at_loci(self, ps: PS, a:int, b:int) -> (PS, PS, PS, PS):
        if ps[a] == -1 or ps[b] == -1:
            raise Exception("I need fixed vars!!!")

        p_ss = ps.copy()
        p_cs = ps.with_fixed_value(a, 1-ps[a])
        p_sc = ps.with_fixed_value(b, 1-ps[b])
        p_cc = p_cs.with_fixed_value(b, 1-ps[b])
        return p_ss, p_cs, p_sc, p_cc


    def get_perturbation_at_loci(self, ps: PS, a: int, b: int) -> float:
        p_ss, p_cs, p_sc, p_cc = self.get_perturbation_items_at_loci(ps, a, b)

        def m(input_ps: PS) -> float:
            return np.average(self.pRef.fitnesses_of_observations(input_ps))

        return abs(m(p_ss) + m(p_cc) - m(p_sc) - m(p_cs))


    def get_single_score(self, ps: PS) -> float:
        return -utils.get_mean_error(self.pRef.fitnesses_of_observations(ps))



