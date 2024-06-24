
"""
This file is uniquely to implement the function get_S_MF_A,
which stands for get Simplicity, Mean Fitness, Atomicity

In simple terms, there is a lot of redundancy in calculating the various observations for a ps for these 3 metrics,
and by calculating the PRefs together we can save a lot of time.
"""
from typing import Optional

import numpy as np
from numba import njit

import utils
from Core.PRef import PRef
from Core.PS import PS, STAR
from Core.PSMetric.Additivity import MutualInformation
from Core.PSMetric.Metric import Metric
from Core.custom_types import ArrayOfFloats


@njit
def filter_by_var_val(fsm: np.ndarray,
                      fitnesses,
                      var: int,
                      val: int) -> (np.ndarray, np.ndarray, np.ndarray):
    which = fsm[:, var] == val
    new_fsm = fsm[which]
    if fitnesses is None:
        new_fitnesses = None
    else:
        new_fitnesses = fitnesses[which]
    return new_fsm, new_fitnesses

class RowsOfPRef:
    fsm: np.ndarray
    fitnesses: Optional[ArrayOfFloats]

    def __init__(self, fsm: np.ndarray, fitnesses: Optional[ArrayOfFloats]):
        self.fsm = fsm
        self.fitnesses = fitnesses

    @classmethod
    def all_from_pRef(cls, pRef: PRef):
        fsm = pRef.full_solution_matrix.copy()
        fitnesses = pRef.fitness_array.copy()
        return cls(fsm, fitnesses)

    def filter_by_var_val(self, var: int, val: int):
        self.fsm, self.fitnesses = filter_by_var_val(self.fsm,
                                                                                self.fitnesses,
                                                                                var,
                                                                                val)


    def get_mean_fitness(self) -> float:
        if self.fitnesses is None:
            raise ValueError("in RowsOfPRef, fitnesses is None")

        if len(self.fitnesses) == 0:
            return -np.inf
        return np.average(self.fitnesses)
    def copy(self):
        return RowsOfPRef(self.fsm, self.fitnesses)


class PSSearchMetricsEvaluator:
    pRef: PRef
    used_evaluations: int

    alternative_atomicity_evaluator: Metric


    def __init__(self, pRef: PRef):
        self.pRef = pRef

        self.used_evaluations = 0


        self.alternative_atomicity_evaluator = MutualInformation()
        self.alternative_atomicity_evaluator.set_pRef(pRef)


    def mf_of_rows(self, which_rows: RowsOfPRef)->float:
        return which_rows.get_mean_fitness()

    def get_simplicity_of_PS(self, ps: PS) -> float:
        return float(np.sum(ps.values == STAR))

    def get_relevant_rows_for_ps(self, ps: PS) -> (RowsOfPRef, list[RowsOfPRef]):
        """Returns the mean rows for ps, and the rows for the simplifications of ps"""

        """Eg for * 1 2 3 *, it returns
           rows(* 1 2 3*), [rows(* * 2 3), rows(* 1 * 3 *), rows(* 1 2 * *)]
        """

        def subset_where_column_has_value(superset: RowsOfPRef, variable: int, value: int) -> RowsOfPRef:
            result = superset.copy()
            result.filter_by_var_val(variable, value)
            return result


        with_all_fixed = RowsOfPRef.all_from_pRef(self.pRef)
        except_one_fixed = []

        for var in ps.get_fixed_variable_positions():
            value = ps[var]
            with_all_fixed = subset_where_column_has_value(with_all_fixed, var, value)

        return with_all_fixed, except_one_fixed



    def get_S_MF_A(self, ps: PS, invalid_value: float = 0) -> np.ndarray:   # it is 3 floats
        """this one is normalised"""
        self.used_evaluations += 1
        rows_all_fixed, excluding_one = self.get_relevant_rows_for_ps(ps)


        simplicity = self.get_simplicity_of_PS(ps)

        mean_fitness = self.mf_of_rows(rows_all_fixed)
        atomicity = self.alternative_atomicity_evaluator.get_single_score(ps)

        if not np.isfinite(mean_fitness) or np.isnan(mean_fitness):
            mean_fitness = invalid_value
        if not np.isfinite(atomicity):
            mean_fitness = invalid_value
        return np.array([simplicity, mean_fitness, atomicity])


