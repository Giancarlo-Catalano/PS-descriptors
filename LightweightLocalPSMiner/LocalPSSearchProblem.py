import numpy as np
from pymoo.core.problem import Problem

from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS
from Core.PSMetric.Classic3 import Classic3PSEvaluator
from Core.PSMetric.Metric import Metric
from Core.SearchSpace import SearchSpace
from LightweightLocalPSMiner.FastPSEvaluator import FastPSEvaluator


class LocalPSPymooProblem(Problem):
    objectives_evaluator: FastPSEvaluator

    solution_to_explain: FullSolution

    def __init__(self,
                 solution_to_explain: FullSolution,
                 objectives_evaluator: FastPSEvaluator):
        self.solution_to_explain = solution_to_explain
        self.objectives_evaluator = objectives_evaluator

        n_var = len(solution_to_explain.values)
        lower_bounds = np.full(shape=n_var, fill_value=0)  # the stars
        upper_bounds = lower_bounds + 1
        super().__init__(n_var=n_var,
                         n_obj=3,
                         n_ieq_constr=0,
                         xl=lower_bounds,
                         xu=upper_bounds,
                         vtype=bool)

    def individual_to_ps(self, x):
        return PS(sol_value if x_value else -1 for (sol_value, x_value) in zip(self.solution_to_explain.values, x))

    def _evaluate(self, X, out, *args, **kwargs):
        """ I believe that since this class inherits from Problem, x should be a group of solutions, and not just one"""
        metrics = np.array([self.objectives_evaluator.get_S_MF_A(self.individual_to_ps(row)) for row in X])
        out["F"] = -metrics  # minus sign because it's a maximisation task
