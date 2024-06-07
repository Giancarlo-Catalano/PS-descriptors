import numpy as np
from pymoo.core.problem import Problem

from Core.PRef import PRef
from Core.PS import PS
from Core.PSMetric.Classic3 import Classic3PSEvaluator
from Core.SearchSpace import SearchSpace


class PSPyMooProblem(Problem):
    pRef: PRef
    objectives_evaluator: Classic3PSEvaluator


    def __init__(self,
                 pRef: PRef):
        self.pRef = pRef
        self.objectives_evaluator = Classic3PSEvaluator(self.pRef)

        lower_bounds = np.full(shape=self.search_space.amount_of_parameters, fill_value=-1)  # the stars
        upper_bounds = self.search_space.cardinalities - 1
        super().__init__(n_var = self.search_space.amount_of_parameters,
                         n_obj=3,
                         n_ieq_constr=0,
                         xl=lower_bounds,
                         xu=upper_bounds,
                         vtype=int)

    @property
    def search_space(self) -> SearchSpace:
        return self.pRef.search_space

    def individual_to_ps(self, x):
        return PS(x)

    def _evaluate(self, X, out, *args, **kwargs):
        """ I believe that since this class inherits from Problem, x should be a group of solutions, and not just one"""
        metrics = np.array([self.objectives_evaluator.get_S_MF_A(self.individual_to_ps(row)) for row in X])
        out["F"] = -metrics  # minus sign because it's a maximisation task

        # sharing_values = get_sharing_scores(X, 0.5, 12)
        # out["F"] /= (sharing_values.reshape((-1, 1)))+1




def pymoo_result_to_pss(res) -> list[PS]:
    return [PS(row) for row in res.X]


