from typing import Iterable

import numpy as np
import pymoo
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize

from BenchmarkProblems.Trapk import Trapk
from Core.EvaluatedPS import EvaluatedPS
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS
from Core.PSMetric.MeanFitness import FitnessDelta
from Core.PSMetric.SignificantlyHighAverage import SignificantlyHighAverage, MannWhitneyU
from Core.PSMetric.ValueSpecificMutualInformation import SolutionSpecificMutualInformation, \
    FasterSolutionSpecificMutualInformation, NotValueSpecificMI
from LightweightLocalPSMiner.Operators import LocalPSGeometricSampling, ObjectiveSpaceAvoidance
from LightweightLocalPSMiner.TwoMetrics import TMEvaluator, TMLocalPymooProblem
from LinkageExperiments.LocalVarianceLinkage import LocalVarianceLinkage, BivariateLinkage


class TMLocalRestrictedPymooProblem(Problem):
    tm_evaluator: TMEvaluator
    solution_to_explain: FullSolution
    must_include_mask: np.ndarray

    def __init__(self,
                 solution_to_explain: FullSolution,
                 must_include_mask: np.ndarray,
                 objectives_evaluator: TMEvaluator):
        self.difference_variables = np.arange(len(must_include_mask))[must_include_mask]
        self.solution_to_explain = solution_to_explain
        self.objectives_evaluator = objectives_evaluator
        self.objectives_evaluator.set_solution(solution_to_explain)

        n_var = len(solution_to_explain.values)
        lower_bounds = np.full(shape=n_var, fill_value=0)  # the stars
        upper_bounds = lower_bounds + 1
        super().__init__(n_var=n_var,
                         n_obj=2,
                         n_ieq_constr=1,
                         xl=lower_bounds,
                         xu=upper_bounds,
                         vtype=bool)

    def individual_to_ps(self, x):
        return PS(sol_value if x_value else -1 for (sol_value, x_value) in zip(self.solution_to_explain.values, x))


    def _evaluate(self, X, out, *args, **kwargs):
        """ I believe that since this class inherits from Problem, x should be a group of solutions, and not just one"""
        metrics = np.array([self.objectives_evaluator.get_A_D(self.individual_to_ps(row)) for row in X])
        out["F"] = metrics


        rows_that_satisfy_mask = np.any(X[:, self.difference_variables], axis=1)
        out["G"] = rows_that_satisfy_mask - 0.5   # if the constraint is not satisfied, G = -0.5, otherwise +0.5

def local_restricted_tm_ps_search(to_explain: FullSolution,
                       must_include_mask: np.ndarray,
                       pss_to_avoid: Iterable[PS],
                       ps_evaluator: TMEvaluator,
                       ps_budget: int,
                       population_size: int,
                       verbose=True) -> list[PS]:
    problem = TMLocalRestrictedPymooProblem(solution_to_explain=to_explain,
                                            objectives_evaluator = ps_evaluator,
                                            must_include_mask = must_include_mask)

    algorithm = NSGA2(pop_size=population_size,
                      sampling=LocalPSGeometricSampling(),
                      crossover=SimulatedBinaryCrossover(prob=0.5),
                      mutation=BitflipMutation(prob=1 / problem.n_var),
                      eliminate_duplicates=True,
                      survival=ObjectiveSpaceAvoidance(pss_to_avoid)
                      )

    res = minimize(problem,
                   algorithm,
                   termination=('n_evals', ps_budget),
                   verbose=verbose)

    result_pss = [EvaluatedPS(problem.individual_to_ps(values).values, metric_scores=ms)
                  for values, ms in zip(res.X, res.F)]

    # then we arrange them in such a way that if a really good one is found, it will be the first one

    correct_sign_pss = []
    wrong_signs = []
    for ps in result_pss:
        a, d = ps.metric_scores
        if a < 0 and d > 0:
            correct_sign_pss.append(ps)
        else:
            wrong_signs.append(ps)

    if verbose:
        print(f"The search for PSs in {to_explain} resulted in ")
        for ps in result_pss:
            print("\t", ps)

    if len(correct_sign_pss) > 0:
        return correct_sign_pss
    else:
        return wrong_signs


def test_local_search():
    problem = Trapk(4, 4)

    already_mined = []  # [PS([1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]),
    # PS([-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1])]

    to_explain = FullSolution([1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0])

    pRef = problem.get_reference_population(10000)

    tm_evaluator = TMEvaluator(pRef)

    results = local_tm_ps_search(to_explain=to_explain,
                                 pss_to_avoid=already_mined,
                                 ps_evaluator=tm_evaluator,
                                 ps_budget=1000,
                                 population_size=50,
                                 verbose=True)

    print("results of search:")
    for result in results:
        print(result)

    print("The results by dependence-atomicity are")
    results.sort(key=lambda x: x.metric_scores[0] - x.metric_scores[1], reverse=True)
    for result in results:
        a, d = result.metric_scores
        if a < 0 and d > 0:
            print(result)