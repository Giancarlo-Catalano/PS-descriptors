from typing import Iterable, Optional

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize

from BenchmarkProblems.Trapk import Trapk
from Core.EvaluatedPS import EvaluatedPS
from Core.FullSolution import FullSolution
from Core.PS import PS
from LCS.Operators import LocalPSGeometricSampling, ObjectiveSpaceAvoidance, ForceDifferenceMaskByActivatingOne, \
    ForceDifferenceMaskByActivatingAll
from LCS.PSEvaluator import GeneralPSEvaluator
from LCS.PSFilter import keep_with_lowest_dependence, keep_biggest


class LocalRestrictedPymooProblem(Problem):
    objectives_evaluator: GeneralPSEvaluator
    solution_to_explain: FullSolution
    must_include_mask: np.ndarray

    amount_of_metrics_in_use = 3

    def __init__(self,
                 solution_to_explain: FullSolution,
                 must_include_mask: np.ndarray,
                 objectives_evaluator: GeneralPSEvaluator):
        self.difference_variables = np.arange(len(must_include_mask))[must_include_mask]
        self.solution_to_explain = solution_to_explain
        self.objectives_evaluator = objectives_evaluator
        self.objectives_evaluator.set_solution(solution_to_explain)

        n_var = len(solution_to_explain.values)
        lower_bounds = np.full(shape=n_var, fill_value=0)  # the stars
        upper_bounds = lower_bounds + 1
        super().__init__(n_var=n_var,
                         n_obj=self.amount_of_metrics_in_use,
                         n_ieq_constr=1,
                         xl=lower_bounds,
                         xu=upper_bounds,
                         vtype=bool)

    def individual_to_ps(self, x):
        return PS(sol_value if x_value else -1 for (sol_value, x_value) in zip(self.solution_to_explain.values, x))

    def get_which_rows_satisfy_mask_constraint(self, X: np.ndarray) -> np.ndarray:
        return np.any(X[:, self.difference_variables], axis=1)

    def get_metrics_for_ps(self, ps: PS) -> list[float]:
        atomicity = self.objectives_evaluator.traditional_linkage.get_atomicity(ps)

        simplicity = len(ps) - ps.fixed_count()

        def use_dependency():
            dependency = self.objectives_evaluator.traditional_linkage.get_dependence(ps)
            return [-simplicity, dependency, -atomicity]

        def use_mean_fitness():
            mean_fitness = self.objectives_evaluator.mean_fitness_metric.get_single_score(ps)
            return [-simplicity, -mean_fitness, -atomicity]

        def use_p_value():
            p_value = self.objectives_evaluator.fitness_p_value_metric.get_single_score(ps)
            return [-simplicity, p_value, -atomicity]

        return use_p_value()

    def _evaluate(self, X, out, *args, **kwargs):
        """ I believe that since this class inherits from Problem, x should be a group of solutions, and not just one"""
        metrics = np.array([self.get_metrics_for_ps(self.individual_to_ps(row)) for row in X])
        out["F"] = metrics

        out["G"] = 0.5 - self.get_which_rows_satisfy_mask_constraint(
            X)  # if the constraint is satisfied, it is negative (which is counterintuitive)


def local_constrained_ps_search(to_explain: FullSolution,
                                background_solution: FullSolution,
                                ps_evaluator: GeneralPSEvaluator,
                                ps_budget: int,
                                population_size: int,
                                only_return_least_dependent: bool = False,
                                only_return_biggest: bool = False,
                                verbose=True) -> list[PS]:
    must_include_mask: np.ndarray = to_explain.values != background_solution.values

    problem = LocalRestrictedPymooProblem(solution_to_explain=to_explain,
                                          objectives_evaluator=ps_evaluator,
                                          must_include_mask=must_include_mask)

    algorithm = NSGA2(pop_size=population_size,
                      sampling=LocalPSGeometricSampling(),
                      crossover=SimulatedBinaryCrossover(prob=0.3),
                      mutation=BitflipMutation(prob=1 / problem.n_var),
                      eliminate_duplicates=True,
                      # survival=ObjectiveSpaceAvoidance(pss_to_avoid), # not done here.
                      repair=ForceDifferenceMaskByActivatingOne(),
                      )

    max_attempts = 5
    res = None
    for attempt in range(max_attempts):  # we have to do this because sometimes the search produces Nones
        res = minimize(problem,
                       algorithm,
                       termination=('n_evals', ps_budget),
                       verbose=verbose)

        if (res.X is not None) and (res.F is not None) and (res.G is not None):
            break
        elif verbose:
            print("The PS search returned all None, so we're attempting again..")
    else:  # yes I am happy to use a for else
        raise Exception("For some mysterious reason, Pymoo keeps returning None instead of search results...")

    result_pss = [EvaluatedPS(problem.individual_to_ps(values).values, metric_scores=-ms)
                  for values, ms, satisfies_constr in zip(res.X, res.F, res.G)
                  if satisfies_constr]

    if verbose:
        pss_that_dont_satisfy = [EvaluatedPS(problem.individual_to_ps(values).values, metric_scores=-ms)
                                 for values, ms, satisfies_constr in zip(res.X, res.F, res.G)
                                 if not satisfies_constr]
        if pss_that_dont_satisfy:
            print("The pss which don't satisfy the constraints that appeared in the last generation are")
            for ps in pss_that_dont_satisfy:
                print(f"\t{ps}")

    if len(result_pss) == 0:
        if verbose:
            print("Count not find any PSs that satisfied the constraint, so I'll relax it")
            result_pss = [EvaluatedPS(problem.individual_to_ps(values).values, metric_scores=ms)
                          for values, ms, satisfies_constr in zip(res.X, res.F, res.G)]
            if len(result_pss) == 0:
                print("Even after relaxing the constraint, no solutions were found...")
                raise Exception("No matching PSs could be found")  # perhaps I should return nothing?

    if verbose:
        print("The results of the search are")
        for ps in result_pss:
            print(ps)

    if only_return_least_dependent:
        result_pss = keep_with_lowest_dependence(result_pss, ps_evaluator.traditional_linkage)

    if only_return_biggest:
        result_pss = keep_biggest(result_pss)

    return result_pss
