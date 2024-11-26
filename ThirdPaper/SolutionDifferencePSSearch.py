from typing import Iterable, Optional, Literal

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
from LCS.PSFilter import keep_with_lowest_dependence, keep_biggest, merge_pss_into_one, keep_middle, \
    keep_with_best_atomicity


class PSWithinSolutionSearch(Problem):
    solution_to_explain: FullSolution
    unexplained_mask: np.ndarray
    objectives_evaluator: GeneralPSEvaluator
    proportion_unexplained_that_needs_used: float  # alpha
    proportion_used_that_should_be_unexplained: float  # beta

    difference_variables: list[int]

    amount_of_metrics_in_use = 3

    # NOTE:
    #  if you want the PS to completely ignore the already explained stuff, set beta to 1
    #  if you want the unexplained stuff to be completely contained in the PS, set alpha to 1
    # if alpha = 0.5, beta = 0.5, then
    # at least half of the PS is new stuff
    # at least half of the new stuff is in the PS
    # generally, you would want at least one new thing to be used, set alpha to > 0
    # generally, you would want most of the PS to contain new things, set beta to > 0.5

    def __init__(self,
                 solution_to_explain: FullSolution,
                 unexplained_mask: np.ndarray,
                 objectives_evaluator: GeneralPSEvaluator,
                 proportion_unexplained_that_needs_used: float,
                 proportion_used_that_should_be_unexplained: float):
        self.solution_to_explain = solution_to_explain
        self.objectives_evaluator = objectives_evaluator
        self.proportion_unexplained_that_needs_used = proportion_unexplained_that_needs_used
        self.proportion_used_that_should_be_unexplained = proportion_used_that_should_be_unexplained
        self.objectives_evaluator.set_solution(solution_to_explain)
        self.difference_variables = np.arange(len(unexplained_mask))[unexplained_mask]  # gets the indexes

        # then the stuff to satisfy pymoo
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

    def get_which_rows_satisfy_constraint(self, X: np.ndarray) -> np.ndarray:
        # for a ps with some fixed variables, there are
        #  F which are used in the PS
        #  U is the amount of unexplained variables
        #  H which are unexplained and used in the PS
        #  (a) (H/F)% is how many of the used variables are unexplained, should be greater than proportion alpha
        # -> H / F >= alpha <=> H >= F * alpha
        #  (b) (H/U)% is how many of the unexplained variables are used, should be greater than proportion beta
        # -> H / U >= beta <=> H >= U * beta

        f = np.sum(X, axis=1)
        u = len(self.difference_variables)
        h = np.sum(X[:, self.difference_variables], axis=1)

        threshold_h_A = f * self.proportion_used_that_should_be_unexplained
        threshold_h_B = u * self.proportion_unexplained_that_needs_used

        satisfies_A = h >= threshold_h_A
        satisfies_B = h >= threshold_h_B

        return np.logical_and(satisfies_A, satisfies_B)

    def get_metrics_for_ps(self, ps: PS) -> list[float]:
        atomicity = self.objectives_evaluator.traditional_linkage.get_atomicity(ps)
        simplicity = len(ps) - ps.fixed_count()
        p_value = self.objectives_evaluator.fitness_p_value_metric.get_single_score(ps)

        return [-simplicity, p_value, -atomicity]

    def _evaluate(self, X, out, *args, **kwargs):
        """ I believe that since this class inherits from Problem, x should be a group of solutions, and not just one"""
        metrics = np.array([self.get_metrics_for_ps(self.individual_to_ps(row)) for row in X])
        out["F"] = metrics

        out["G"] = 0.5 - self.get_which_rows_satisfy_constraint(
            X)  # if the constraint is satisfied, it is negative (which is counterintuitive)


def find_ps_in_solution(to_explain: FullSolution,
                        unexplained_mask: np.ndarray,
                        ps_evaluator: GeneralPSEvaluator,
                        ps_budget: int,
                        population_size: int,
                        culling_method=Optional[Literal["biggest", "least_dependent", "overlap"]],
                        reattempts_when_fail: int = 1,
                        proportion_unexplained_that_needs_used: Optional[float] = None,
                        proportion_used_that_should_be_unexplained: Optional[float] = None,
                        verbose=True) -> list[PS]:

    if proportion_unexplained_that_needs_used is None:
        proportion_unexplained_that_needs_used = 1 / len(to_explain)
    if proportion_used_that_should_be_unexplained is None:
        proportion_used_that_should_be_unexplained = 0.5
    # construct the optimisation problem instance
    problem = PSWithinSolutionSearch(solution_to_explain=to_explain,
                                     objectives_evaluator=ps_evaluator,
                                     unexplained_mask=unexplained_mask,
                                     proportion_unexplained_that_needs_used=proportion_unexplained_that_needs_used,
                                     proportion_used_that_should_be_unexplained=proportion_used_that_should_be_unexplained)

    # construct the optimisation algorithm
    algorithm = NSGA2(pop_size=population_size,
                      sampling=LocalPSGeometricSampling(),
                      crossover=SimulatedBinaryCrossover(prob=0.3),
                      mutation=BitflipMutation(prob=1 / problem.n_var),
                      eliminate_duplicates=True,
                      # survival=ObjectiveSpaceAvoidance(pss_to_avoid), # not done here.
                      repair=ForceDifferenceMaskByActivatingOne(),
                      )

    def run_and_get_results() -> list[EvaluatedPS]:
        res = minimize(problem,
                       algorithm,
                       termination=('n_evals', ps_budget),
                       verbose=verbose)

        if (res.X is None) or (res.F is None) or (res.G is None):
            print("Result had some Nones")
            return []

        result_pss = [EvaluatedPS(problem.individual_to_ps(values).values, metric_scores=-ms)
                      for values, ms in zip(res.X, res.F)]
        if len(result_pss) == 0:
            print("The pss population returned by NSGAII is empty...?")

        filtered_pss = [ps for ps, satisfies_constr in zip(result_pss, res.G)
                        if satisfies_constr]

        return filtered_pss if filtered_pss else result_pss

    for attempt in range(reattempts_when_fail):
        pss = run_and_get_results()
        if pss:
            break
    else:  # yes I am happy to use a for else
        raise Exception("For some mysterious reason, Pymoo keeps returning None instead of search results...")

    if verbose:
        print("The pss are ")
        for ps in pss:
            print("\t", ps)

    match culling_method:
        case None:
            return pss
        case "best_atomicity":
            return keep_biggest(keep_with_best_atomicity(pss))
        case "biggest":
            return keep_with_best_atomicity(keep_biggest(pss))
        case "least_dependent":
            return keep_with_lowest_dependence(pss, ps_evaluator.traditional_linkage)
        case "overlap":
            return [merge_pss_into_one(pss)]
        case "elbow":
            return keep_middle(pss)
        case _:
            raise Exception(f"The culling method {culling_method} was not recognised")
