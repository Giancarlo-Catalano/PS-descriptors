from typing import Optional, Literal, Callable, TypeAlias

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.EvaluatedPS import EvaluatedPS
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS, STAR
from Core.PSMetric.FitnessQuality.SignificantlyHighAverage import MannWhitneyU
from Core.PSMetric.Linkage.TraditionalPerturbationLinkage import TraditionalPerturbationLinkage
from Core.PSMetric.Linkage.ValueSpecificMutualInformation import FasterSolutionSpecificMutualInformation
from LCS.Operators import LocalPSGeometricSampling
from LCS.PSFilter import keep_biggest, merge_pss_into_one, keep_middle, \
    keep_with_best_atomicity
from VarianceDecisionTree.optimised_variance_objective import SplitVarianceAndConsistency

PSObjective: TypeAlias = Callable[[PS], float]


class SimplePSSearchTask(Problem):
    solution_to_explain: FullSolution
    unexplained_mask: np.ndarray
    proportion_unexplained_that_needs_used: float  # alpha
    proportion_used_that_should_be_unexplained: float  # beta

    objectives: list[Callable]

    difference_variables: list[int]

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
                 objectives: list[Callable],
                 unexplained_mask: Optional[np.ndarray] = None,
                 proportion_unexplained_that_needs_used: float = 0.01,  # at least
                 proportion_used_that_should_be_unexplained: float = 0.5):  # at least
        self.solution_to_explain = solution_to_explain
        self.objectives = objectives
        self.unexplained_mask = np.ones(shape=len(solution_to_explain),
                                        dtype=bool) if unexplained_mask is None else unexplained_mask
        self.difference_variables = np.arange(len(self.unexplained_mask))[self.unexplained_mask]  # gets the indexes

        self.proportion_unexplained_that_needs_used = proportion_unexplained_that_needs_used
        self.proportion_used_that_should_be_unexplained = proportion_used_that_should_be_unexplained

        # then the stuff to satisfy pymoo
        n_var = len(solution_to_explain.values)
        lower_bounds = np.full(shape=n_var, fill_value=0)  # the stars
        upper_bounds = lower_bounds + 1
        super().__init__(n_var=n_var,
                         n_obj=len(self.objectives),
                         n_ieq_constr=1,
                         xl=lower_bounds,
                         xu=upper_bounds,
                         vtype=bool)

    def individual_to_ps(self, x):
        return PS(sol_value if x_value == 1 else -1 for (sol_value, x_value) in zip(self.solution_to_explain.values, x))

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
        #return np.ones(len(X))

    def get_metrics_for_ps(self, ps: PS) -> list[float]:
        return [objective(ps) for objective in self.objectives]

    def _evaluate(self, X, out, *args, **kwargs):
        """ I believe that since this class inherits from Problem, x should be a group of solutions, and not just one"""
        metrics = np.array([self.get_metrics_for_ps(self.individual_to_ps(row)) for row in X])
        out["F"] = metrics

        out["G"] = 0.5 - self.get_which_rows_satisfy_constraint(
            X)  # if the constraint is satisfied, it is negative (which is counterintuitive)



def construct_objectives_list(metrics_str: str,
                              pRef: PRef,
                              solution: FullSolution,
                              problem: Optional[BenchmarkProblem] = None,
                              ):
    metrics_list_str = metrics_str.split()
    objectives = []

    def simplicity(ps: PS) -> float:
        return -float(np.sum(ps.values == STAR))

    if "simplicity" in metrics_list_str:
        objectives.append(simplicity)

    if "ground_truth_atomicity" in metrics_list_str:
        ground_truth_atomicity_metric = TraditionalPerturbationLinkage(problem)
        ground_truth_atomicity_metric.set_solution(solution)

        def ground_truth_atomicity(ps: PS) -> float:
            return -ground_truth_atomicity_metric.get_atomicity(ps)

        objectives.append(ground_truth_atomicity)

    if "estimated_atomicity" in metrics_list_str:
        estimated_atomicity_metric = FasterSolutionSpecificMutualInformation()
        estimated_atomicity_metric.set_pRef(pRef)
        estimated_atomicity_metric.set_solution(solution)

        def estimated_atomicity(ps: PS) -> float:
            return -estimated_atomicity_metric.get_atomicity(ps)

        objectives.append(estimated_atomicity)

    fitness_consistency = MannWhitneyU()
    fitness_consistency.set_pRef(pRef)



    if "consistency" in metrics_list_str or "variance" in metrics_list_str:
        variance_and_consistency_metric = SplitVarianceAndConsistency(pRef)

        def variance(ps: PS) -> float:
            variance_and_consistency_metric.evaluate(ps)
            return variance_and_consistency_metric.get_split_variance(ps)
            # return variance_metric.get_single_score(ps)

        def consistency(ps: PS) -> float:
            # always needs to be called AFTER variance.
            return variance_and_consistency_metric.get_consistency(ps)

        if "variance" in metrics_list_str:
            objectives.append(variance)
        if "consistency" in metrics_list_str:
            objectives.append(consistency)

    return objectives

def find_ps_in_solution(to_explain: FullSolution,
                        pRef: PRef,
                        ps_budget: int,
                        population_size: int = 100,
                        proportion_unexplained_that_needs_used: float = 0.01,
                        proportion_used_that_should_be_unexplained: float = 0.5,
                        culling_method=Optional[Literal["biggest", "least_dependent", "overlap"]],
                        reattempts_when_fail: int = 1,
                        unexplained_mask: Optional[np.ndarray] = None,
                        problem: Optional[BenchmarkProblem] = None,
                        metrics: str = "variance",
                        verbose=True) -> list[PS]:

    objectives = construct_objectives_list(metrics, pRef, to_explain, problem)

    if len(objectives) == 0:
        raise Exception("Somehow there are no objectives")


    # construct the optimisation problem instance
    problem = SimplePSSearchTask(solution_to_explain=to_explain,
                                 objectives=objectives,
                                 unexplained_mask=unexplained_mask,
                                 proportion_unexplained_that_needs_used=proportion_unexplained_that_needs_used,
                                 proportion_used_that_should_be_unexplained=proportion_used_that_should_be_unexplained)

    # the next line of code is a bit odd, but it works!
    algorithm = (GA if len(objectives) < 2 else NSGA2)(pop_size=population_size,
                   sampling=LocalPSGeometricSampling(),
                   crossover=SimulatedBinaryCrossover(prob=0.3),
                   mutation=BitflipMutation(prob=1 / problem.n_var),
                   eliminate_duplicates=True)

    def run_and_get_results() -> list[EvaluatedPS]:
        res = minimize(problem,
                       algorithm,
                       termination=('n_evals', ps_budget),
                       verbose=verbose)

        if (res.X is None) or (res.F is None) or (res.G is None):
            print("Result had some Nones")
            return []

        if len(res.X.shape) == 1:
            result_pss = [EvaluatedPS(problem.individual_to_ps(res.X).values, metric_scores=res.F)]  # if there is only one result, the array has a different shape...
        else:
            result_pss = [EvaluatedPS(problem.individual_to_ps(values).values, metric_scores=ms)
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
            return keep_biggest(pss)#return keep_with_best_atomicity(keep_biggest(pss))
        case "overlap":
            return [merge_pss_into_one(pss)]
        case "elbow":
            return keep_middle(pss)
        case _:
            raise Exception(f"The culling method {culling_method} was not recognised")
