from typing import Optional, Literal, Callable, TypeAlias

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from CollaborationMarch.SimplifiedSystem.Operators.Crossover import GlobalPSUniformCrossover
from CollaborationMarch.SimplifiedSystem.Operators.Mutation import GlobalPSUniformMutation
from CollaborationMarch.SimplifiedSystem.Operators.Sampling import GlobalPSGeometricSampling
from CollaborationMarch.SimplifiedSystem.ps_search_utils import construct_objectives_list, apply_culling_method, \
    run_pymoo_algorithm_with_checks
from Core.EvaluatedPS import EvaluatedPS
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS, STAR
from Core.PSMetric.FitnessQuality.SignificantlyHighAverage import MannWhitneyU
from Core.PSMetric.Linkage.TraditionalPerturbationLinkage import TraditionalPerturbationLinkage
from Core.PSMetric.Linkage.ValueSpecificMutualInformation import FasterSolutionSpecificMutualInformation
from Core.SearchSpace import SearchSpace
from LCS.Operators import LocalPSGeometricSampling
from LCS.PSFilter import keep_biggest, merge_pss_into_one, keep_middle, \
    keep_with_best_atomicity
from VarianceDecisionTree.optimised_variance_objective import SplitVarianceAndConsistency

PSObjective: TypeAlias = Callable[[PS], float]


class GlobalPSSearchTask(Problem):
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
                 original_problem_search_space: SearchSpace,
                 objectives: list[Callable],
                 unexplained_mask: Optional[np.ndarray] = None,
                 proportion_unexplained_that_needs_used: float = 0.01,  # at least
                 proportion_used_that_should_be_unexplained: float = 0.5):  # at least
        self.original_problem_search_space = original_problem_search_space
        self.objectives = objectives
        self.unexplained_mask = np.ones(shape=self.original_problem_search_space.amount_of_parameters,
                                        dtype=bool) if unexplained_mask is None else unexplained_mask
        self.difference_variables = np.arange(len(self.unexplained_mask))[self.unexplained_mask]  # gets the indexes

        self.proportion_unexplained_that_needs_used = proportion_unexplained_that_needs_used
        self.proportion_used_that_should_be_unexplained = proportion_used_that_should_be_unexplained

        # then the stuff to satisfy pymoo
        n_var = self.original_problem_search_space.amount_of_parameters
        lower_bounds = np.full(shape=n_var, fill_value=-1)  # the stars
        upper_bounds = self.original_problem_search_space.cardinalities + 1  ## the bounds are not inclusive
        super().__init__(n_var=n_var,
                         n_obj=len(self.objectives),
                         n_ieq_constr=1,
                         xl=lower_bounds,
                         xu=upper_bounds,
                         vtype=int)

    def individual_to_ps(self, x):
        return PS(x)  # TODO check

    def get_which_rows_satisfy_constraint(self, X: np.ndarray) -> np.ndarray:
        # for a ps with some fixed variables, there are
        #  F which are used in the PS
        #  U is the amount of unexplained variables
        #  H which are unexplained and used in the PS
        #  (a) (H/F)% is how many of the used variables are unexplained, should be greater than proportion alpha
        # -> H / F >= alpha <=> H >= F * alpha
        #  (b) (H/U)% is how many of the unexplained variables are used, should be greater than proportion beta
        # -> H / U >= beta <=> H >= U * beta

        f = np.sum(X != -1, axis=1)  # check
        u = len(self.difference_variables)
        h = np.sum(X[:, self.difference_variables], axis=1)

        threshold_h_A = f * self.proportion_used_that_should_be_unexplained
        threshold_h_B = u * self.proportion_unexplained_that_needs_used

        satisfies_A = h >= threshold_h_A
        satisfies_B = h >= threshold_h_B

        return np.logical_and(satisfies_A, satisfies_B)

    def get_metrics_for_ps(self, ps: PS) -> list[float]:
        return [objective(ps) for objective in self.objectives]

    def _evaluate(self, X, out, *args, **kwargs):
        """ I believe that since this class inherits from Problem, x should be a group of solutions, and not just one"""
        metrics = np.array([self.get_metrics_for_ps(self.individual_to_ps(row)) for row in X])
        out["F"] = metrics

        out["G"] = 0.5 - self.get_which_rows_satisfy_constraint(
            X)  # if the constraint is satisfied, it is negative (which is counterintuitive)


def find_ps_in_problem(original_problem_search_space: SearchSpace,
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
    objectives = construct_objectives_list(metrics_str=metrics, pRef=pRef, problem=problem)

    if len(objectives) == 0:
        raise Exception("Somehow there are no objectives")

    # construct the optimisation problem instance
    problem = GlobalPSSearchTask(
        original_problem_search_space=original_problem_search_space,
        objectives=objectives,
        unexplained_mask=unexplained_mask,
        proportion_unexplained_that_needs_used=proportion_unexplained_that_needs_used,
        proportion_used_that_should_be_unexplained=proportion_used_that_should_be_unexplained)

    # the next line of code is a bit odd, but it works! It uses a GA if there is one objective
    algorithm = (GA if len(objectives) < 2 else NSGA2)(pop_size=population_size,
                                                       sampling=GlobalPSGeometricSampling(),
                                                       crossover=GlobalPSUniformCrossover(prob=0.3),  #
                                                       mutation=GlobalPSUniformMutation(prob=1 / problem.n_var, search_space = original_problem_search_space),
                                                       eliminate_duplicates=True)

    pss = run_pymoo_algorithm_with_checks(problem=problem,
                                          algorithm=algorithm,
                                          reattempts_when_fail=reattempts_when_fail,
                                          ps_budget=ps_budget,
                                          verbose=verbose)

    return apply_culling_method(pss, culling_method)
