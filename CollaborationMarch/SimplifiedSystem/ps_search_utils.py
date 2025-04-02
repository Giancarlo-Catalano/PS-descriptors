from typing import Optional, Literal, Callable, TypeAlias

import numpy as np
import pymoo.core.problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.algorithm import Algorithm
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.EvaluatedPS import EvaluatedPS
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS, STAR
from Core.PSMetric.FitnessQuality.MeanFitness import MeanFitness
from Core.PSMetric.FitnessQuality.SignificantlyHighAverage import MannWhitneyU
from Core.PSMetric.Linkage.TraditionalPerturbationLinkage import TraditionalPerturbationLinkage
from Core.PSMetric.Linkage.ValueSpecificMutualInformation import FasterSolutionSpecificMutualInformation
from Core.SearchSpace import SearchSpace
from LCS.Operators import LocalPSGeometricSampling
from LCS.PSFilter import keep_biggest, merge_pss_into_one, keep_middle, \
    keep_with_best_atomicity
from VarianceDecisionTree.SplitVariance import SplitVariance
from VarianceDecisionTree.optimised_variance_objective import SplitVarianceAndConsistency

PSObjective: TypeAlias = Callable[[PS], float]


def get_metric_function(metric_name: str,
                        pRef: Optional[PRef] = None,
                        solution: Optional[FullSolution] = None,
                        problem: Optional[BenchmarkProblem] = None,
                        search_space: Optional[SearchSpace] = None):

    # note that pymoo always wants to MINIMISE the objectives, so some signs have to be flipped
    def with_inverted_sign(func):
        def result_func(ps: PS):
            return -func(ps)
        return result_func

    if metric_name == "simplicity":
        def simplicity(ps: PS) -> float:
            return -float(np.sum(ps.values == STAR))
        return simplicity

    if metric_name == "mean_fitness":
        mean_fitness_evaluator = MeanFitness()
        mean_fitness_evaluator.set_pRef(pRef)
        return with_inverted_sign(mean_fitness_evaluator.get_single_score)

    if metric_name == "ground_truth_atomicity":
        ground_truth_atomicity_evaluator = TraditionalPerturbationLinkage(problem)
        ground_truth_atomicity_evaluator.set_solution(solution)
        return with_inverted_sign(ground_truth_atomicity_evaluator.get_atomicity)

    if metric_name == "estimated_atomicity":
        estimated_atomicity_metric = FasterSolutionSpecificMutualInformation()
        estimated_atomicity_metric.set_pRef(pRef)
        estimated_atomicity_metric.set_solution(solution)

        return with_inverted_sign(estimated_atomicity_metric.get_atomicity)

    if metric_name == "estimated_atomicity&evaluator":
        estimated_atomicity_metric = FasterSolutionSpecificMutualInformation()
        estimated_atomicity_metric.set_pRef(pRef)
        estimated_atomicity_metric.set_solution(solution)

        return estimated_atomicity_metric, with_inverted_sign(estimated_atomicity_metric.get_atomicity)


    if metric_name == "consistency":
        fitness_consistency_evaluator = MannWhitneyU()
        fitness_consistency_evaluator.set_pRef(pRef)
        return fitness_consistency_evaluator.get_single_score

    if metric_name == "variance":
        variance_evaluator = SplitVariance(pRef)
        return variance_evaluator.get_single_score

    return None


def construct_objectives_list(metrics_str: str,
                              pRef: PRef,
                              solution: Optional[FullSolution] = None,
                              problem: Optional[BenchmarkProblem] = None,
                              search_space: Optional[SearchSpace] = None,
                              ) -> list[PSObjective]:
    result = []
    for metric_name in metrics_str.split():
        converted = get_metric_function(metric_name, pRef, solution, problem, search_space)
        if converted is None:
            raise Exception(f"The metric was not recognised : {metric_name}")
        result.append(converted)
    return result


def construct_objectives_list_old(metrics_str: str,
                              pRef: PRef,
                              solution: Optional[FullSolution] = None,
                              problem: Optional[BenchmarkProblem] = None,
                              search_space: Optional[SearchSpace] = None,
                              ) -> list[PSObjective]:
    metrics_list_str = metrics_str.split()
    objectives = []

    def simplicity(ps: PS) -> float:
        return -float(np.sum(ps.values == STAR))

    if "simplicity" in metrics_list_str:
        objectives.append(simplicity)


    def mean_fitness(ps: PS) -> float:
        return

    if "mean_fitness" in metrics_list_str:
        objectives.append(mean_fitness)

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



def apply_culling_method(pss: list[EvaluatedPS], culling_method: str):
    match culling_method:
        case None:
            return pss
        case "best_atomicity":
            return keep_biggest(keep_with_best_atomicity(pss))
        case "biggest":
            return keep_biggest(pss)  # return keep_with_best_atomicity(keep_biggest(pss))
        case "overlap":
            return [merge_pss_into_one(pss)]
        case "elbow":
            return keep_middle(pss)
        case _:
            raise Exception(f"The culling method {culling_method} was not recognised")



def run_pymoo_algorithm_with_checks(problem: pymoo.core.problem.Problem,
                                    algorithm: Algorithm,
                                    ps_budget: int,
                                    verbose: bool,
                                    reattempts_when_fail: int):
    def run_and_get_results() -> list[EvaluatedPS]:
        res = minimize(problem,
                       algorithm,
                       termination=('n_evals', ps_budget),
                       verbose=verbose)

        if (res.X is None) or (res.F is None) or (res.G is None):
            print("Result had some Nones")
            return []

        if len(res.X.shape) == 1:
            result_pss = [EvaluatedPS(problem.individual_to_ps(res.X).values,
                                      metric_scores=res.F)]  # if there is only one result, the array has a different shape...
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
        print("The pss found in the search are ")
        for ps in pss:
            print("\t", ps)

    return pss
