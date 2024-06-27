from typing import Literal

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from FSStochasticSearch.HistoryPRefs import uniformly_random_distribution_pRef, pRef_from_GA, pRef_from_SA, \
    pRef_from_GA_best, pRef_from_SA_best
from utils import announce


def get_history_pRef(benchmark_problem: BenchmarkProblem,
                     sample_size: int,
                     which_algorithm: Literal["uniform", "GA", "SA", "GA_best", "SA_best"],
                     verbose=True):
    """
    This function basically dispatches the algorithm to generate the reference population
    @param benchmark_problem: provides the search space to get the reference population for
    @param sample_size: the amount of individuals in the reference population
    @param which_algorithm: the algorithm to be used to generate the reference population
    @param verbose: whether it prints something to the terminal or not
    @return: the reference population
    """
    with announce(f"Running the algorithm to generate the PRef using {which_algorithm}", verbose=verbose):
        match which_algorithm:
            case "uniform": return uniformly_random_distribution_pRef(sample_size=sample_size,
                                                                      benchmark_problem=benchmark_problem)
            case "GA": return pRef_from_GA(benchmark_problem=benchmark_problem,
                                           sample_size=sample_size,
                                           ga_population_size=300)
            case "SA": return pRef_from_SA(benchmark_problem=benchmark_problem,
                                           sample_size=sample_size,
                                           max_trace = sample_size)
            case "GA_best": return pRef_from_GA_best(benchmark_problem=benchmark_problem,
                                                     sample_size=sample_size,
                                                     fs_evaluation_budget=sample_size * 100)
            case "SA_best": return pRef_from_SA_best(benchmark_problem=benchmark_problem,
                                                     sample_size=sample_size)
            case _: raise ValueError

