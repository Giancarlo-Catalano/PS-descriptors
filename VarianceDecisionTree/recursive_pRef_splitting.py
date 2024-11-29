import numpy as np

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.PRef import PRef
from Core.PS import PS
from GuestLecture.show_off_problems import get_unexplained_parts
from VarianceDecisionTree.SimplePSSearchTask import find_ps_in_solution
from VarianceDecisionTree.VarianceMetric import SplitVariance


def split_pRef_using_ps(pRef: PRef, ps: PS) -> (PRef, PRef):

    matching, not_matching = SplitVariance.get_split_indexes_of_ps(pRef, ps)

    matching_pRef = PRef(fitness_array=pRef.fitness_array[matching],
                         full_solution_matrix=pRef.full_solution_matrix[matching],
                         search_space=pRef.search_space)

    not_matching_pRef = PRef(fitness_array=pRef.fitness_array[not_matching],
                         full_solution_matrix=pRef.full_solution_matrix[not_matching],
                         search_space=pRef.search_space)
    return matching_pRef, not_matching_pRef


def split_pRef(pRef: PRef, problem: BenchmarkProblem, accumulated_patterns: list[PS]) -> (PS, PRef, PRef):
    best_solution = pRef.get_best_solution()
    unexplained_vars = get_unexplained_parts(best_solution, accumulated_patterns)
    print(f"Splitting the pRef where the best solution is {problem.repr_fs(best_solution)}, "
          f"with fitness {best_solution.fitness}")
    print(f"The unexplained mask is {''.join('U' if v else '-' for v in unexplained_vars)}")

    pss = find_ps_in_solution(pRef=pRef,
                              problem=problem,
                              ps_budget=1000,
                              culling_method="biggest",
                              population_size=50,
                              to_explain=best_solution,
                              unexplained_mask=unexplained_vars,
                              proportion_unexplained_that_needs_used=0.01,
                              proportion_used_that_should_be_unexplained=0.5,
                              verbose=False)

    print(f"The winning ps is ")
    split_ps = pss[0]
    print(problem.repr_ps(split_ps))
    matches, unmatches = split_pRef_using_ps(pRef, split_ps)
    return split_ps, matches, unmatches



def recursively_split_pRef(starting_pRef: PRef, problem: BenchmarkProblem, accumulated_winners: list[PS] = None):
    accumulated_patterns = [] if accumulated_winners is None else list(accumulated_winners)
    def should_split_pRef(pRef: PRef) -> bool:
        return pRef.sample_size >= 100

    if should_split_pRef(starting_pRef):
        ps, matches, unmatches = split_pRef(starting_pRef, problem, accumulated_patterns)
        recursively_split_pRef(matches, problem, accumulated_patterns+[ps])
        recursively_split_pRef(unmatches, problem)
    else:
        if starting_pRef.sample_size < 1:
            print("Actually, this PRef is Empty!")
        else:
            best_solution = starting_pRef.get_best_solution()
            print(f"Could not split the pRef were the best solution is {problem.repr_fs(best_solution)}, "
                  f"with fitness {best_solution.fitness}")
