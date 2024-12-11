from typing import Callable

import numpy as np

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.PRef import PRef
from Core.PS import PS
from GuestLecture.show_off_problems import get_unexplained_parts
from VarianceDecisionTree.SimplePSSearchTask import find_ps_in_solution
from VarianceDecisionTree.SplitVariance import SplitVariance


def split_pRef_using_ps(pRef: PRef, ps: PS) -> (PRef, PRef):
    matching_indexes = pRef.get_indexes_matching_ps(ps)
    return pRef.split_by_indexes(matching_indexes)


def split_pRef(pRef: PRef, problem: BenchmarkProblem, accumulated_patterns: list[PS]) -> (PS, PRef, PRef):
    best_solution = pRef.get_best_solution()
    unexplained_vars = get_unexplained_parts(best_solution, accumulated_patterns)
    # print(f"Splitting the pRef where the best solution is {best_solution}, "
    #       f"with fitness {best_solution.fitness}, (size = {pRef.sample_size})")
    print(f"The unexplained mask is {''.join('U' if v else '-' for v in unexplained_vars)}")

    pss = find_ps_in_solution(pRef=pRef,
                              problem=problem,
                              ps_budget=5000,
                              culling_method="biggest",
                              population_size=100,
                              to_explain=best_solution,
                              unexplained_mask=unexplained_vars,
                              proportion_unexplained_that_needs_used=0.01,
                              proportion_used_that_should_be_unexplained=0.5,
                              verbose=True)

    split_ps = pss[0]
    matches, unmatches = split_pRef_using_ps(pRef, split_ps)
    return split_ps, matches, unmatches

def recursively_split_pRef(starting_pRef: PRef,
                           problem: BenchmarkProblem,
                           repr_ps: Callable,
                           repr_fs: Callable,
                           max_depth: int,
                           fitness_threshold: float
                           ):

    def recursive_step(pRef_to_split, current_depth, current_branch, ancestors):
        best_solution = pRef_to_split.get_best_solution()


        if pRef_to_split.sample_size > 20 and current_depth < max_depth and best_solution.fitness > fitness_threshold:
            ps, matches, unmatches = split_pRef(starting_pRef, problem, ancestors)
            matching_branch = []
            unmatching_branch = []
            new_branch_entry = (ps, matching_branch, unmatching_branch)
            current_branch.append(new_branch_entry)
            print(f"Splitting a pRef of size {starting_pRef.sample_size}, best_fitness = {best_solution.fitness} where the ancestors are")
            print("\n".join(f"\t{w}" for w in ancestors))
            print("The splitting ps is \n")
            print(repr_ps(ps))
            recursive_step(matches, current_depth+1,
                           matching_branch, ancestors + [ps])
            recursive_step(unmatches, current_depth+1,
                           unmatching_branch, ancestors)


    tree = []
    recursive_step(starting_pRef, ancestors=[], current_branch=tree, current_depth=0)
    return tree
