from typing import Callable

import utils
from BenchmarkProblems.GraphColouring import GraphColouring
from BenchmarkProblems.RoyalRoad import RoyalRoad
from BenchmarkProblems.TSP import TSP
from Core.PRef import PRef
from Explanation.PRefManager import PRefManager
from VarianceDecisionTree.SimplePSSearchTask import find_ps_in_solution
from VarianceDecisionTree.recursive_pRef_splitting import recursively_split_pRef

from RepresentationBasedSearch.ProblemRepresentation import TrivialRepresentation, CombinedProblemRepresentations
from RepresentationBasedSearch.TSPPredicates import TSPPrecedenceRepresentation, TSPVicinityRepresentation


def test_variance_tree():
    problem = RoyalRoad(5)
    print(f"The problem is {problem}")
    pRef = PRefManager.generate_pRef(problem=problem,
                                     sample_size=10000,
                                     which_algorithm="uniform GA",
                                     verbose=True)

    best_solution = pRef.get_best_solution()
    print(f"The best solution has fitness = {best_solution.fitness}")
    print(problem.repr_fs(best_solution))

    pss = find_ps_in_solution(pRef=pRef,
                              problem=problem,
                              ps_budget=1000,
                              population_size=50,
                              to_explain=best_solution,
                              culling_method=None,
                              proportion_unexplained_that_needs_used=0,
                              verbose=True)

    print("The pss obtained are")
    for ps in pss:
        print(f"\t{problem.repr_ps(ps)}, fitness = {ps.metric_scores}")


def repr_tree(node: list,
              repr_ps: Callable):
    if len(node) == 0:
        return "leaf"

    head, left_branch, right_branch = node[0]
    head_repr = repr_ps(head)
    left_repr = "(L)"+repr_tree(left_branch, repr_ps)
    right_repr = "(R)"+repr_tree(right_branch, repr_ps)
    return (f"{head_repr}"
            f"\n{utils.indent(left_repr)}"
            f"\n{utils.indent(right_repr)}")


def test_recursive_splitting():
    # problem = RoyalRoad(5)
    problem = GraphColouring.make_insular_instance(6)
    print(f"The problem is {problem}")
    pRef = PRefManager.generate_pRef(problem=problem,
                                     sample_size=10000,
                                     which_algorithm="uniform GA",
                                     verbose=True)
    pRef = PRef.unique(pRef)

    tree = recursively_split_pRef(pRef, problem, repr_fs=problem.repr_fs, repr_ps=problem.repr_ps,
                           max_depth=3,
                           fitness_threshold=0)

    print("The tree is")
    print(repr_tree(tree, problem.repr_ps))

    print(tree)


def test_recursive_splitting_with_representation():
    problem = TSP(
        cities=[(1, 5), (2, 5), (2, 4), (5, 2), (6, 2), (6, 3), (7, 2), (6, 7), (6, 8), (7, 8)],
        starting_ending_city=(5, 5))

    # trivial_representation = TrivialRepresentation(problem)
    # precedence_representation = TSPPrecedenceRepresentation(problem)
    vicinity_representation = TSPVicinityRepresentation(problem, vicinity_threshold=3)

    representation = CombinedProblemRepresentations([vicinity_representation])
    print(f"The problem is {problem}")
    pRef = PRefManager.generate_pRef(problem=problem,
                                     sample_size=10000,
                                     which_algorithm="uniform GA",
                                     verbose=True)

    pRef = PRef.unique(pRef)

    print(f"The best solution is {problem.repr_fs(pRef.get_best_solution())}")

    extended_pRef = representation.make_representation_pRef(pRef)

    tree = recursively_split_pRef(extended_pRef, problem,
                                  repr_fs=representation.repr_representation,
                                  repr_ps=representation.repr_partial_representation,
                                  max_depth=4,
                                  fitness_threshold=-10000)

    print("The tree is")
    print(repr_tree(tree, problem.repr_ps))

    print(tree)


test_recursive_splitting()
#test_recursive_splitting_with_representation()
