from BenchmarkProblems.RoyalRoad import RoyalRoad
from BenchmarkProblems.TSP import TSP
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


def test_recursive_splitting():
    problem = RoyalRoad(5)
    print(f"The problem is {problem}")
    pRef = PRefManager.generate_pRef(problem=problem,
                                     sample_size=10000,
                                     which_algorithm="uniform GA",
                                     verbose=True)


    recursively_split_pRef(pRef, problem, [], repr_fs=problem.repr_fs, repr_ps=problem.repr_ps)


def test_recursive_splitting_with_representation():
    problem = TSP(
        cities=[(1, 5), (2, 5), (2, 4), (5, 2), (6, 2), (6, 3), (7, 2), (6, 7), (6, 8), (7, 8), (7, 9)],
        starting_ending_city=(5, 5))


    trivial_representation = TrivialRepresentation(problem)
    #precedence_representation = TSPPrecedenceRepresentation(problem)
    vicinity_representation = TSPVicinityRepresentation(problem, vicinity_threshold=3)

    representation = CombinedProblemRepresentations([vicinity_representation])
    print(f"The problem is {problem}")
    pRef = PRefManager.generate_pRef(problem=problem,
                                     sample_size=10000,
                                     which_algorithm="GA",
                                     verbose=True)


    print(f"The best solution is {problem.repr_fs(pRef.get_best_solution())}")


    extended_pRef = representation.make_representation_pRef(pRef)

    recursively_split_pRef(extended_pRef, problem, [],
                           repr_fs=representation.repr_representation, repr_ps = representation.repr_partial_representation)


#test_recursive_splitting()
test_recursive_splitting_with_representation()
