from BenchmarkProblems.RoyalRoad import RoyalRoad
from Explanation.PRefManager import PRefManager
from VarianceDecisionTree.SimplePSSearchTask import find_ps_in_solution
from VarianceDecisionTree.recursive_pRef_splitting import recursively_split_pRef


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
    recursively_split_pRef(pRef, problem, [])


test_recursive_splitting()
