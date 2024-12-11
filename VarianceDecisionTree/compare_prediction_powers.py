from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from BenchmarkProblems.RoyalRoad import RoyalRoad
from Core.EvaluatedFS import EvaluatedFS
from Core.PRef import PRef
from Explanation.PRefManager import PRefManager
from VarianceDecisionTree.PSDecisionTree import PSDecisionTree
from VarianceDecisionTree.naive_decision_tree import NaiveRegressorWrapper
from utils import simple_scatterplot


def compare_decision_trees():
    problem = EfficientBTProblem.from_default_files()
    pRef_size = 10000
    pRef_generation_algorithm = "GA"
    test_size = 0.2
    maximum_depth = 3

    pRef = PRefManager.generate_pRef(problem=problem,
                                     sample_size=pRef_size,
                                     which_algorithm=pRef_generation_algorithm,
                                     verbose=True)
    pRef = PRef.unique(pRef)

    test_pRef, train_pRef = pRef.train_test_split(test_size=test_size, random_state=42)

    naive_dt = NaiveRegressorWrapper(maximum_depth=maximum_depth)
    ps_dt = PSDecisionTree(maximum_depth=maximum_depth, ps_budget=3000, ps_search_population_size=100)

    for tree in [naive_dt, ps_dt]:
        tree.train_from_pRef(train_pRef)

    print("At the end of the training, the trees are")
    print(naive_dt)
    print(ps_dt)

    test_solutions: list[EvaluatedFS] = test_pRef.get_evaluated_FSs()
    print("Testing on the solutions")
    for solution in test_solutions:
        actual_fitness = solution.fitness
        naive_guess = naive_dt.get_prediction(solution)
        ps_guess = ps_dt.get_prediction(solution)
        print("\t".join(map(repr, [solution, actual_fitness, naive_guess, ps_guess])))

    print("The mse's are")
    naive_mse = naive_dt.get_mse_on_test_data(test_pRef)
    ps_mse = ps_dt.get_mse_on_test_data(test_pRef)
    print(f"{naive_mse = }, {ps_mse = }")


compare_decision_trees()
