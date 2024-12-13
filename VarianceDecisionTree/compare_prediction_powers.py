import json
import os
import random
from typing import Optional, Any

import matplotlib.pyplot as plt
import numpy as np

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from BenchmarkProblems.GraphColouring import GraphColouring
from BenchmarkProblems.RoyalRoad import RoyalRoad
from BenchmarkProblems.TSP import TSP
from BenchmarkProblems.Trapk import Trapk
from Core.EvaluatedFS import EvaluatedFS
from Core.PRef import PRef
from Explanation.PRefManager import PRefManager
from RepresentationBasedSearch.ProblemRepresentation import ProblemRepresentation
from RepresentationBasedSearch.TSPPredicates import TSPVicinityRepresentation
from VarianceDecisionTree.PSDecisionTree import PSDecisionTree
from VarianceDecisionTree.naive_decision_tree import NaiveRegressorWrapper
from utils import simple_scatterplot, plot_ground_truth_vs_predictions


def compare_decision_trees(problem: BenchmarkProblem,
                           test_pRef: PRef,
                           train_pRef: PRef,
                           maximum_depth: int,
                           ps_budget: int,
                           ps_population: int) -> (dict, Any):
    naive_dt = NaiveRegressorWrapper(maximum_depth=maximum_depth)
    ps_dt = PSDecisionTree(maximum_depth=maximum_depth, ps_budget=ps_budget, ps_search_population_size=ps_population)

    for tree in [naive_dt, ps_dt]:
        tree.train_from_pRef(train_pRef)

    print("At the end of the training, the trees are")
    #print(naive_dt)
    #print(ps_dt)

    test_solutions: list[EvaluatedFS] = test_pRef.get_evaluated_FSs()
    ground_truths = np.array([solution.fitness for solution in test_solutions])
    naive_predictions = np.array([naive_dt.get_prediction(solution) for solution in test_solutions])
    ps_predictions = np.array([ps_dt.get_prediction(solution) for solution in test_solutions])
    # print("Testing on the solutions")
    # for solution, naive_guess, ps_guess in zip(test_solutions, naive_predictions, ps_predictions):
    #     actual_fitness = solution.fitness
    #     print("\t".join(map(repr, [actual_fitness, naive_guess, ps_guess])))

    print("The mse's are")
    naive_mse = naive_dt.get_mse_on_test_data(test_pRef)
    ps_mse = ps_dt.get_mse_on_test_data(test_pRef)
    print(f"{naive_mse = }, {ps_mse = }")

    plt = plot_ground_truth_vs_predictions(x_axis_label="ground_truth",
                                           x_axis_values=ground_truths,
                                           y_axis_label="predictions",
                                           y_axis_values=ps_predictions,
                                           title=f"actual vs ps_pred, {maximum_depth = }, problem = {problem}")
    # plt.show()

    result_dict = {"mse_naive": naive_mse,
                   "mse_ps": ps_mse,
                   "problem": repr(problem),
                   "depth": maximum_depth,
                   "budget": ps_budget,
                   "ps_pop_size": ps_population}

    return result_dict, plt


def compare_for_multiple_problems():
    tsp = TSP(
        cities=[(1, 5), (2, 5), (2, 4), (5, 2), (6, 2), (6, 3), (7, 2), (6, 7), (6, 8), (7, 8)],
        starting_ending_city=(5, 5))

    trap5 = Trapk(4, 5)

    rr = RoyalRoad(5)

    graph_colouring = GraphColouring.random(amount_of_colours=3, amount_of_nodes=7, chance_of_connection=0.3)

    bt_problem = EfficientBTProblem.from_default_files()

    problems = [tsp, trap5, rr, graph_colouring, bt_problem]

    pRef_size = 10000
    test_size = 0.2
    ps_budget = 5000
    ps_population = 100

    destination_folder = r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\resources\variance_tree_materials\results"

    all_results = []

    for problem in problems:
        for pRef_method in ["GA", "uniform"]:
            pRef = PRefManager.generate_pRef(problem=problem,
                                             sample_size=pRef_size,
                                             which_algorithm=pRef_method,
                                             verbose=False)
            pRef = PRef.unique(pRef)

            if isinstance(problem, TSP):
                representation = TSPVicinityRepresentation(problem, vicinity_threshold=3)
                pRef = representation.make_representation_pRef(pRef)

            test_pRef, train_pRef = pRef.train_test_split(test_size=test_size, random_state=random.randint(0, 500))

            for depth in [3, 6, 9]:
                try:
                    result_dict, result_plt = compare_decision_trees(problem, test_pRef, train_pRef, depth, ps_budget, ps_population)
                    timestamp = utils.get_formatted_timestamp()
                    plt_file_name = f"{problem}_{pRef_method}_{depth}_{timestamp}.png"
                    print(f"Will save to {plt_file_name}")
                    result_dict["pRef_method"] = pRef_method
                    all_results.append(result_dict)
                    plt.savefig(os.path.join(destination_folder, plt_file_name))
                    plt.close()
                except:
                    print(f"There was an error for {pRef_method =}, {depth = }, {problem =}")


    results_json_file = os.path.join(destination_folder, "results.json")
    with utils.open_and_make_directories(results_json_file) as file:
        json.dump(all_results, file)


compare_for_multiple_problems()



