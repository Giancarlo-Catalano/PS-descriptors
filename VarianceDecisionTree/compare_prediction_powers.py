import itertools
import json
import os
import random
from typing import Optional, Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from BenchmarkProblems.GraphColouring import GraphColouring
from BenchmarkProblems.RoyalRoad import RoyalRoad
from BenchmarkProblems.SATProblem import SATProblem
from BenchmarkProblems.TSP import TSP
from BenchmarkProblems.Trapk import Trapk
from Core.EvaluatedFS import EvaluatedFS
from Core.PRef import PRef
from Explanation.PRefManager import PRefManager
from RepresentationBasedSearch.ProblemRepresentation import ProblemRepresentation
from RepresentationBasedSearch.TSPPredicates import TSPVicinityRepresentation
from VarianceDecisionTree.AbstractDecisionTreeRegressor import AbstractDecisionTreeRegressor
from VarianceDecisionTree.IAIDecisionTree import IAIDecisionTree
from VarianceDecisionTree.PSDecisionTree import PSDecisionTree, PSDecisionTreeRestrictedDepth
from VarianceDecisionTree.naive_decision_tree import NaiveRegressorWrapper
from utils import simple_scatterplot, plot_ground_truth_vs_predictions


def compare_decision_trees(problem: BenchmarkProblem,
                           test_pRef: PRef,
                           train_pRef: PRef,
                           maximum_depth: int,
                           ps_budget: int,
                           ps_population: int) -> (dict, Any, PSDecisionTree):
    print(f"Comparing for {problem = }")
    naive_dts = [NaiveRegressorWrapper(maximum_depth=depth) for depth in range(1, maximum_depth + 1)]
    iai_dts = [IAIDecisionTree(maximum_depth=depth) for depth in range(1, maximum_depth + 1)]
    ps_dt = PSDecisionTree(maximum_depth=maximum_depth, ps_budget=ps_budget, ps_search_population_size=ps_population)
    ps_dt_views = [PSDecisionTreeRestrictedDepth(ps_dt, depth) for depth in range(1, maximum_depth + 1)]

    for tree in iai_dts:  # [ps_dt]+naive_dts+
        tree.train_from_pRef(train_pRef)

    # print(naive_dt)
    # print(ps_dt)

    # print("Testing on the solutions")
    # for solution, naive_guess, ps_guess in zip(test_solutions, naive_predictions, ps_predictions):
    #     actual_fitness = solution.fitness
    #     print("\t".join(map(repr, [actual_fitness, naive_guess, ps_guess])))

    # code for making the plot
    test_solutions: list[EvaluatedFS] = test_pRef.get_evaluated_FSs()
    ground_truths = np.array([solution.fitness for solution in test_solutions])
    # naive_predictions = np.array([naive_dts[-1].get_prediction(solution) for solution in test_solutions])
    ps_predictions = np.array([ps_dt.get_prediction(solution) for solution in test_solutions])
    plt = plot_ground_truth_vs_predictions(x_axis_label="ground_truth",
                                           x_axis_values=ground_truths,
                                           y_axis_label="predictions",
                                           y_axis_values=ps_predictions,
                                           title=f"actual vs ps_pred, {maximum_depth = }, problem = {problem}")
    # plt.show()

    result_dict = {"mse_naive": {tree.maximum_depth: tree.get_mse_on_test_data(test_pRef)
                                 for tree in naive_dts},
                   "mse_ps": {tree.maximum_depth: tree.get_mse_on_test_data(test_pRef)
                              for tree in ps_dt_views},
                   "mse_iai": {tree.maximum_depth: tree.get_mse_on_test_data(test_pRef)
                               for tree in iai_dts},
                   "problem": repr(problem),
                   "depth": maximum_depth,
                   "budget": ps_budget,
                   "ps_pop_size": ps_population}

    return result_dict, plt, ps_dt


def compare_for_multiple_problems():
    # tsp = TSP(
    #     cities=[(1, 5), (2, 5), (2, 4), (5, 2), (6, 2), (6, 3), (7, 2), (6, 7), (6, 8), (7, 8)],
    #     starting_ending_city=(5, 5))
    #
    # trap5 = Trapk(4, 5)
    #
    # rr = RoyalRoad(5)
    #
    # graph_colouring = GraphColouring.random(amount_of_colours=3, amount_of_nodes=7, chance_of_connection=0.3)

    sat_directory = r"/Users/gian/PycharmProjects/PS-descriptors/resources/problem_definitions/SAT/"
    small_SAT = SATProblem.from_cnf_file(os.path.join(sat_directory, "uf20-01.cnf"))
    medium_SAT = SATProblem.from_cnf_file(os.path.join(sat_directory, "uf50-01.cnf"))
    large_SAT = SATProblem.from_cnf_file(os.path.join(sat_directory, "uf100-01.cnf"))

    gc_directory = r"/Users/gian/PycharmProjects/PS-descriptors/resources/problem_definitions/GC"
    small_GC = GraphColouring.from_json(os.path.join(gc_directory, "anna.json"))
    big_GC = GraphColouring.from_json(os.path.join(gc_directory, "jean.json"))

    bt_problem = EfficientBTProblem.from_default_files()

    rr = RoyalRoad(5)

    problems = [rr]  # [bt_problem, big_GC, small_GC, large_SAT, medium_SAT, small_SAT]
    problem_names = ["RR"]  # ["BT", "GC_L", "GC_S", "SAT_L", "SAT_M", "SAT_S"]

    pRef_size = 10000
    test_size = 0.2
    ps_budget = 50
    ps_population = 100

    depth = 4
    # destination_folder = r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\resources\variance_tree_materials\results"+utils.get_formatted_timestamp()
    destination_folder = r"/Users/gian/PycharmProjects/PS-descriptors/resources/variance_tree_materials/results" + utils.get_formatted_timestamp()
    utils.make_directory(destination_folder)
    all_results = []

    for problem, problem_name in zip(problems, problem_names):
        for pRef_method in ["GA", "uniform"]:
            pRef = PRefManager.generate_pRef(problem=problem,
                                             sample_size=pRef_size,
                                             which_algorithm=pRef_method,
                                             verbose=False)
            pRef = PRef.unique(pRef)

            train_pRef, test_pRef = pRef.train_test_split(test_size=test_size, random_state=random.randint(0, 500))

            # try:
            result_dict, result_plt, _ = compare_decision_trees(problem, test_pRef, train_pRef, depth, ps_budget,
                                                                ps_population)

            result_dict["problem"] = problem_name  # just because it's so ugly otherwise
            timestamp = utils.get_formatted_timestamp()
            plt_file_name = f"{problem_name}_{pRef_method}_{depth}_{timestamp}.png"
            print(f"Will save to {plt_file_name}")
            result_dict["pRef_method"] = pRef_method
            all_results.append(result_dict)
            result_plt.savefig(os.path.join(destination_folder, plt_file_name))
            result_plt.close()
            # except Exception as e:
            #     print(f"There was an error for {pRef_method =}, {depth = }, {problem =}")
            #     if hasattr(e, "message"):
            #         print(f"The error was {e.message}")
            #         result_dict = {"problem":problem, "method": pRef_method, "error": e.message}
            #         all_results.append(result_dict)

    results_json_file = os.path.join(destination_folder, "results.json")
    with utils.open_and_make_directories(results_json_file) as file:
        json.dump(all_results, file, indent=4)


#compare_for_multiple_problems()


def test_benchmark_problem():
    problem = RoyalRoad(5)  #TSP.get_berlin52_instance()
    pRef_size = 10000
    pRef_methods = "GA"

    depth = 4
    ps_budget = 50
    ps_population = 100

    pRef = PRefManager.generate_pRef(problem=problem,
                                     sample_size=pRef_size,
                                     which_algorithm="uniform GA",
                                     verbose=True)
    pRef = PRef.unique(pRef)

    if isinstance(problem, TSP):
        representation = TSPVicinityRepresentation(problem, vicinity_threshold=5)
        pRef = representation.make_representation_pRef(pRef)

    test_size = 0.2
    test_pRef, train_pRef = pRef.train_test_split(test_size=test_size, random_state=random.randint(0, 500))
    result_dict, result_plt, ps_dt = compare_decision_trees(problem, test_pRef, train_pRef, depth, ps_budget,
                                                            ps_population)

    if isinstance(problem, TSP):
        ps_dt.set_repr_ps(representation.repr_representation)

    print(ps_dt)
    result_plt.show()

    print(json.dumps(result_dict, indent=4))


#compare_for_multiple_problems()


def get_problems_with_names():
    sat_directory = r"/Users/gian/PycharmProjects/PS-descriptors/resources/problem_definitions/SAT/"
    small_SAT = SATProblem.from_cnf_file(os.path.join(sat_directory, "uf20-01.cnf"))
    medium_SAT = SATProblem.from_cnf_file(os.path.join(sat_directory, "uf50-01.cnf"))
    large_SAT = SATProblem.from_cnf_file(os.path.join(sat_directory, "uf100-01.cnf"))

    gc_directory = r"/Users/gian/PycharmProjects/PS-descriptors/resources/problem_definitions/GC"
    small_GC = GraphColouring.from_json(os.path.join(gc_directory, "anna.json"))
    big_GC = GraphColouring.from_json(os.path.join(gc_directory, "jean.json"))

    bt_problem = EfficientBTProblem.from_default_files()

    return {"SAT_S": small_SAT,
            "SAT_M": medium_SAT,
            "SAT_L": large_SAT,
            "GC_S": small_GC,
            "GC_L": big_GC,
            "BT": bt_problem}


def get_error_datapoint(problem_name: str,
                        own_method_settings: dict,
                        sample_size: int,
                        pRef_method: str,
                        max_depth: int,
                        exception: Exception
                        ) -> dict:
    error_message = str(exception)  # exception.message if hasattr(exception, "message") else "no_error_message"
    return {"problem_name": problem_name,
            "own_method_settings": own_method_settings,
            "sample_size": sample_size,
            "pRef_method": pRef_method,
            "max_depth": max_depth,
            "error": error_message}


def get_datapoint_for_instance(problem_name: str,
                               problem: BenchmarkProblem,
                               own_method_settings: dict,
                               sample_size: int,
                               pRef_method: str,
                               max_depth: int,
                               crash_on_error: bool = False,
                               ) -> dict:
    def generate_datapoint():
        pRef = PRefManager.generate_pRef(problem, sample_size, pRef_method)
        pRef = PRef.unique(pRef)

        train_pRef, test_pRef = pRef.train_test_split(0.2, 42)

        tested_depths = list(range(2, max_depth + 1))
        iai_dts = [IAIDecisionTree(depth) for depth in tested_depths]
        traditional_dts = [NaiveRegressorWrapper(depth) for depth in tested_depths]
        own_dt = PSDecisionTree(max_depth, ps_budget=own_method_settings["ps_budget"],
                                ps_search_population_size=own_method_settings["ps_population"])
        own_dt_views = [PSDecisionTreeRestrictedDepth(own_dt, depth) for depth in tested_depths]

        for tree in [own_dt] + iai_dts + traditional_dts:
            tree.train_from_pRef(train_pRef)

        def get_mses_at_different_depths(trees: Iterable[AbstractDecisionTreeRegressor]):
            return {tree.maximum_depth: tree.get_mse_on_test_data(test_pRef)
                    for tree in trees}

        return {"problem_name": problem_name,
                "own_method_settings": own_method_settings,
                "sample_size": sample_size,
                "pRef_method": pRef_method,
                "iai": get_mses_at_different_depths(iai_dts),
                "naive": get_mses_at_different_depths(traditional_dts),
                "ps": get_mses_at_different_depths(own_dt_views)}

    if crash_on_error:
        return generate_datapoint()
    else:
        try:
            return generate_datapoint()
        except Exception as e:
            return get_error_datapoint(problem_name=problem_name,
                                       own_method_settings=own_method_settings,
                                       sample_size=sample_size,
                                       pRef_method=pRef_method,
                                       max_depth=max_depth,
                                       exception=e)


def sanity_check():
    problems = get_problems_with_names()
    problems = dict(list(problems.items())[:2])  # TODO restore this
    pRef_methods = ["GA", "uniform"]
    sample_size = 1000
    own_method_settings = {"ps_budget": 20,
                           "ps_population": 50}

    results = []

    for problem_name, problem in problems.items():
        for pRef_method in pRef_methods:
            datapoint = get_datapoint_for_instance(problem_name=problem_name,
                                                   problem=problem,
                                                   own_method_settings=own_method_settings,
                                                   sample_size=sample_size,
                                                   pRef_method=pRef_method,
                                                   max_depth=3,
                                                   crash_on_error=False)
            results.append(datapoint)

    print(json.dumps(results, indent=4))


sanity_check()
