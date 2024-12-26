
import json
import os
import random

import utils
from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from BenchmarkProblems.GraphColouring import GraphColouring
from BenchmarkProblems.SATProblem import SATProblem
from BenchmarkProblems.TSP import TSP
from Core.PRef import PRef
from Explanation.PRefManager import PRefManager
from RepresentationBasedSearch.TSPPredicates import TSPVicinityRepresentation
from VarianceDecisionTree.compare_prediction_powers import compare_decision_trees


def generate_data_for_native_own():

    sat_directory = r"/Users/gian/PycharmProjects/PS-descriptors/resources/problem_definitions/SAT/"
    small_SAT = SATProblem.from_cnf_file(os.path.join(sat_directory, "uf20-01.cnf"))
    medium_SAT = SATProblem.from_cnf_file(os.path.join(sat_directory, "uf50-01.cnf"))
    large_SAT = SATProblem.from_cnf_file(os.path.join(sat_directory, "uf100-01.cnf"))


    gc_directory = r"/Users/gian/PycharmProjects/PS-descriptors/resources/problem_definitions/GC"
    small_GC = GraphColouring.from_json(os.path.join(gc_directory, "anna.json"))
    big_GC = GraphColouring.from_json(os.path.join(gc_directory, "jean.json"))

    bt_problem = EfficientBTProblem.from_default_files()

    problems = [bt_problem, big_GC, small_GC, large_SAT, medium_SAT, small_SAT]
    problem_names = ["BT", "GC_L", "GC_S", "SAT_L", "SAT_M", "SAT_S"]

    pRef_size = 10000
    test_size = 0.2
    ps_budget = 50
    ps_population = 100

    depth = 7
    all_results = []

    for problem, problem_name in zip(problems, problem_names):
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

            # try:
            result_dict, result_plt, _ = compare_decision_trees(problem, test_pRef, train_pRef, depth, ps_budget,
                                                             ps_population)

            result_dict["problem"] = problem_name  # just because it's so ugly otherwise
            timestamp = utils.get_formatted_timestamp()
            plt_file_name = f"{problem_name}_{pRef_method}_{depth}_{timestamp}.png"
            print(f"Will save to {plt_file_name}")
            result_dict["pRef_method"] = pRef_method
            all_results.append(result_dict)

    print(json.dumps(all_results, indent=4))