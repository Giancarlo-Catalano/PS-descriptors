import itertools
import json
import os
import random
from typing import Optional, Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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
        cps_for_iai = [0.2] # [0.25, 0.5, 0.75]
        pRef = PRefManager.generate_pRef(problem, sample_size, pRef_method)
        pRef = PRef.unique(pRef)

        train_pRef, test_pRef = pRef.train_test_split(0.2, 42)

        tested_depths = list(range(2, max_depth + 1))
        iai_dts: dict[float, list[IAIDecisionTree]] = {cp: [IAIDecisionTree(depth, cp)
                   for depth in tested_depths]
                   for cp in cps_for_iai}

        traditional_dts = [NaiveRegressorWrapper(depth) for depth in tested_depths]
        own_dt = PSDecisionTree(max_depth, ps_budget=own_method_settings["ps_budget"],
                                ps_search_population_size=own_method_settings["ps_population"])
        own_dt_views = [PSDecisionTreeRestrictedDepth(own_dt, depth) for depth in tested_depths]

        for tree in itertools.chain([own_dt], traditional_dts, *(iai_dts.values())):
            tree.train_from_pRef(train_pRef)

        def get_mses_at_different_depths(trees: Iterable[AbstractDecisionTreeRegressor]):
            return {tree.maximum_depth: tree.get_mse_on_test_data(test_pRef)
                    for tree in trees}

        return {"problem_name": problem_name,
                "own_method_settings": own_method_settings,
                "sample_size": sample_size,
                "pRef_method": pRef_method,
                "iai": {cp: get_mses_at_different_depths(iai_dts[cp])
                        for cp in cps_for_iai},
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


def many_dt_gather_data():
    problems = get_problems_with_names()
    problems = dict(list(problems.items())[:1])
    pRef_methods = ["GA", "uniform"]
    sample_size = 1000
    own_method_settings = {"ps_budget": 20,
                           "ps_population": 50}


    repeats = 3
    destination_folder = r"/Users/gian/PycharmProjects/PS-descriptors/resources/variance_tree_materials/dt_data" + utils.get_formatted_timestamp()
    utils.make_directory(destination_folder)
    print(f"Storing the results in {destination_folder}")

    def make_file_with_json_contents(json_dict):
        json_file_name = os.path.join(destination_folder, "output_"+utils.get_formatted_timestamp()+".json")
        with open(json_file_name, "w") as file:
            json.dump(json_dict, file, indent=4)


    def single_run():
        results = []

        for problem_name, problem in problems.items():
            for pRef_method in pRef_methods:
                datapoint = get_datapoint_for_instance(problem_name=problem_name,
                                                       problem=problem,
                                                       own_method_settings=own_method_settings,
                                                       sample_size=sample_size,
                                                       pRef_method=pRef_method,
                                                       max_depth=6,
                                                       crash_on_error=False)
                results.append(datapoint)

        make_file_with_json_contents(results)


    for iteration in tqdm(range(repeats)):
        single_run()

many_dt_gather_data()
