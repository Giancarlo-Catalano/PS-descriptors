import itertools
import json
import os
import random
import warnings
from typing import Iterable

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from BenchmarkProblems.GraphColouring import GraphColouring
from BenchmarkProblems.SATProblem import SATProblem
from Core.PRef import PRef
from Explanation.PRefManager import PRefManager
from VarianceDecisionTree.AbstractDecisionTreeRegressor import AbstractDecisionTreeRegressor

import platform

if platform.system() in {"DarwinNOT", "Windows"}:  # I KNOW THAT THIS IS DODGY, BUT THE LIBRARY WON'T WORK ON CONDOR
    from VarianceDecisionTree.IAIDecisionTree import IAIDecisionTree

from VarianceDecisionTree.PSDecisionTree import PSDecisionTree, PSDecisionTreeRestrictedDepth
from VarianceDecisionTree.naive_decision_tree import NaiveRegressorWrapper


#compare_for_multiple_problems()


def get_problems_with_names():
    resources_directory = utils.get_resources_directory()
    problem_definition_directory = os.path.join(resources_directory, "problem_definitions")
    sat_directory = os.path.join(problem_definition_directory, "SAT")
    small_SAT = SATProblem.from_cnf_file(os.path.join(sat_directory, "uf20-01.cnf"))
    medium_SAT = SATProblem.from_cnf_file(os.path.join(sat_directory, "uf50-01.cnf"))
    large_SAT = SATProblem.from_cnf_file(os.path.join(sat_directory, "uf100-01.cnf"))

    gc_directory = os.path.join(problem_definition_directory, "GC")
    small_GC = GraphColouring.from_json(os.path.join(gc_directory, "anna.json"))
    big_GC = GraphColouring.from_json(os.path.join(gc_directory, "jean.json"))

    bt_problem_root = os.path.join(resources_directory, "BT", "MartinsInstance")
    bt_problem = EfficientBTProblem.from_csv_files(employee_data_file=os.path.join(bt_problem_root, "employeeData.csv"),
                                                   employee_skills_file=os.path.join(bt_problem_root,
                                                                                     "employeeSkillsData.csv"),
                                                   rota_file=os.path.join(bt_problem_root, "rosterPatternDaysData.csv"),
                                                   calendar_length=7 * 13)

    return {"SAT_S": small_SAT,
            "SAT_M": medium_SAT,
            "SAT_L": large_SAT,
            "GC_S": small_GC,
            "GC_L": big_GC,
            "BT": bt_problem}


def get_error_datapoint(problem_name: str,
                        tree_settings_list: list[dict],
                        sample_size: int,
                        pRef_method: str,
                        exception: Exception
                        ) -> dict:
    error_message = str(exception)  # exception.message if hasattr(exception, "message") else "no_error_message"
    return {"problem_name": problem_name,
            "tree_settings_list": tree_settings_list,
            "sample_size": sample_size,
            "pRef_method": pRef_method,
            "error": error_message}


def get_trees_from_dict(tree_dict: dict,
                        train_pRef: PRef,
                        problem: BenchmarkProblem) -> list[AbstractDecisionTreeRegressor]:
    kind = tree_dict["kind"]
    depths = tree_dict["depths"]
    warnings.warn(f"Training {tree_dict}")

    if kind == "ps":
        max_depth = max(depths)
        tree = PSDecisionTree(max_depth,
                              ps_budget=tree_dict["ps_budget"],
                              ps_search_population_size=tree_dict["ps_population"],
                              avoid_ancestors=tree_dict["avoid_ancestors"],
                              metrics_to_use=tree_dict["metrics"],
                              problem=problem)
        tree.train_from_pRef(train_pRef)
        views = [PSDecisionTreeRestrictedDepth(tree, depth) for depth in depths]
        return views
    else:
        if kind == "naive":
            trees = [NaiveRegressorWrapper(depth) for depth in depths]
        elif kind == "iai":
            cp = tree_dict["cp"]
            trees = eval("[IAIDecisionTree(depth, cp) for depth in depths]")
        else:
            raise NotImplemented
        for tree in trees:
            print(f"Training a tree {tree}")
            tree.train_from_pRef(train_pRef)
        return trees


def get_datapoint_for_tree(tree: AbstractDecisionTreeRegressor, test_pRef):
    error_metrics = tree.get_error_metrics(test_pRef)
    if isinstance(tree, PSDecisionTreeRestrictedDepth):
        output = {"kind": "ps",
                  "depth": tree.depth,
                  "ps_budget": tree.original_dt.ps_budget,
                  "ps_population": tree.original_dt.ps_search_population_size,
                  "avoid_ancestors": tree.original_dt.avoid_ancestors,
                  "metrics": tree.original_dt.metrics_to_use}
        if tree.depth == tree.original_dt.maximum_depth:
            output["order_tree"] = tree.get_orders()

    elif isinstance(tree, NaiveRegressorWrapper):
        output = {"kind": "naive",
                "depth": tree.maximum_depth}
    else:  # isinstance(tree, IAIDecisionTree): # we can't call the class directly because that would require for the IAI libraries to be imported even when we're using the Condor cluserte
        output = {"kind": "iai",
                "depth": tree.maximum_depth,
                "cp": tree.prescription_factor}

    output["results"] = error_metrics
    warnings.warn(json.dumps(output))
    return output


def get_datapoint_for_instance(problem_name: str,
                               problem: BenchmarkProblem,
                               tree_settings_list: list[dict],
                               sample_size: int,
                               pRef_method: str,
                               seed: int,
                               crash_on_error: bool = False,
                               ) -> dict:
    def generate_datapoint():
        random.seed(seed)
        pRef = PRefManager.generate_pRef(problem, sample_size, pRef_method)
        pRef = PRef.unique(pRef)

        train_pRef, test_pRef = pRef.train_test_split(0.2, 42)

        return {"problem_name": problem_name,
                "sample_size": sample_size,
                "pRef_method": pRef_method,
                "results_by_tree": [get_datapoint_for_tree(tree, test_pRef)
                                    for tree_dict in tree_settings_list
                                    for tree in get_trees_from_dict(tree_dict, train_pRef, problem)
                                    ]}

    if crash_on_error:
        return generate_datapoint()
    else:
        try:
            return generate_datapoint()
        except Exception as e:
            return get_error_datapoint(problem_name=problem_name,
                                       tree_settings_list=tree_settings_list,
                                       sample_size=sample_size,
                                       pRef_method=pRef_method,
                                       exception=e)


def gather_data_compare_dts():
    problems = get_problems_with_names()
    problems = dict(list(problems.items())[:1])
    pRef_methods = ["GA", "uniform"]
    sample_size = 1000
    own_method_settings = {"ps_budget": 20,
                           "ps_population": 50}

    depths = [2, 3, 4]
    tree_dicts = []
    tree_dicts.extend([{"kind": "ps",
                        "ps_budget": 20,
                        "ps_population": 50,
                        "depths": depths}])

    tree_dicts.extend([{"kind": "iai",
                        "cp": cp,
                        "depths": depths}
                       for cp in [0.25, 0.5]])

    tree_dicts.extend([{"kind": "naive",
                        "depths": depths}])

    repeats = 5

    destination_folder = r"/Users/gian/PycharmProjects/PS-descriptors/resources/variance_tree_materials/dt_data" + utils.get_formatted_timestamp()
    utils.make_directory(destination_folder)
    print(f"Storing the results in {destination_folder}")

    def make_file_with_json_contents(json_dict):
        json_file_name = os.path.join(destination_folder, "output_" + utils.get_formatted_timestamp() + ".json")
        with open(json_file_name, "w") as file:
            json.dump(json_dict, file, indent=4)

    def single_run():
        results = []

        for problem_name, problem in problems.items():
            for pRef_method in pRef_methods:
                datapoint = get_datapoint_for_instance(problem_name=problem_name,
                                                       problem=problem,
                                                       tree_settings_list=tree_dicts,
                                                       sample_size=sample_size,
                                                       pRef_method=pRef_method,
                                                       crash_on_error=False)
                results.append(datapoint)

        make_file_with_json_contents(results)

    for iteration in range(repeats):
        single_run()


def gather_data_compare_own():
    problems = get_problems_with_names()
    problems = dict(list(problems.items()))
    pRef_methods = ["GA", "uniform"]
    sample_size = 10000

    depths = [2, 3, 4, 5, 6]
    tree_dicts = []
    tree_dicts.extend([{"kind": "ps",
                        "ps_budget": ps_budget,
                        "ps_population": 100,
                        "depths": depths}
                       for ps_budget in [1000, 2000, 5000]])

    mode = "local"

    def make_file_with_json_contents(json_dict):
        json_file_name = os.path.join(destination_folder, "output_" + utils.get_formatted_timestamp() + ".json")
        with open(json_file_name, "w") as file:
            json.dump(json_dict, file, indent=4)

    def single_run():
        results = []

        for problem_name, problem in problems.items():
            for pRef_method in pRef_methods:
                datapoint = get_datapoint_for_instance(problem_name=problem_name,
                                                       problem=problem,
                                                       tree_settings_list=tree_dicts,
                                                       sample_size=sample_size,
                                                       pRef_method=pRef_method,
                                                       crash_on_error=False)
                results.append(datapoint)

        if mode == "local":
            make_file_with_json_contents(results)
        else:
            print(json.dumps(results, indent=4))

    if mode == "local":

        repeats = 10
        destination_folder = r"/Users/gian/PycharmProjects/PS-descriptors/resources/variance_tree_materials/compare_own_data" + utils.get_formatted_timestamp()
        utils.make_directory(destination_folder)
        print(f"Storing the results in {destination_folder}")

        for iteration in range(repeats):
            single_run()

# gather_data_compare_own()
