#!/usr/bin/env python3
import json
import os
import random
import sys
import warnings

import utils
from VarianceDecisionTree.compare_prediction_powers import get_problems_with_names, get_datapoint_for_instance


warnings.simplefilter("always", UserWarning)
warnings.formatwarning = lambda message, category, filename, lineno, line = None: f"{message}\n"

def gather_data_compare_own():

    problems = get_problems_with_names()
    pRef_methods = ["uniform", "GA"] #, "SA", "Tabu"]
    sample_size = 10000

    depths = [2, 3, 4, 5, 6]
    tree_dicts = []
    tree_dicts.extend([{"kind": "ps",
                        "ps_budget": ps_budget,
                        "ps_population": 100,
                        "depths": depths,
                        "avoid_ancestors": avoid_ancestors,
                        "metrics": metrics}
                       for ps_budget in [5000]
                      for metrics in ["variance", "variance estimated_atomicity"]
                      for avoid_ancestors in [False]])

    mode = "server"
    repeats = 10

    debug = False
    print_progress = True
    if debug:
        print("NOTE: using debug mode")
        problems = dict(list(problems.items())[:1])
        pRef_methods = ["GA"]
        # sample_size = 100
        # tree_dicts = tree_dicts[:1]

    def make_file_with_json_contents(json_dict):
        json_file_name = os.path.join(destination_folder, "output_" + utils.get_formatted_timestamp() + ".json")
        with open(json_file_name, "w") as file:
            json.dump(json_dict, file, indent=4)

    def single_run():
        if len(sys.argv) < 2:
            seed = random.randrange(10000)
        else:
            try:
                seed = int(sys.argv[1])
            except:
                raise Exception(f"The second argument needs to be missing, or a number! {sys.argv[1]} was provided")
        results = []

        for problem_name, problem in problems.items():
            for pRef_method in pRef_methods:
                if print_progress:
                    warnings.warn(f"{problem_name = }, {pRef_method = }")
                datapoint = get_datapoint_for_instance(problem_name=problem_name,
                                                       problem=problem,
                                                       tree_settings_list=tree_dicts,
                                                       sample_size=sample_size,
                                                       pRef_method=pRef_method,
                                                       crash_on_error=debug,
                                                       seed = seed)
                results.append(datapoint)

        if mode == "local":
            make_file_with_json_contents(results)
        else:
            print(json.dumps(results, indent=4))

    if mode == "local":


        destination_folder = r"/Users/gian/PycharmProjects/PS-descriptors/resources/variance_tree_materials/compare_own_data" + utils.get_formatted_timestamp()
        utils.make_directory(destination_folder)
        print(f"Storing the results in {destination_folder}")

        for iteration in range(repeats):
            single_run()

    else:
        # just print out the results to the console at the end
        single_run()


def gather_data_compare_with_naive():

    problems = get_problems_with_names()
    pRef_methods = ["uniform", "GA", "SA", "Tabu"]
    sample_sizes = [10000, 30000]

    depths = [2, 3, 4, 5, 6]
    tree_dicts = []
    tree_dicts.extend([{"kind": "naive",
                        "depths": depths}])
    tree_dicts.extend([{"kind": "ps",
                        "ps_budget": 5000,
                        "ps_population": 100,
                        "depths": depths,
                        "avoid_ancestors": False,
                        "metrics": metrics}
                      for metrics in ["variance", "variance estimated_atomicity"]])



    mode = "server"
    repeats = 10

    debug = False
    print_progress = True
    if debug:
        print("NOTE: using debug mode")
        problems = dict(list(problems.items())[:1])
        # pRef_methods = ["GA"]
        # sample_size = 100
        tree_dicts = tree_dicts[:1]

    def make_file_with_json_contents(json_dict):
        json_file_name = os.path.join(destination_folder, "output_" + utils.get_formatted_timestamp() + ".json")
        with open(json_file_name, "w") as file:
            json.dump(json_dict, file, indent=4)

    def single_run():
        if len(sys.argv) < 2:
            seed = random.randrange(10000)
        else:
            try:
                seed = int(sys.argv[1])
            except:
                raise Exception(f"The second argument needs to be missing, or a number! {sys.argv[1]} was provided")
        results = []
        warnings.warn(f"The seed is {seed}")

        for problem_name, problem in problems.items():
            for pRef_method in pRef_methods:
                for sample_size in sample_sizes:
                    if print_progress:
                        warnings.warn(f"{problem_name = }, {pRef_method = }")
                    datapoint = get_datapoint_for_instance(problem_name=problem_name,
                                                           problem=problem,
                                                           tree_settings_list=tree_dicts,
                                                           sample_size=sample_size,
                                                           pRef_method=pRef_method,
                                                           crash_on_error=debug,
                                                           seed = seed)
                    results.append(datapoint)

        if mode == "local":
            make_file_with_json_contents(results)
        else:
            print(json.dumps(results, indent=4))

    if mode == "local":


        destination_folder = r"/Users/gian/PycharmProjects/PS-descriptors/resources/variance_tree_materials/compare_own_data" + utils.get_formatted_timestamp()
        utils.make_directory(destination_folder)
        print(f"Storing the results in {destination_folder}")

        for iteration in range(repeats):
            single_run()

    else:
        # just print out the results to the console at the end
        single_run()



def gather_data_compare_iai():

    problems = get_problems_with_names()
    pRef_methods = ["uniform", "GA", "SA", "Tabu"]
    sample_sizes = [10000]

    depths = [2, 3, 4, 5, 6]
    tree_dicts = []
    tree_dicts.extend([{"kind": "iai",
                        "cp": cp,
                        "depths": depths}
                       for cp in [0.25]])


    mode = "local"

    debug = False
    print_progress = True
    if debug:
        print("NOTE: using debug mode")
        problems = dict(list(problems.items())[:1])
        # pRef_methods = ["GA"]
        # sample_size = 100
        tree_dicts = tree_dicts[:1]
        tree_dicts[0]["depths"] = [5]
        sample_sizes = [500]

    def make_file_with_json_contents(json_dict, seed):
        json_file_name = os.path.join(destination_folder, f"output_{seed}_" + utils.get_formatted_timestamp() + ".json")
        print(f"Storing the results at {json_file_name}")
        with open(json_file_name, "w") as file:
            json.dump(json_dict, file, indent=4)

    def single_run(seed: int):

        results = []
        warnings.warn(f"The seed is {seed}")

        for problem_name, problem in problems.items():
            for pRef_method in pRef_methods:
                for sample_size in sample_sizes:
                    if print_progress:
                        warnings.warn(f"{problem_name = }, {pRef_method = }, {sample_size =}")
                    datapoint = get_datapoint_for_instance(problem_name=problem_name,
                                                           problem=problem,
                                                           tree_settings_list=tree_dicts,
                                                           sample_size=sample_size,
                                                           pRef_method=pRef_method,
                                                           crash_on_error=debug,
                                                           seed = seed)
                    results.append(datapoint)

        if mode == "local":
            make_file_with_json_contents(results, seed)
        else:
            print(json.dumps(results, indent=4))

    if mode == "local":
        on_windows = utils.get_os() == "Windows"

        run_name = "iai_run_W_2"+utils.get_formatted_timestamp()
        if on_windows:
            destination_folder = r"C:\Users\gac8\Desktop\CondorResults\VDT\compareown"+"\\"+run_name
        else:
            destination_folder = r"/Users/gian/Desktop/CondorResults/VDT/compareown/" + run_name
        utils.make_directory(destination_folder)
        print(f"Storing the results in {destination_folder}")

        seeds = list(range(97, -1, -1),)
        for seed in seeds:
            single_run(seed)

    else:
        # just print out the results to the console at the end
        if len(sys.argv) < 2:
            seed = random.randrange(10000)
        else:
            try:
                seed = int(sys.argv[1])
            except:
                raise Exception(f"The second argument needs to be an integer, or a number! {sys.argv[1]} was provided")
        single_run(seed)


gather_data_compare_iai()
