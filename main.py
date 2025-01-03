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
    pRef_methods = ["uniform", "GA", "SA", "Tabu"]
    sample_size = 10000

    depths = [2, 3, 4, 5]
    tree_dicts = []
    tree_dicts.extend([{"kind": "ps",
                        "ps_budget": ps_budget,
                        "ps_population": 100,
                        "depths": depths,
                        "avoid_ancestors": avoid_ancestors,
                        "metrics": metrics}
                       for ps_budget in [1000]
                      for metrics in ["variance", "variance estimated_atomicity", "variance consistency", "variance consistency estimated_atomicity", "variance simplicity", "variance simplicity estimated_atomicity"]
                      for avoid_ancestors in [False]])

    mode = "server"
    repeats = 10

    debug = True
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


gather_data_compare_own()
