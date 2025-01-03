import itertools
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

import utils



def json_to_entries(data: dict):
    def item_to_list_of_entries(item) -> list[dict]:
        problem_name = item["problem_name"]
        pRef_method = item["pRef_method"]

        entries = item["results_by_tree"]

        def get_modified_entry(entry):
            entry["problem"] = problem_name
            entry["pRef_method"] = pRef_method

            errors = entry["results"]
            entry = entry | errors
            del entry["results"]

            if "order_tree" in entry:
                del entry["order_tree"]

            return entry

        entries = list(map(get_modified_entry, entries))
        return entries

    return [entry for item in data for entry in item_to_list_of_entries(item)]

def convert_accuracy_data_to_df():
    input_directory = r"/Users/gian/Desktop/CondorResults/VDT/compareown/run2/"
    output_filename = r"/Users/gian/Desktop/CondorResults/VDT/compareown/run2/results.csv"

    all_dicts = []
    # Iterate through all files in the input directory
    for filename in os.listdir(input_directory):
        # Construct full file path
        file_path = os.path.join(input_directory, filename)

        # Check if the file is a JSON file
        if not os.path.isfile(file_path):
            continue

        if file_path.endswith(".csv"):
            continue

        with open(file_path, 'r') as file:
            data = json.load(file)
            entries = json_to_entries(data)
            all_dicts.extend(entries)

    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(all_dicts)

    # Write the DataFrame to a CSV file
    df.to_csv(output_filename, index=False)



def json_to_tree_data(data: dict):
    def item_to_list_of_entries(item) -> list[dict]:
        surrounding_information = {prop: item[prop]
                                   for prop in ["problem_name", "pRef_method"]}

        entries = item["results_by_tree"]
        entries = [thing for thing in entries if "order_tree" in thing]

        def convert_order_tree(order_tree, accumulator = None, current_depth: int = 0):
            if accumulator is None:
                accumulator = defaultdict(list)
            accumulator[current_depth].append(order_tree["own"])
            if len(order_tree["maching"]) > 0:
                convert_order_tree(order_tree["maching"], accumulator, current_depth+1)

            if len(order_tree["unmatching"]) > 0:
                convert_order_tree(order_tree["unmatching"], accumulator, current_depth+1)

            return accumulator
        def convert_tree_to_averages_by_level(entry):
            ps_search_info = {prop: entry[prop]
                                   for prop in ["ps_budget", "ps_population"]}
            tree_structure = entry["order_tree"]
            just_depths = convert_order_tree(tree_structure)
            average_depths = {f"average_at_{depth}": np.average(orders)
                              for depth, orders in just_depths.items()}
            standard_deviations = {f"sd_at_{depth}": np.std(orders)
                              for depth, orders in just_depths.items()}
            return surrounding_information | ps_search_info | average_depths | standard_deviations


        entries = list(map(convert_tree_to_averages_by_level, entries))
        return entries

    return [entry for item in data for entry in item_to_list_of_entries(item)]

def convert_tree_data_to_df():
    input_directory = r"/Users/gian/Desktop/CondorResults/VDT/compareown/run2/data/"
    output_filename = r"/Users/gian/Desktop/CondorResults/VDT/compareown/run2/tree_data.csv"

    all_dicts = []
    # Iterate through all files in the input directory
    for filename in os.listdir(input_directory):
        # Construct full file path
        file_path = os.path.join(input_directory, filename)

        # Check if the file is a JSON file
        if not os.path.isfile(file_path):
            continue

        if file_path.endswith(".csv") or (not filename.startswith("gc")):
            continue

        print(f"will process {filename}")

        with open(file_path, 'r') as file:
            data = json.load(file)
            entries = json_to_tree_data(data)
            all_dicts.extend(entries)

    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(all_dicts)

    # Write the DataFrame to a CSV file
    df.to_csv(output_filename, index=False)


convert_tree_data_to_df()