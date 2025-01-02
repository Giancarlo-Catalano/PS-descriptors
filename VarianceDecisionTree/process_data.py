import itertools
import json
import os

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

def produce_to_console():
    input_directory = r"/Users/gian/Desktop/CondorResults/compareown_out/"
    output_filename = r"/Users/gian/PycharmProjects/PS-descriptors/resources/variance_tree_materials/processed_condor"+utils.get_formatted_timestamp()+".csv"

    all_dicts = []
    # Iterate through all files in the input directory
    for filename in os.listdir(input_directory):
        # Construct full file path
        file_path = os.path.join(input_directory, filename)

        # Check if the file is a JSON file
        if not os.path.isfile(file_path):
            continue

        with open(file_path, 'r') as file:
            data = json.load(file)
            entries = json_to_entries(data)
            all_dicts.extend(entries)

    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(all_dicts)

    # Write the DataFrame to a CSV file
    df.to_csv(output_filename, index=False)

produce_to_console()