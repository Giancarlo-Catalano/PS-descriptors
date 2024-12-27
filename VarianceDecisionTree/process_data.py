import itertools
import json

import pandas as pd


# {
#         "problem_name": "SAT_S",
#         "own_method_settings": {
#             "ps_budget": 2000,
#             "ps_population": 50
#         },
#         "sample_size": 10000,
#         "pRef_method": "GA",
#         "iai": {
#             "2": 10.048661805416899,
#             "3": 8.750811010146373,
#             "4": 7.603486883425121
#         },
#         "naive": {
#             "2": 8.452766413913302,
#             "3": 7.435766892645918,
#             "4": 6.630736099736666
#         },
#         "ps": {
#             "2": 6.8800212108979855,
#             "3": 6.38874425352788,
#             "4": 5.680608471756772
#         }
#     },

def json_to_df(data: dict) -> pd.DataFrame:
    def item_to_list_of_entries(item) -> list[dict]:
        problem_name = item["problem_name"]
        pRef_method = item["pRef_method"]
        dt_methods = ["iai", "naive", "ps"]
        return [{"problem_name": problem_name,
                 "pRef_method": pRef_method,
                 "dt_method": dt_method,
                 "depth": depth,
                 "mse": mse}
                for dt_method in dt_methods
                for depth, mse in item[dt_method].items()
                ]

    return pd.DataFrame(row
                        for item in data
                        for row in item_to_list_of_entries(item))


def read_file_and_make_table(filename: str) -> pd.DataFrame:

    with open(filename, "r") as file:
        data = json.load(file)

    return json_to_df(data)

def produce_to_console():
    path = r"/Users/gian/PycharmProjects/PS-descriptors/resources/variance_tree_materials/peogram_output.json"
    table = read_file_and_make_table(path)
    output_path = r"/Users/gian/PycharmProjects/PS-descriptors/resources/variance_tree_materials/result_as_csv.csv"
    table.to_csv(output_path)

produce_to_console()