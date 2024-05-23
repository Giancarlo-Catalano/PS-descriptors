import os
import time
from contextlib import ContextDecorator
from typing import Iterable, Any, Callable, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime


def unzip(zipped):
    if len(zipped) == 0:
        return []

    group_amount = len(zipped[0])

    def get_nth_group(n):
        return [elem[n] for elem in zipped]

    return tuple(get_nth_group(n) for n in range(group_amount))


def remap_array_in_zero_one(input_array: np.ndarray):
    """remaps the values in the given array to be between 0 and 1"""
    min_value = np.min(input_array)
    max_value = np.max(input_array)

    if min_value == max_value:
        return np.full_like(input_array, 0.5, dtype=float)  # all 0.5!

    return (input_array - min_value) / (max_value - min_value)


def harmonic_mean(values: Iterable[float]) -> float:
    if len(values) == 0:
        raise Exception("Trying to get the harmonic mean of no values!")

    sum_of_inverses = sum(value ** (-1) for value in values)
    return (sum_of_inverses / len(sum_of_inverses)) ** (-1)


def get_descriptive_stats(data: np.ndarray) -> (float, float, float, float, float):
    return np.min(data), np.median(data), np.max(data), np.average(data), np.std(data)


class ExecutionTime(ContextDecorator):
    start_time: float
    end_time: float
    execution_time: float

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.execution_time = self.end_time - self.start_time

    def __str__(self):
        return f"{self.execution_time:.6f}"


def execution_time():
    return ExecutionTime()


class Announce(ContextDecorator):
    action_str: str
    timer: ExecutionTime
    verbose: bool

    def __init__(self, action_str: str, verbose=True):
        self.action_str = action_str
        self.timer = ExecutionTime()
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            print(self.action_str, end="...")
        self.timer.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.timer.__exit__(exc_type, exc_val, exc_tb)
        runtime = self.timer.execution_time
        if self.verbose:
            print(f"...Finished (took {runtime:2f} seconds)")


def announce(action: str, verbose=True):
    return Announce(action, verbose)


""" Timing example
    with execution_time() as time:
        data = function()
    
    time.execution_time
    print(time)
    print(data)

"""


def indent(input: str) -> str:
    lines = input.split("\n")
    lines = ["\t" + line for line in lines]
    return "\n".join(lines)


def break_list(input_list: list[Any], group_size: int) -> list[list[Any]]:
    def start(which):
        return group_size * which

    def end(which):
        return group_size * (which + 1)

    return [input_list[start(i):end(i)] for i in range(len(input_list) // group_size)]


def join_lists(many_lists: Iterable[list]) -> list:
    result = []
    for sub_list in many_lists:
        result.extend(sub_list)

    return result


def plot_sequence_of_points(sequence):
    x_points, y_points = unzip(list(enumerate(sequence)))
    plot = plt.plot(x_points, y_points)
    return plot


def merge_csv_files(first_file_name: str, second_file_name: str, output_file_name: str):
    concatenated_df = pd.concat([pd.read_csv(file) for file in [first_file_name, second_file_name]], ignore_index=True)
    concatenated_df.to_csv(output_file_name, index=False)

def get_mean_error(values: Iterable) -> float:
    if len(values) < 1:
        return np.nan
    mean = np.average(values)
    return sum(abs(x - mean) for x in values) / len(values)

def get_max_difference(values: Iterable) -> float:
    return max(values) - min(values)


def get_min_difference(values: Iterable) -> float:
    to_check = np.array(sorted(values))
    differences = to_check[1:] - to_check[:-1]
    return min(differences)


def get_statistical_info_about_iterable(values: Iterable, var_name: str) -> dict:
    return {f"{var_name}_mean": np.average(values),
            f"{var_name}_mean_error": get_mean_error(values),
            f"{var_name}_min_diff": get_min_difference(values),
            f"{var_name}_max_diff": get_max_difference(values)}



def get_formatted_timestamp():
    # Get the current time
    now = datetime.now()

    # Format the timestamp as minutes:hours_day-month
    formatted_timestamp = now.strftime("%m-%d-H%H'M'%m's%S")


    return formatted_timestamp


def prepend_to_file_name(file_path: str, prefix: str) -> str:

    directory, file_name = os.path.split(file_path)

    # Define the new file name with "indexed_" prefix
    new_file_name = prefix + file_name

    # Combine the directory with the new file name to get the full path
    return os.path.join(directory, new_file_name)

def make_copy_of_CSV_file_with_rank_column(file_name: str):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_name)

        # Add a "Rank" column, starting from 1
        df['Rank'] = df.index + 1

        # Define the new file name
        new_file_name = prepend_to_file_name(file_name, "indexed_")

        # Save the DataFrame to a new CSV file
        df.to_csv(new_file_name, index=False)

        print("Saved the ranked file as {")

        return new_file_name  # Return the name of the new CSV file



def as_float_tuple(items: Iterable) -> tuple:
    """mainly to prevent json issues"""
    return tuple(float(item) for item in items)


def make_folder_if_not_present(file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)



def repr_with_precision(iterable: Iterable, significant_digits: int) -> str:
    return "["+", ".join(f"{v:.{significant_digits}}" for v in iterable) + "]"



def get_mean_and_mean_error(iterable: Iterable) -> (float, float):
    mean = np.mean(iterable)
    mean_error = get_mean_error(iterable)
    return mean, mean_error




def make_joined_bt_dataset():


    # Paths to your CSV files
    faulty_csv = r"C:\Users\gac8\PycharmProjects\PS-PDF\Experimentation\FaultyerBTTemp\ps_properties.csv"
    correct_csv = r"C:\Users\gac8\PycharmProjects\PS-PDF\Experimentation\BTDetectorTemp\ps_properties.csv"

    # Read the CSV files into DataFrames
    faulty_df = pd.read_csv(faulty_csv)
    correct_df = pd.read_csv(correct_csv)

    # Add a new column "Faulty" with True for the first DataFrame and False for the second
    faulty_df['Faulty'] = True
    correct_df['Faulty'] = False

    # Concatenate the two DataFrames
    concatenated_df = pd.concat([faulty_df, correct_df], ignore_index=True)

    # Export the concatenated DataFrame to a new CSV file
    output_csv_file = r"C:\Users\gac8\PycharmProjects\PS-PDF\Experimentation\FaultyerBTTemp\ps_properties_again.csv"
    concatenated_df.to_csv(output_csv_file, index=False)

    print("CSV files have been concatenated and a 'Faulty' column has been added.")



def ecdf(value:float, dataset: list[float]):
    dataset.sort()


    index_just_before = 0
    while index_just_before < len(dataset) and dataset[index_just_before] < value :
        index_just_before +=1

    index_just_after = index_just_before
    while index_just_after < len(dataset) and dataset[index_just_after] == value:
        index_just_after += 1


    mean_index = (index_just_before + index_just_after) / 2
    total_quantity = len(dataset)

    return mean_index / total_quantity



def second(p):
    return p[1]


def first(p):
    return p[0]

def sort_by_combination_of(items: list, key_functions: list[Callable], reverse=False) -> list:
    def get_sorting_for_func(func) -> list[int]:
        sorted_items = sorted(enumerate(items), key=lambda x: func(x[1]))
        return [index for index, item in sorted_items]


    def invert_sorting(sorting: list[int]) -> list[int]:
        final_pos_and_index = sorted(enumerate(sorting), key=second)
        return [final_pos for final_pos, original_index in final_pos_and_index]

    ranks = np.array([invert_sorting(get_sorting_for_func(func)) for func in key_functions])

    summed_ranks = np.sum(ranks, axis=0)

    return [items[index] for index, rank in sorted(enumerate(summed_ranks), key=second, reverse=reverse)]

