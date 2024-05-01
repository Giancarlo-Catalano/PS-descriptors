import os
import time
from contextlib import ContextDecorator
from typing import Iterable, Any

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
    plt.plot(x_points, y_points)
    plt.show()


def merge_csv_files(first_file_name: str, second_file_name: str, output_file_name: str):
    concatenated_df = pd.concat([pd.read_csv(file) for file in [first_file_name, second_file_name]], ignore_index=True)
    concatenated_df.to_csv(output_file_name, index=False)





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