import errno
import json
import os
import platform
import random
import re
import sys
import time
import traceback
import warnings
from collections import defaultdict
from contextlib import ContextDecorator, contextmanager
from typing import Iterable, Any, Callable, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
import plotly.express as px


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

    if any(value <= 1e-5 for value in values):
        raise Exception("In harmonic mean, there are zero values or negative values")

    sum_of_inverses = sum(value ** (-1) for value in values)
    return (sum_of_inverses / len(values)) ** (-1)


def get_descriptive_stats(data: np.ndarray) -> (float, float, float, float, float):
    return np.min(data), np.median(data), np.max(data), np.average(data), np.std(data)


class ExecutionTime(ContextDecorator):
    start_time: float
    end_time: float
    runtime: float

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.runtime = self.end_time - self.start_time

    def __str__(self):
        return f"{self.runtime:.6f}"


def execution_timer():
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
        runtime = self.timer.runtime
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

    # Format the timestamp
    formatted_timestamp = now.strftime("%m-%d-H%H'm'%M's%S")


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


def make_directory(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def open_and_make_directories(path):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    make_directory(os.path.dirname(path))
    return open(path, 'w', encoding= "utf-8")



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



def sort_using_remap_on_functions(items:list, key_functions: list[Callable], reverse=False) -> list:
    values_matrix = np.array([[func(item) for func in key_functions] for item in items])
    normalised_matrix = values_matrix.copy()
    for i in range(normalised_matrix.shape[1]):
        normalised_matrix[:, i] = remap_array_in_zero_one(normalised_matrix[:, i])
    total_scores = np.sum(normalised_matrix, axis=1)
    indexed_total_scores = list(enumerate(total_scores))
    indexed_total_scores.sort(key=second, reverse=reverse)
    return [items[index] for index, value in indexed_total_scores]


def sort_by_combination_of(items: list, key_functions: list[Callable], reverse=False, use_remap=False) -> list:
    if use_remap:
        return sort_using_remap_on_functions(items, key_functions, reverse)
    def get_sorting_for_func(func) -> list[int]:
        sorted_items = sorted(enumerate(items), key=lambda x: func(x[1]))
        return [index for index, item in sorted_items]


    def invert_sorting(sorting: list[int]) -> list[int]:
        final_pos_and_index = sorted(enumerate(sorting), key=second)
        return [final_pos for final_pos, original_index in final_pos_and_index]

    ranks = np.array([invert_sorting(get_sorting_for_func(func)) for func in key_functions])

    summed_ranks = np.sum(ranks, axis=0)

    return [items[index] for index, rank in sorted(enumerate(summed_ranks), key=second, reverse=reverse)]






def flatten(list_of_lists: list[list]) -> list:
    """ crazy that I have to make this function"""
    flat_list = []
    for item in list_of_lists:
        flat_list += item
    return flat_list



def simple_scatterplot(x_label:str, y_label:str, xs: Iterable[float], ys: Iterable[float]):
    if len(xs) != len(ys):
        raise ValueError("The lists must have the same length")

    # Create scatter plot
    plt.scatter(xs, ys)

    # Add title and labels
    plt.title(f'Scatter Plot of {x_label} vs {y_label}')
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Display the plot
    plt.show()




def decode_data_from_islets(input_directory: str, output_filename: str):
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
            all_dicts.extend(data)

    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(all_dicts)

    # Write the DataFrame to a CSV file
    df.to_csv(output_filename, index=False)

def cycle(items: list, shift: int) -> list:
        return items[-shift:]+items[:-shift]


def count_frequency_in_containers(containers: list[Iterable], catalog: list) -> np.ndarray:
    def get_presence_array(container: Iterable) -> np.ndarray:
        return np.array([item in container for item in catalog])

    counts = list(map(get_presence_array, containers))
    return np.average(counts, axis=0)


def make_interactive_3d_plot(first_metric, second_metric, third_metric, names: list[str]):
    metric_matrix = np.array(list(zip(first_metric, second_metric, third_metric)))
    df = pd.DataFrame(metric_matrix, columns=names)
    # Create a 3D scatter plot with Plotly Express
    fig = px.scatter_3d(
        df,
        x=names[0],
        y=names[1],
        z=names[2],
        title=f"3D Scatter Plot of {names}",
        labels={
            names[0]: names[0],
            names[1]: names[1],
            names[2]: names[2]
        }
    )
    # Determine tick values and text for each axis
    x_ticks = np.linspace(df[names[0]].min(), df[names[0]].max(), 10)
    y_ticks = np.linspace(df[names[1]].min(), df[names[1]].max(), 10)
    z_ticks = np.linspace(df[names[2]].min(), df[names[2]].max(), 10)

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                tickvals=x_ticks,
                ticktext=[f"{tick:.2f}" for tick in x_ticks]
            ),
            yaxis=dict(
                tickvals=y_ticks,
                ticktext=[f"{tick:.2f}" for tick in y_ticks]
            ),
            zaxis=dict(
                tickvals=z_ticks,
                ticktext=[f"{tick:.2f}" for tick in z_ticks]
            )
        )
    )


    fig.show()



def remap_in_range_0_1_knowing_range(value: float, known_range: (float, float)) -> float:
    return (value - known_range[0]) / (known_range[1] - known_range[0])


def get_count_report(iterable: Iterable) -> dict:
    result_dict = defaultdict(int)
    for item in iterable:
        result_dict[item] = result_dict[item]+1

    return result_dict


def third(x):
    return x[2]


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))


def make_path(original_path: str):
    to_remove = r"C:\Users\gac8\PycharmProjects\PS-PDF"+"\\"
    if not original_path.startswith(to_remove):
        raise Exception("The string does not start right")
    to_break = original_path[len(to_remove):]
    words = to_break.split("\\")
    return "os.path.join("+(", ".join(f"\"{w}\"" for w in words))+")"




# regex building utilities

def space_on_either_side(given: str) -> str:
    return r"\s*" + given + r"\s*"

def an_integer() -> str:
    return r"-?\d+"


def capture(given: str) -> str:
    return f"({given})"


def parse_simple_input(format_string: str, user_input: str, explain_error=False) -> Optional[list[int]]:
    """numbers are indicated with X"""
    amount_of_numbers = format_string.count("X")
    pattern =format_string.replace("(", r"\(")
    pattern = pattern.replace(")", r"\)")
    pattern = pattern.replace("X", r"\s*(-?\d+)\s*")
    match = re.match(pattern, user_input)

    if not match:
        print(f"Wrong format (expected is {format_string}), please try again")
        return None

    result_strs = [match.group(i+1) for i in range(amount_of_numbers)]
    result_numbers = []
    for index, item in enumerate(result_strs):
        try:
            result_numbers.append(int(item))
        except:
            print(f"The {index+1}th number is wrong, please try again")
            return None

    return result_numbers



weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def remap_array(original: np.ndarray, new_min: float, new_max: float):
    observed_min = np.min(original)
    observed_max = np.max(original)

    if observed_max - observed_min < 1e-05:
        return np.ones_like(original) / 2

    in_zero_one = (original - observed_min) / (observed_max - observed_min)

    return in_zero_one * (new_max-new_min) + new_min


def get_p_value_significance(p_value: float) -> str:
    if p_value > 0.05:
        return "INSIGNIFICANT"
    elif 0.01 < p_value <= 0.05:
        return "SIGNIFICANT"
    elif 0.001 < p_value <= 0.01:
        return "VERY SIGNIFICANT"
    else:
        return "HIGHLY SIGNIFICANT"


def top_with_safe_ties(items: list, key: Callable, lowest: bool = False, highest: bool = False) -> list:
    items_with_keys = [(item, key(item)) for item in items]
    if not (lowest or highest):
        highest = True
    best_key = (max if highest else min)(key for item, key in items_with_keys)
    return [item for item, key in items_with_keys if key == best_key]


def plot_ground_truth_vs_predictions(x_axis_label, x_axis_values, y_axis_label, y_axis_values, title):
    # thanks to Mr GPT
    """
    Plots a scatterplot comparing ground truth and predicted values,
    and visualizes variance with a line of best fit and a 45-degree reference line.

    Parameters:
    ground_truth (numpy.ndarray): Array of ground truth values.
    predictions (numpy.ndarray): Array of predicted values.
    """
    if x_axis_values.shape != y_axis_values.shape:
        raise ValueError(f"ground_truth and predictions must have the same shape. (shapes are {x_axis_values.shape}, {y_axis_values.shape})")

    # Calculate the variance of the residuals
    residuals = x_axis_values - y_axis_values
    variance = np.var(residuals)

    # Create the scatterplot
    plt.figure(figsize=(8, 8))
    plt.scatter(x_axis_values, y_axis_values, alpha=0.6, label=f'Variance: {variance:.2f}')

    # Plot the 45-degree line for reference
    max_val = max(max(x_axis_values), max(y_axis_values))
    min_val = min(min(x_axis_values), min(y_axis_values))
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='red', label='45-degree line')

    # Fit a line to the data for visualization
    m, b = np.polyfit(x_axis_values, y_axis_values, 1)
    plt.plot(x_axis_values, m * x_axis_values + b, color='blue', label=f'Best Fit: y={m:.2f}x+{b:.2f}')

    # Add labels and legend
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Show the plot
    plt.tight_layout()
    return plt


def get_resources_directory():
    script_location = sys.argv[0]
    script_folder = os.path.dirname(script_location)

    # construct the location of a resource
    return os.path.join(script_folder, "resources")



def get_os():
    return platform.system()


def shuffled(original: Iterable) -> list:
    result = list(original)
    random.shuffle(result)
    return result