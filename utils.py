import itertools
from typing import Any

import numpy as np
from pandas import DataFrame
import plotly.express as px


def first(pair: (Any, Any)) -> Any:
    return pair[0]


def second(pair: (Any, Any)) -> Any:
    return pair[1]


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


def remap_each_column_in_zero_one(input_matrix: np.ndarray) -> np.ndarray:
    result_matrix = np.zeros_like(input_matrix)
    _, columns = input_matrix.shape
    for column in range(columns):
        result_matrix[:, column] = remap_array_in_zero_one(input_matrix[:, column])
    return result_matrix


def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))



def make_interactive_3d_plot(points, labels: list[str]):
    df = DataFrame(data=points, columns=labels)
    fig = px.scatter_3d(df, x=labels[0], y=labels[1], z=labels[2])
    fig.show()
