import json
from typing import TypeAlias

import numpy as np

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from FirstPaper.FullSolution import FullSolution
from FirstPaper.PS import PS, STAR
from FirstPaper.SearchSpace import SearchSpace

LinkValue: TypeAlias = int

"""This class will contain the links J between nodes arranged in a grid.
Also note that the grid has toroidal topology, so the bottom nodes are connected to the top,
and the left nodes are connected to the right nodes"""

""" Let the grid be 

  N0   --H0--  N1   --H1--  N2   --H2--

  |            |            |           
  V0           V1           V2
  |            |            |           

  N3   --H3--  N4   --H4--  N5   --H5-- 

  |            |            |           
  V3           V4           V5
  |            |            | 

  N6   --H6--  N7   --H7--  N8   --H8-- 

  |            |            |           
  V6           V7           V8
  |            |            | 

  N9   --H9--  N10  -H10--  N13  -H13--  

  |            |            |           
  V9           V10           V11
  |            |            | 
"""

""" Then 
  *  horizontal_link_values = [[H0, H1, H2], [H3, H4, H5]..]
  *  vertical_link_values = [[V0, V1, V2], [V3, V4, V5]..]
"""


class IsingSpinGlassProblem(BenchmarkProblem):
    width: int
    height: int
    horizontal_link_values: np.ndarray
    vertical_link_values: np.ndarray
    global_optima: int

    amount_of_variables: int

    def __init__(self,
                 horizontal_link_values: np.ndarray,
                 vertical_link_values: np.ndarray,
                 best_fitness: int):
        self.horizontal_link_values = horizontal_link_values
        self.vertical_link_values = vertical_link_values
        self.global_optima = best_fitness

        assert (self.horizontal_link_values.shape == self.vertical_link_values.shape)

        self.width, self.height = self.horizontal_link_values.shape
        self.amount_of_variables = self.width * self.height

        search_space = SearchSpace([2 for var in range(self.amount_of_variables)])
        super().__init__(search_space)

    @classmethod
    def empty(cls, width: int, height: int, best_fitness: int):
        horizontal_link_values = np.zeros((height, width), dtype=LinkValue)
        vertical_link_values = np.zeros((height, width), dtype=LinkValue)
        return cls(horizontal_link_values, vertical_link_values, best_fitness)

    def set_link_value_right(self, row: int, column: int, new_value: LinkValue):
        self.horizontal_link_values[row][column] = new_value

    def set_link_value_down(self, row: int, column: int, new_value: LinkValue):
        self.vertical_link_values[row][column] = new_value

    def set_link_value_up(self, row: int, column: int, new_value: LinkValue):
        new_row = row - 1 if row != 0 else self.height - 1
        self.set_link_value_down(new_row, column, new_value)

    def set_link_value_left(self, row: int, column: int, new_value: LinkValue):
        new_column = column - 1 if column != 0 else self.width - 1
        self.set_link_value_right(row, new_column, new_value)

    def get_link_value_right(self, row: int, column: int):
        return self.horizontal_link_values[row][column]

    def get_link_value_down(self, row: int, column: int):
        return self.vertical_link_values[row][column]

    def get_link_value_up(self, row: int, column: int):
        new_row = row - 1 if row != 0 else self.height - 1
        return self.get_link_value_down(new_row, column)

    def get_link_value_left(self, row: int, column: int):
        new_column = column - 1 if column != 0 else self.width - 1
        return self.get_link_value_down(row, new_column)

    @classmethod
    def from_sandys_files_unsafe(cls, connections_file_name: str):
        with open(connections_file_name, "r") as file:
            def read_single_number_line() -> int:
                try:
                    return int(file.readline())
                except ValueError:
                    raise Exception(f"Was not able to read a single-integer line from the file {connections_file_name}")

            best_fitness = read_single_number_line()
            amount_of_variables = read_single_number_line()
            dimensions = read_single_number_line()
            unknown_parameter = read_single_number_line()
            width = read_single_number_line()

            height = amount_of_variables // width
            result = cls.empty(width, height, best_fitness)

            for node_index in range(amount_of_variables):
                row = node_index // width
                column = node_index % width
                file_line = file.readline()
                values = [int(substring) for substring in file_line.split()]
                values = values[-4:]  # discard the redundant information

                result.set_link_value_up(row, column, values[0])
                result.set_link_value_down(row, column, values[1])
                result.set_link_value_left(row, column, values[2])
                result.set_link_value_right(row, column, values[3])

            return result

    @classmethod
    def from_sandys_files(cls, filename: str):
        try:
            return cls.from_sandys_files_unsafe(filename)
        except FileNotFoundError:
            raise Exception(f"The file {filename} could not be read")

    def to_gian_file(self, filename: str):
        result = dict()
        result["best_fitness"] = self.global_optima
        result["width"] = self.width
        result["height"] = self.height
        result["horizontal_links"] = self.horizontal_link_values.tolist()
        result["vertical_links"] = self.vertical_link_values.tolist()

        try:
            # Open the file in write mode (w+ creates the file if it doesn't exist)
            with open(filename, 'w+') as file:
                json.dump(result, file, indent=4)
            print(f"IsingProblem data has been successfully written to {filename}.")
        except IOError:
            print(f"Error writing to file {filename}.")

    @classmethod
    def from_json_file(cls, filename: str):
        try:
            with open(filename) as file:
                data = json.load(file)
                best_fitness = data["best_fitness"]
                vertical_links = np.array(data["vertical_links"], dtype=LinkValue)
                horizontal_links = np.array(data["horizontal_links"], dtype=LinkValue)
                return cls(horizontal_link_values=horizontal_links,
                           vertical_link_values=vertical_links,
                           best_fitness=best_fitness)
        except FileExistsError:
            raise Exception(f"The file {filename} could not be written")

    def fitness_function(self, fs: FullSolution) -> float:
        node_values = np.array(fs.values.reshape((self.height, self.width)), dtype=int)
        node_values[node_values == 0] = -1  # before the values were 0 and 1, now they are -1 and 1

        cycled_left = np.hstack((node_values[:, 1:], node_values[:, :1]))
        cycled_up = np.vstack((node_values[1:, :], node_values[:1, :]))

        horizontal_differentials = np.sum(node_values * cycled_left * self.horizontal_link_values)
        vertical_differentials = np.sum(node_values * cycled_up * self.vertical_link_values)

        return horizontal_differentials + vertical_differentials

    def repr_ps(self, ps: PS) -> str:
        def repr_cell(cell_value):
            if cell_value == STAR:
                return " "
            else:
                return f"{cell_value}"

        def repr_row(row: np.ndarray):
            return "[" + (" ".join(repr_cell(cell) for cell in row)) + "]"

        grid = ps.values.reshape((self.height, self.width))

        return "\n" + ("\n".join(repr_row(row) for row in grid))

    def __repr__(self):
        return f"IsingSpinGlass({self.amount_of_variables} x {self.amount_of_variables})"

    def get_global_optima_fitness(self) -> float:
        return float(self.global_optima)
