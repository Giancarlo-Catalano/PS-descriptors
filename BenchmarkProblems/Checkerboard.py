import numpy as np

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from FirstPaper.FullSolution import FullSolution
from FirstPaper.PS import PS, STAR
from FirstPaper.SearchSpace import SearchSpace


class CheckerBoard(BenchmarkProblem):
    def __init__(self, rows: int, columns: int):
        self.rows = rows
        self.columns = columns
        super().__init__(SearchSpace([2 for _ in range(rows * columns)]))

    def __repr__(self):
        return f"Checkerboard({self.rows}, {self.columns})"

    def repr_ps(self, ps: PS) -> str:
        def repr_cell(cell_value: int) -> str:
            if cell_value == STAR:
                return " "
            else:
                return f"{cell_value}"

        def repr_row(row: np.ndarray) -> str:
            return "[" + " ".join(repr_cell(cell) for cell in row) + "]"

        return "\n".join(repr_row(row) for row in ps.values.reshape((self.rows, self.columns)))

    def fitness_function(self, fs: FullSolution) -> float:
        grid = fs.values.reshape((self.rows, self.columns))
        grid_without_last_row = grid[:-1]
        grid_without_last_column = grid[:, :-1]

        grid_shifted_up = grid[1:]
        grid_shifted_left = grid[:, 1:]

        vertical_diffs = np.sum(grid_without_last_row != grid_shifted_up)
        horizontal_diffs = np.sum(grid_without_last_column != grid_shifted_left)
        return float(horizontal_diffs + vertical_diffs)

        # could have been np.sum(grid[1:] != grid[:-1]) + np.sum(grid[:, 1:] != grid[:, :-1])

    @staticmethod
    def fitness_of_flat_clique(fs: FullSolution) -> float:
        return float(np.sum(fs.values[:-1] != fs.values[1:]))
