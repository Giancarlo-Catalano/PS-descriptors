from typing import Optional

import numpy as np

from Core.PRef import PRef
from Core.PS import PS
from Core.SearchSpace import SearchSpace
from RepresentationBasedSearch.ContinuousVariables.ContinuousVariableProblem import ContinuousSearchSpace, \
    ContinuousPRef


class RangeRepresentationForProblem:
    original_problem_search_space: ContinuousSearchSpace
    split_depth: int

    variable_names: list[str]

    combinatorial_search_space: SearchSpace


    precomputed_range_fractions: np.ndarray


    def __init__(self,
                 original_problem_search_space: ContinuousSearchSpace,
                 split_depth: int,
                 variable_names: Optional[list[str]] = None):
        self.original_problem_search_space = original_problem_search_space
        self.split_depth = split_depth
        self.combinatorial_search_space = self.get_combinatorial_search_space()

        self.precomputed_range_fractions = (self.original_problem_search_space.upper_bounds - self.original_problem_search_space.lower_bounds) / (2 ** self.split_depth)

        self.variable_names = [f"Var#{v}" for v in range(self.original_problem_search_space.quantity_of_parameters)] if variable_names is None else variable_names

    def get_combinatorial_search_space(self) -> SearchSpace:
        return SearchSpace([2
                            for _ in range(self.original_problem_search_space.quantity_of_parameters)
                            for _ in range(self.split_depth)])


    def encode_continuous_pRef(self, continuous_pRef: ContinuousPRef) -> PRef:
        lb = self.original_problem_search_space.lower_bounds
        ub = self.original_problem_search_space.upper_bounds
        original = continuous_pRef.full_solution_matrix
        normalised_cont_data = (original - lb) / (ub - lb)

        integer_data = normalised_cont_data / self.precomputed_range_fractions



        def convert_row(row: np.ndarray) -> np.ndarray:
            # converts the integers into booleans
            remaining = row.copy()
            accumulated = []
            for _ in range(self.split_depth):
                accumulated.append(remaining % 2)
                remaining //=2

            result_matrix = np.array(accumulated)
            result_matrix = np.flip(result_matrix, 0)
            return result_matrix.T.ravel()

        return np.array([convert_row(row) for row in integer_data])

