import itertools
from typing import Optional

import numpy as np

from BenchmarkProblems.TSP import TSP
from Core.FullSolution import FullSolution
from Core.PS import PS
from Core.SearchSpace import SearchSpace
from RepresentationBasedSearch.ProblemRepresentation import ProblemRepresentation, Representation


class TSPPrecedenceRepresentation(ProblemRepresentation):
    original_problem: TSP

    def __init__(self, original_problem: TSP):
        super().__init__(original_problem)

    @property
    def n(self) -> int:
        return self.original_problem.n

    def get_representation_search_space(self) -> SearchSpace:
        return SearchSpace(2 for _ in range(int((self.n ** 2 - self.n) / 2)))

    def get_representation(self, solution: FullSolution) -> Representation:
        indexes = self.original_problem.convert_solution_to_city_indexes(solution)
        indexes = indexes[:-1]  # we ignore returning to the start
        result_as_matrix = np.zeros(shape=(self.n, self.n), dtype=bool)
        for city_index_before, city_index_after in itertools.combinations(indexes, r=2):
            result_as_matrix[city_index_before, city_index_after] = True

        return FullSolution(result_as_matrix[np.triu_indices(self.n, k=1)])

    def repr_partial_representation(self, ps: PS) -> str:
        precedence = np.zeros(shape=(self.n, self.n), dtype=int)
        precedence[np.triu_indices(n=self.n, k=1)] = ps.values

        precedence_pairs = []
        for (city_before, city_after) in itertools.combinations(range(self.n), r=2):
            cell_value = precedence[city_before, city_after]
            match cell_value:
                case 1:
                    precedence_pairs.append((city_after, city_before))
                case 0:
                    precedence_pairs.append((city_before, city_after))
                case _:  # a star
                    pass

        def repr_item(item) -> str:
            return f"{self.original_problem.repr_city_index(item[0])} -> {self.original_problem.repr_city_index(item[1])}"

        return "\n".join(map(repr_item, precedence_pairs))


class TSPVicinityRepresentation(ProblemRepresentation):
    original_problem: TSP
    vicinity_threshold: int

    def __init__(self,
                 original_problem: TSP,
                 vicinity_threshold: Optional[int] = None):
        super().__init__(original_problem)
        self.vicinity_threshold = vicinity_threshold if vicinity_threshold is not None else self.n // 5

    @property
    def n(self) -> int:
        return self.original_problem.n

    def get_representation_search_space(self) -> SearchSpace:
        return SearchSpace(2 for _ in range(int((self.n ** 2 - self.n) / 2)))

    def get_representation(self, solution: FullSolution) -> Representation:
        indexes = self.original_problem.convert_solution_to_city_indexes(solution)
        indexes = indexes[:-1]  # we ignore returning to the start
        result_as_matrix = np.zeros(shape=(self.n, self.n), dtype=bool)

        def register_for_threshold_equal_to(v: int):
            for city_a, city_b in zip(indexes, indexes[v:]):
                result_as_matrix[city_a, city_b] = True
                result_as_matrix[city_b, city_a] = True  # just in case

        for v in range(1, self.vicinity_threshold):
            register_for_threshold_equal_to(v)

        return FullSolution(result_as_matrix[np.triu_indices(self.n, k=1)])

    def repr_partial_representation(self, ps: PS) -> str:
        vicinities = np.zeros(shape=(self.n, self.n), dtype=int)-1
        vicinities[np.triu_indices(n=self.n, k=1)] = ps.values

        vicinity_pairs = []
        for (city_before, city_after) in itertools.combinations(range(self.n), r=2):
            cell_value = vicinities[city_before, city_after]
            if cell_value > -1:
                vicinity_pairs.append((city_before, city_after, cell_value))

        def repr_item(item) -> str:
            return f"{self.original_problem.repr_city_index(item[0])} {'NOT' if item[2] == 0 else ''} near {self.original_problem.repr_city_index(item[1])}"

        return "\n".join(map(repr_item, vicinity_pairs))
