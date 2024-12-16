import itertools

import numpy as np

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.EvaluatedFS import EvaluatedFS
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS
from Core.SearchSpace import SearchSpace


class TSP(BenchmarkProblem):
    cities: list[(int, int)]
    starting_ending_city: (int, int)

    def __init__(self,
                 cities: list[(int, int)],
                 starting_ending_city: (int, int)):
        self.cities = cities
        self.starting_ending_city = starting_ending_city

        search_space = SearchSpace(reversed(range(1, len(cities))))
        super().__init__(search_space)

    @property
    def n(self):
        return len(self.cities)

    def convert_solution_to_city_indexes(self, fs: FullSolution) -> list[int]:
        path = []
        remaining_indexes = list(range(self.n))
        for value in fs.values:
            path.append(remaining_indexes.pop(value))
        path.extend(remaining_indexes)  # the forced choice at the end
        return path

    def city_indexes_to_solution(self, path: list[int]) -> FullSolution:
        remaining_indexes = list(range(self.n))

        combinatorial_indexes = []
        for city_index in path[:(self.n + 1)]:  # this [:self.n] is to exclude the retuning trip if it's present
            index_in_stack = remaining_indexes.index(city_index)
            remaining_indexes.pop(index_in_stack)
            combinatorial_indexes.append(index_in_stack)
        return FullSolution(combinatorial_indexes)

    @classmethod
    def distance_between_cities(cls, city_a, city_b) -> float:
        return np.sqrt(sum((coord_a - coord_b) ** 2 for coord_a, coord_b in zip(city_a, city_b)))

    def fitness_function(self, fs: FullSolution) -> float:
        city_indexes = self.convert_solution_to_city_indexes(fs)
        path = [self.cities[index] for index in city_indexes]
        path = [self.starting_ending_city] + path + [self.starting_ending_city]
        total_distance = sum(self.distance_between_cities(a, b) for a, b in itertools.pairwise(path))
        return -total_distance

    def __repr__(self):
        return f"TSP(cities = {len(self.cities)})"

    def repr_city_index(self, city_index: int) -> str:
        return utils.alphabet[city_index]

    def repr_fs(self, fs: FullSolution) -> str:
        path_str = "->".join(map(self.repr_city_index, self.convert_solution_to_city_indexes(fs)))
        return "O->" + path_str + "->O"

    def get_boolean_search_space(self) -> SearchSpace:
        amount_of_cells = (self.n ** 2 - self.n) / 2

        return SearchSpace(2 for _ in range(amount_of_cells))

    def convert_fs_to_boolean_fs(self, fs: FullSolution) -> FullSolution:
        indexes = self.convert_solution_to_city_indexes(fs)
        indexes = indexes[:-1]  # we ignore returning to the start
        result_as_matrix = np.zeros(shape=(self.n, self.n), dtype=bool)
        for city_index_before, city_index_after in itertools.combinations(indexes, r=2):
            result_as_matrix[city_index_before, city_index_after] = True

        return FullSolution(result_as_matrix[np.triu_indices(self.n, k=1)])

    def convert_pRef(self, pRef: PRef) -> PRef:
        new_search_space = self.get_boolean_search_space()
        old_solutions = pRef.get_evaluated_FSs()
        new_solutions = [EvaluatedFS(self.convert_fs_to_boolean_fs(fs), fs.fitness)
                         for fs in old_solutions]
        return PRef.from_evaluated_full_solutions(evaluated_fss=new_solutions, search_space=new_search_space)

    def represent_boolean_ps(self, ps: PS) -> str:
        precedence = np.zeros(shape=(self.n, self.n), dtype=bool)
        precedence[np.triu_indices(n=self.n, k=1)] = ps.values

        precedence_pairs = []
        for (city_before, city_after) in itertools.combinations(range(self.n), r=2):
            cell_value = precedence[city_before, city_after]
            match cell_value:
                case True:
                    precedence_pairs.append((city_before, city_after))
                case False:
                    precedence_pairs.append((city_after, city_before))
                case _:
                    pass

        def repr_item(item) -> str:
            return f"{self.repr_city_index(item[0])} -> {self.repr_city_index(item[1])}"

        return "\n".join(map(repr_item, precedence_pairs))

    @classmethod
    def get_berlin52_instance(cls):
        return cls(cities=
                   [(25, 185), (345, 750), (945, 685), (845, 655), (880, 660),
                    (25, 230), (525, 1000), (580, 1175), (650, 1130), (1605, 620),
                    (1220, 580), (1465, 200), (1530, 5), (845, 680), (725, 370), (145, 665), (415, 635), (510, 875),
                    (560, 365), (300, 465), (520, 585), (480, 415), (835, 625), (975, 580), (1215, 245), (1320, 315),
                    (1250, 400), (
                        660, 180), (410, 250), (420, 555), (575, 665), (1150, 1160), (700, 580), (685, 595), (685, 610),
                    (770, 610), (
                        795, 645), (720, 635), (760, 650), (475, 960), (95, 260), (875, 920), (700, 500), (555, 815),
                    (830, 485), (
                        1170, 65), (830, 610), (605, 625), (595, 360), (1340, 725), (1740, 245)],
                   starting_ending_city=(565, 575))


class BooleanSearchSpaceTSP(BenchmarkProblem):
    original_problem: TSP

    def __init__(self,
                 original_problem: TSP):
        self.original_problem = original_problem
        n = original_problem.n

        super().__init__(SearchSpace(2 for _ in range(int((n ** 2 - n) / 2))))

    def __repr__(self):
        return f"BooleanSearchSpace(original = {self.original_problem})"

    @property
    def n(self):
        return self.original_problem.n

    def permutation_fs_to_boolean_fs(self, fs: FullSolution) -> FullSolution:
        indexes = self.original_problem.convert_solution_to_city_indexes(fs)
        indexes = indexes[:-1]  # we ignore returning to the start
        result_as_matrix = np.zeros(shape=(self.n, self.n), dtype=bool)
        for city_index_before, city_index_after in itertools.combinations(indexes, r=2):
            result_as_matrix[city_index_before, city_index_after] = True

        return FullSolution(result_as_matrix[np.triu_indices(self.n, k=1)])

    def boolean_fs_to_permutation_fs(self, permutation_fs: FullSolution) -> FullSolution:
        # and this is not a one to one mapping, so there is some fixing that will take place
        precedence = np.zeros(shape=(self.n, self.n), dtype=bool)
        precedence[np.triu_indices(n=self.n, k=1)] = permutation_fs.values
        precedence = precedence.T
        precedence[np.triu_indices(n=self.n, k=1)] = np.logical_not(permutation_fs.values)

        counts_for_each_row = np.sum(precedence, axis=1)
        city_indexes_and_rank = sorted(enumerate(counts_for_each_row), key=utils.second, reverse=True)
        city_indexes = utils.unzip(city_indexes_and_rank)[0]
        return self.original_problem.city_indexes_to_solution(city_indexes)

    def repr_ps(self, ps: PS) -> str:
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

    def fitness_function(self, fs: FullSolution) -> float:
        combinatorial_fs = self.boolean_fs_to_permutation_fs(fs)
        return self.original_problem.fitness_function(combinatorial_fs)

    def repr_fs(self, full_solution: FullSolution) -> str:
        converted = self.boolean_fs_to_permutation_fs(full_solution)
        return self.original_problem.repr_fs(converted)
