from enum import Enum, auto

import numpy as np

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from BenchmarkProblems.Constant import Constant
from BenchmarkProblems.OneMax import OneMax
from BenchmarkProblems.ParityProblem import ParityProblem
from BenchmarkProblems.RoyalRoad import RoyalRoad
from BenchmarkProblems.Trapk import Trapk
from FirstPaper.FullSolution import FullSolution
from FirstPaper.PS import PS
from FirstPaper.SearchSpace import SearchSpace
from FirstPaper.custom_types import ArrayOfInts


class ToyProblem(Enum):
    CONSTANT = auto()
    ONEMAX = auto()
    ROYALROAD = auto()
    TRAPK = auto()
    PARITY = auto()

    def __repr__(self):
        return "CORTP"[self.value - 1]

    def __str__(self):
        return "CORTP"[self.value - 1]


def toy_enum_from_string(string: str) -> ToyProblem:
    match string:
        case "O":
            return ToyProblem.ONEMAX
        case "R":
            return ToyProblem.ROYALROAD
        case "T":
            return ToyProblem.TRAPK
        case "P":
            return ToyProblem.PARITY
        case _:
            return ToyProblem.CONSTANT


class ToyAmalgam(BenchmarkProblem):
    problems = list[ToyProblem]
    clique_size = int

    def __init__(self, toy_problems: str, clique_size: int):
        self.problems = [toy_enum_from_string(s) for s in toy_problems]
        self.clique_size = clique_size
        super().__init__(SearchSpace([2 for _ in range(self.amount_of_bits)]))

    @property
    def amount_of_bits(self) -> int:
        return len(self.problems) * self.clique_size

    @staticmethod
    def get_class_per_problem_enum(problem: ToyProblem):
        match problem:
            case ToyProblem.CONSTANT:
                return Constant
            case ToyProblem.ONEMAX:
                return OneMax
            case ToyProblem.ROYALROAD:
                return RoyalRoad
            case ToyProblem.TRAPK:
                return Trapk
            case ToyProblem.PARITY:
                return ParityProblem

    def fitness_for_clique_bitcount(self, bitcount: int, problem: ToyProblem) -> float:
        if problem == ToyProblem.CONSTANT:
            return float(self.clique_size)
        else:
            return self.get_class_per_problem_enum(problem).unitary_function(bitcount, self.clique_size)

    def get_bit_counts(self, full_solution: FullSolution) -> ArrayOfInts:
        bits = full_solution.values.reshape((-1, self.clique_size))
        return np.sum(bits, axis=1)

    def fitness_function(self, fs: FullSolution) -> float:
        bit_counts = self.get_bit_counts(fs)
        return sum(self.fitness_for_clique_bitcount(bc, problem) for bc, problem in zip(bit_counts, self.problems))

    def __repr__(self):
        return f"ToyAmalgam({''.join(f'{p}' for p in self.problems)}, clique size = {self.clique_size}"

    def get_targets(self) -> list[PS]:
        def make_target(problem_class, position):
            start = position * self.clique_size
            end = start + self.clique_size
            result_values = np.full(shape=self.search_space.amount_of_parameters, fill_value=-1, dtype=int)
            result_values[start:end] = problem_class.get_optimal_clique(self.clique_size).values
            return PS(result_values)

        return [make_target(self.get_class_per_problem_enum(problem), clique_index)
                for clique_index, problem in enumerate(self.problems)]
