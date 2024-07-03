import random

import numpy as np

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from FirstPaper.FullSolution import FullSolution
from FirstPaper.PS import PS, STAR
from FirstPaper.SearchSpace import SearchSpace


class RoyalRoadWithOverlaps(BenchmarkProblem):
    amount_of_cliques: int
    size_of_cliques: int
    amount_of_bits: int

    target_pss: set[PS]

    def __init__(self, amount_of_cliques: int,
                 size_of_cliques: int,
                 amount_of_bits: int):
        self.amount_of_cliques = amount_of_cliques
        self.size_of_cliques = size_of_cliques
        self.amount_of_bits = amount_of_bits
        super().__init__(SearchSpace([2 for _ in range(self.amount_of_bits)]))

        self.target_pss = self.generate_target_PSs()

    def __repr__(self):
        return (f"RRO(amount_of_cliques = {self.amount_of_cliques}, "
                f"size_of_cliques = {self.size_of_cliques}, "
                f"amount_of_bits = {self.amount_of_bits})")

    def long_repr(self) -> str:
        return "Contains the following features:\n" + "\n".join([f"\t{ps}" for ps in self.target_pss])

    def generate_target_PSs(self) -> set[PS]:
        def random_PS() -> PS:
            start_position = random.randrange(self.amount_of_bits - self.size_of_cliques)
            value = random.randrange(2)
            values = np.full(self.amount_of_bits, STAR)
            values[start_position:start_position + self.size_of_cliques] = value
            return PS(values)

        def generate_random_target_set() -> set[PS]:
            return set(random_PS() for _ in range(self.amount_of_cliques))

        def set_contained_duplicates(target_set: set[PS]) -> bool:
            return len(target_set) < self.amount_of_cliques

        target_set = generate_random_target_set()
        while set_contained_duplicates(target_set):
            target_set = generate_random_target_set()
        return target_set

    def fitness_function(self, fs: FullSolution) -> float:
        return float(len([ps for ps in self.target_pss if ps.present_in(fs)]))

    def get_targets(self) -> list[PS]:
        return list(self.target_pss)
