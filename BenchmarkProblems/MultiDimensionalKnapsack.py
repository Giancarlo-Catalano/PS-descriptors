import random
from typing import TypeAlias, Iterable

import numpy as np

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from FirstPaper.FullSolution import FullSolution
from FirstPaper.PS import PS, STAR
from FirstPaper.SearchSpace import SearchSpace
from FirstPaper.custom_types import ArrayOfInts

Item: TypeAlias = ArrayOfInts


class MultiDimensionalKnapsack(BenchmarkProblem):
    amount_of_dimensions: int
    items: np.ndarray
    targets: ArrayOfInts

    penalty: int

    def __init__(self,
                 items: list[Iterable[int]],
                 targets: Iterable[int]):
        assert (len(items) > 0)
        amount_of_dimensions = len(items[0])
        self.amount_of_dimensions = amount_of_dimensions
        self.items = np.array([np.array(item) for item in items])
        self.targets = np.array(targets)
        search_space = SearchSpace([2 for _ in items])
        super().__init__(search_space)
        self.penalty = max(targets) * self.amount_of_dimensions

    @classmethod
    def random(cls,
               amount_of_dimensions: int,
               amount_of_items: int,
               max_value: int):

        def random_item() -> Iterable[int]:
            return [random.randrange(max_value) for _ in range(amount_of_dimensions)]

        items = [random_item() for _ in range(amount_of_items)]
        targets = random_item()

        return cls(items=items,
                   targets=targets)

    def full_solution_to_sum_of_metrics(self, fs: FullSolution) -> ArrayOfInts:
        return np.sum(self.items[fs.values.astype(bool)], axis=0)

    def manhattan_distance(self, gotten: ArrayOfInts, targets: ArrayOfInts) -> float:
        return float(np.sum(np.abs(gotten - targets)))

    def euclidean_distance(self, gotten: ArrayOfInts, targets: ArrayOfInts) -> float:
        return float(np.sqrt(np.square(gotten - targets)))

    def distance_between_metrics(self, gotten: ArrayOfInts, targets: ArrayOfInts) -> float:
        return self.manhattan_distance(gotten, targets)

    def fitness_function(self, fs: FullSolution) -> float:
        gotten = self.full_solution_to_sum_of_metrics(fs)

        fitness = -self.distance_between_metrics(gotten, self.targets)
        if any(gotten > target for gotten, target in zip(gotten, self.targets)):
            return float(fitness - self.penalty)
        else:
            return float(fitness)

    def get_worst_fitness(self):
        all_zeros = FullSolution([0 for item in self.items])
        return self.fitness_function(all_zeros)

    def __repr__(self):
        return f"MultiDimensionalKnapsack(dimensions={self.amount_of_dimensions}, #items={len(self.items)})"

    def long_repr(self) -> str:
        items_str = "\n\t".join(f"{index}\t{item}" for index, item in enumerate(self.items))
        return "MDK(items = \n\t" + items_str + f"\ntargets = {self.targets}"

    def repr_ps(self, ps: PS) -> str:
        def repr_pair(index, value):
            if value == 0:
                return f"¬{index}"
            else:
                return f"{index}"

        return "[" + ", ".join(repr_pair(index, value) for index, value in enumerate(ps.values) if value != STAR) + "]"
