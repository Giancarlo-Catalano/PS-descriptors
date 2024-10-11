import numpy as np

from BenchmarkProblems.UnitaryProblem import UnitaryProblem
from Core.FullSolution import FullSolution
from Core.PS import PS


class RoyalRoad(UnitaryProblem):

    def __init__(self, amount_of_cliques: int, clique_size: int = 4):
        super().__init__(amount_of_cliques, clique_size)

    def get_problem_name(self) -> str:
        return "RoyalRoad"

    @staticmethod
    def unitary_function(bitcount: int, clique_size: int) -> float:
        if bitcount == clique_size:
            return float(clique_size)
        else:
            return 0.0

    @staticmethod
    def get_optimal_clique(clique_size: int) -> FullSolution:
        return FullSolution([1 for _ in range(clique_size)])


    def get_descriptors_of_ps(self, ps: PS) -> dict:
        return {"Functionality": 0}


    def get_targets(self) -> set[PS]:
        result = set()
        empty_values = np.full(fill_value=-1, shape=self.amount_of_bits)
        for clique in range(self.amount_of_cliques):
            values = empty_values.copy()
            start = clique * self.clique_size
            end = start + self.clique_size
            values[start:end] = 1
            result.add(PS(values))
        return result


    def get_short_code(self)->str:
        return "RR"