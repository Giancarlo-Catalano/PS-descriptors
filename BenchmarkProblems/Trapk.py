import numpy as np

from BenchmarkProblems.UnitaryProblem import UnitaryProblem
from Core.FullSolution import FullSolution
from Core.PS import PS, STAR


class Trapk(UnitaryProblem):

    def __init__(self, amount_of_cliques: int, clique_size: int):
        super().__init__(amount_of_cliques, clique_size)

    def get_problem_name(self) -> str:
        return "Trapk"

    @staticmethod
    def unitary_function(bitcount: int, clique_size: int) -> float:
        if bitcount == clique_size:
            return float(clique_size)
        else:
            return float(clique_size - bitcount - 1)

    @staticmethod
    def get_optimal_clique(clique_size: int) -> FullSolution:
        return FullSolution([1 for _ in range(clique_size)])


    def get_descriptors_of_ps(self, ps: PS) -> dict:
        def get_average_value_of_fixed_values():
            return np.average([value for value in ps.values if value != STAR])

        average_value_of_cells = 0.5 if ps.is_empty() else get_average_value_of_fixed_values()

        return {"value_avg": average_value_of_cells}
