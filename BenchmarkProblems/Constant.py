from BenchmarkProblems.UnitaryProblem import UnitaryProblem
from FirstPaper.FullSolution import FullSolution


class Constant(UnitaryProblem):

    def __init__(self, amount_of_cliques: int, clique_size: int):
        super().__init__(amount_of_cliques, clique_size)

    def get_problem_name(self) -> str:
        return "Constant"

    @staticmethod
    def unitary_function(bitcount: int, clique_size: int) -> float:
        return float(clique_size)

    @staticmethod
    def get_optimal_clique(clique_size: int) -> FullSolution:
        return FullSolution([0 for _ in range(clique_size)])
