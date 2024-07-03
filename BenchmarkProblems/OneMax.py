from BenchmarkProblems.UnitaryProblem import UnitaryProblem
from FirstPaper.FullSolution import FullSolution


class OneMax(UnitaryProblem):

    def __init__(self, amount_of_cliques: int, clique_size: int):
        super().__init__(amount_of_cliques, clique_size)

    def get_problem_name(self) -> str:
        return "OneMax"

    @staticmethod
    def unitary_function(bitcount: int, clique_size: int) -> float:
        return float(bitcount)

    @staticmethod
    def get_optimal_clique(clique_size: int) -> FullSolution:
        return FullSolution([1 for _ in range(clique_size)])
