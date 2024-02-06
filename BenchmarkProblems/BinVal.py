from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from FullSolution import FullSolution
from SearchSpace import SearchSpace


class BinVal(BenchmarkProblem):
    amount_of_bits: int

    def __init__(self, amount_of_bits: int):
        self.amount_of_bits = amount_of_bits
        super().__init__(SearchSpace([2 for _ in range(self.amount_of_bits)]))

    def fitness_function(self, fs: FullSolution) -> float:
        result = 0

        header = 1
        for value in reversed(fs.values):
            result += header*value
            header *= 2
        return float(result)