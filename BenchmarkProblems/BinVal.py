from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from FirstPaper.FullSolution import FullSolution
from FirstPaper.PS import PS
from FirstPaper.SearchSpace import SearchSpace


class BinVal(BenchmarkProblem):
    amount_of_bits: int

    def __init__(self, amount_of_bits: int):
        self.amount_of_bits = amount_of_bits
        super().__init__(SearchSpace([2 for _ in range(self.amount_of_bits)]))

    def fitness_function(self, fs: FullSolution) -> float:
        result = 0

        header = 1
        for value in reversed(fs.values):
            result += header * value
            header *= 2
        return float(result)

    def get_targets(self) -> list[PS]:
        empty = PS.empty(self.search_space)
        return [empty.with_fixed_value(variable_position=var, fixed_value=1) for var in range(self.amount_of_bits)]
