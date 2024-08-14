from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.FullSolution import FullSolution
from Core.SearchSpace import SearchSpace


class MultiPlexerProblem(BenchmarkProblem):
    # this is the small one
    def __init__(self):
        super().__init__(search_space=SearchSpace([2 for _ in range(6)]))

    def fitness_function(self, fs: FullSolution) -> float:
        values = fs.values
        index = values[0]*2+values[1]
        return float(values[index+2])


