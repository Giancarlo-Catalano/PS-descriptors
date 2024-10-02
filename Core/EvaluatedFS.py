import functools

from Core.FullSolution import FullSolution


@functools.total_ordering
class EvaluatedFS(FullSolution):
    fitness: float

    def __init__(self,
                 full_solution: FullSolution,
                 fitness: float):
        super().__init__(full_solution.values)
        self.fitness = fitness

    def __repr__(self):
        return f"{super().__repr__()}, fs score = {self.fitness:.2f}"

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __eq__(self, other):
        return super().__eq__(other)

    def __hash__(self):
        return super().__hash__()

    def as_full_solution(self):
        return FullSolution(self.values)
