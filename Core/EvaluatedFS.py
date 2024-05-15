import functools

from Core.FullSolution import FullSolution


@functools.total_ordering
class EvaluatedFS(FullSolution):

    full_solution: FullSolution
    fitness: float


    def __init__(self,
                 full_solution: FullSolution,
                 fitness: float):
        self.full_solution = full_solution
        self.fitness = fitness

    def __repr__(self):
        return f"{self.full_solution}, fs score = {self.fitness:.2f}"

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __eq__(self, other):
        return self.full_solution == other.full_solution


    def __hash__(self):
        return hash(self.full_solution)