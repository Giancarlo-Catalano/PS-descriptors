from typing import TypeAlias, Callable

from Core.EvaluatedFS import EvaluatedFS
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.SearchSpace import SearchSpace

Fitness: TypeAlias = float
FitnessFunction: TypeAlias = Callable[[FullSolution], Fitness]


class FSEvaluator:
    _fitness_function: FitnessFunction
    used_evaluations: int

    def __init__(self,
                 fitness_function: FitnessFunction):
        self._fitness_function = fitness_function
        self.used_evaluations = 0

    def evaluate(self, fs: FullSolution) -> Fitness:
        self.used_evaluations += 1
        return self._fitness_function(fs)

    def evaluate_population(self, population: list[EvaluatedFS]) -> list[EvaluatedFS]:
        for individual in population:
            individual.fitness = self.evaluate(individual)
        return population

    def generate_pRef_from_full_solutions(self,
                                          search_space: SearchSpace,
                                          samples: list[FullSolution]) -> PRef:
        fitnesses = [self.evaluate(sample) for sample in samples]
        return PRef.from_full_solutions(full_solutions=samples,
                                        fitness_values=fitnesses,
                                        search_space=search_space)

    def generate_pRef_from_search_space(self,
                                        search_space: SearchSpace,
                                        amount_of_samples: int) -> PRef:
        samples = [FullSolution.random(search_space) for _ in range(amount_of_samples)]
        return self.generate_pRef_from_full_solutions(search_space, samples)

    def __repr__(self):
        return f"FS Evaluator, used_budget = {self.used_evaluations}"
