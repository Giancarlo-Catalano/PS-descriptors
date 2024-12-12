import heapq
import random
from typing import Iterable, Callable

import numpy as np


class ContinuousSearchSpace:
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray

    def __init__(self, lower_bounds: Iterable[float], upper_bounds: Iterable[float]):
        self.lower_bounds = np.array(lower_bounds, float)
        self.upper_bounds = np.array(upper_bounds, float)
        assert (self.lower_bounds.shape == self.upper_bounds.shape)

    @property
    def quantity_of_parameters(self):
        return len(self.lower_bounds)

    def get_random_item(self) -> np.ndarray:
        return np.random.uniform(self.lower_bounds, self.upper_bounds)

    def get_random_items(self, how_many: int) -> np.ndarray:
        return np.random.uniform(self.lower_bounds, self.upper_bounds, (how_many, len(self.lower_bounds)))

    def get_mutation_operator(self, mutation_rate: float):
        standard_deviations = (self.upper_bounds - self.lower_bounds) * mutation_rate

        def mutate(original):
            gaussians = np.random.normal(loc=original, scale=standard_deviations)
            clipped = np.clip(gaussians, self.lower_bounds, self.upper_bounds)
            return clipped

        return mutate

    def get_crossover_operator(self):
        qty_vars = len(self.lower_bounds)

        def crossover(mother, father) -> (np.ndarray, np.ndarray):
            swapping_positions = np.random.random(qty_vars) < 0.5
            daughter = mother.copy()
            son = father.copy()

            daughter[swapping_positions] = father[swapping_positions]
            son[~swapping_positions] = mother[~swapping_positions]

            return daughter, son

        return crossover

    def get_tournament_selection_operator(self, tournament_size: int):
        def select(fitnesses: np.ndarray) -> int:
            n = len(fitnesses)
            indexes = random.sample(range(n), k=tournament_size)
            winner_index = max(indexes, key=lambda i: fitnesses[i])
            return winner_index

        return select

    def get_truncation_selection_operator(self, qty_to_select: int):
        def select(fitnesses: np.ndarray) -> np.ndarray:
            n = len(fitnesses)
            winner_indexes = heapq.nlargest(n=qty_to_select, iterable=range(n), key=lambda i: fitnesses[i])
            return np.array(winner_indexes)

        return select


class ContinuousOptimisationProblem:
    search_space: ContinuousSearchSpace

    def __init__(self, search_space: ContinuousSearchSpace):
        self.search_space = search_space

    def __repr__(self):
        raise NotImplemented

    def fitness_function(self, values: np.ndarray):
        raise NotImplemented

    def get_fitnesses_for_population(self, population: np.ndarray) -> np.ndarray:
        assert (len(population.shape) == 2)
        return np.array([self.fitness_function(row) for row in population])


class ContinuousPRef:
    full_solution_matrix: np.ndarray
    fitnesses: np.ndarray

    def __init__(self,
                 full_solution_matrix: np.ndarray,
                 fitnesses: np.ndarray):
        self.full_solution_matrix = full_solution_matrix
        self.fitnesses = fitnesses

    @property
    def sample_size(self) -> int:
        return len(self.fitnesses)


def apply_ga_to_find_maximum(problem: ContinuousOptimisationProblem,
                             budget: int = 10000,
                             population_size: int = 100,
                             mutation_rate: float = 0.1,
                             crossover_chance: float = 0.7,
                             tournament_size: int = 3,
                             return_pRef: bool = False):
    mutation_operator = problem.search_space.get_mutation_operator(mutation_rate)
    crossover_operator = problem.search_space.get_crossover_operator()
    tournament_selector = problem.search_space.get_tournament_selection_operator(tournament_size)
    truncation_selector = problem.search_space.get_truncation_selection_operator(qty_to_select=population_size)

    used_evaluations = [0]  # nasty hack to make this modifiable by a function

    def single_generation(current_population: np.ndarray, current_fitnesses: np.ndarray) -> (np.ndarray, np.ndarray):
        def make_babies() -> list[np.ndarray]:
            if random.random() < crossover_chance:
                mother = current_population[tournament_selector(current_fitnesses)]
                father = current_population[tournament_selector(current_fitnesses)]
                daughter, son = crossover_operator(mother, father)
                return [mutation_operator(daughter), mutation_operator(son)]
            else:
                original = current_population[tournament_selector(current_fitnesses)]
                return [mutation_operator(original)]

        babies = []
        while len(babies) < population_size:
            babies.extend(make_babies())

        babies = np.array(babies)

        new_fitnesses = problem.get_fitnesses_for_population(babies)
        used_evaluations[0] += len(new_fitnesses)

        current_population = np.vstack((current_population, babies))
        current_fitnesses = np.concatenate((current_fitnesses, new_fitnesses))

        indexes_to_keep = truncation_selector(current_fitnesses)

        current_population = current_population[indexes_to_keep]
        current_fitnesses = current_fitnesses[indexes_to_keep]

        if return_pRef:
            return babies, new_fitnesses

    population = problem.search_space.get_random_items(how_many=population_size)
    fitnesses = problem.get_fitnesses_for_population(population)

    if return_pRef:
        pRef_population = population
        pRef_fitnesses = fitnesses

    while used_evaluations[0] < budget:
        if return_pRef:
            new_babies, new_fitnesses = single_generation(population, fitnesses)
            pRef_population = np.vstack(pRef_population, new_babies)
            pRef_fitnesses = np.concatenate(pRef_fitnesses, new_fitnesses)
        else:
            single_generation(population, fitnesses)

    winner_index = np.argmax(fitnesses)

    if return_pRef:
        return population[winner_index], fitnesses[winner_index], ContinuousPRef(pRef_population, pRef_fitnesses)
    else:
        return population[winner_index], fitnesses[winner_index]
