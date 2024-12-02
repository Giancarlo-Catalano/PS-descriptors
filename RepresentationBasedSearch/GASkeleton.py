import copy
import heapq
import random
from typing import Optional, Callable, Any

import utils


def tournament_select(population,
                      tournament_size: int) -> list:
    pool = [random.choice(population) for _ in range(tournament_size)]
    return max(pool, key=utils.second)


def get_tournament_operator(tournament_size: int = 3):
    return lambda population: tournament_select(population, tournament_size)


def simple_ga(fitness_function: Callable,
              make_random_solution: Callable,
              make_mutated: Callable,
              make_crossovered: Optional[Callable] = None,
              initial_population: Optional[list] = None,
              selection_method: Optional[Callable] = None,
              population_size: Optional[int] = None,
              evaluation_budget: int = 10000,
              ):
    selection_method = get_tournament_operator() if selection_method is None else selection_method

    population = copy.deepcopy(initial_population) \
        if initial_population is not None \
        else [make_random_solution() for _ in range(population_size)]

    used_budget = [0]  # nasty hack to make the next function easier

    def make_evaluated(individual) -> (Any, float):
        used_budget[0] += 1
        return (individual, fitness_function(individual))

    population = [make_evaluated(individual) for individual in population]

    def make_new_children() -> [Any]:
        if make_crossovered is None:
            mother = selection_method(population)
            return make_mutated(mother)
        else:
            mother = selection_method(population)
            father = selection_method(population)
            daughter, son = make_crossovered(mother, father)
            return [make_mutated(daughter), make_mutated(son)]

    def single_iteration(population):
        # make a child for every element in the population
        children = []
        while len(children) < population_size:
            children.extend(make_new_children())

        # evaluate the children
        children = [make_evaluated(child) for child in children]
        population.extend(children)

        # truncation selection
        population = heapq.nlargest(iterable = population, n = population_size, key=utils.second)

    while used_budget[0] < evaluation_budget:
        single_iteration(population)

    return max(population, key=utils.second)[0]


