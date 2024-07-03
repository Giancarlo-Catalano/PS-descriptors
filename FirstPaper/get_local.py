import random
from typing import Callable

import numpy as np

from FirstPaper.PS import PS, STAR
from FirstPaper.SearchSpace import SearchSpace


def specialisations(ps: PS, search_space: SearchSpace) -> list[PS]:
    return ps.specialisations(search_space)


def simplifications(ps: PS, search_space: SearchSpace) -> list[PS]:
    return ps.simplifications()


def entire_neighbourhood(ps: PS, search_space: SearchSpace) -> list[PS]:
    return specialisations(ps, search_space) + simplifications(ps, search_space)


def mutated(ps: PS, search_space: SearchSpace, chance_of_mutation: float) -> PS:
    def random_value(var_index: int) -> int:
        return random.randrange(search_space.cardinalities[var_index])

    def get_mutated_value_for(index: int):
        if ps.values[index] == STAR:
            return random_value(index)
        else:
            if random.random() < 0.5:
                return STAR
            else:
                return random_value(index)

    new_values = np.copy(ps.values)
    for index in range(len(new_values)):
        if random.random() < chance_of_mutation:
            new_values[index] = get_mutated_value_for(index)

    return PS(new_values)


def make_mutation_operator(mutation_chance: float) -> Callable:
    def mutation_closure(ps: PS, search_space: SearchSpace):
        return mutated(ps, search_space, mutation_chance)

    return mutation_closure
