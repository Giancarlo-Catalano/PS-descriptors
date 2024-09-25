import random

import numpy as np

from Core.PS import STAR
from Core.SearchSpace import SearchSpace
from LCS.XCSComponents.CombinatorialRules import CombinatorialCondition

def uniform_crossover_on_combinatorial_condition(mother: CombinatorialCondition,
                                                 father: CombinatorialCondition) -> (CombinatorialCondition, CombinatorialCondition):
    assert(len(mother) == len(father))

    # what is even going on in the original function!!!

    kids_matrix = np.array([mother.values.copy(), father.values.copy()])
    for variable in range(len(mother)):
        if random.random() < 0.5:
            kids_matrix[:, variable] = np.roll(kids_matrix[:, variable], shift=1)

    daughter = CombinatorialCondition(kids_matrix[0])
    son = CombinatorialCondition(kids_matrix[1])
    return daughter, son


def uniform_mutation_on_combinatorial_condition(condition: CombinatorialCondition,
                                                mutation_chance: float,
                                                search_space: SearchSpace) -> CombinatorialCondition:
    result_values = condition.values.copy()
    for index, original_value in enumerate(condition.values):
        if random.random() < mutation_chance:
            if original_value == STAR:
                result_values[index] = random.randrange(search_space.cardinalities[index])
            else:
                result_values[index] = STAR