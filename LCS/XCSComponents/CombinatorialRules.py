import random
from typing import Iterable, Optional, Any

import numpy as np
from xcs import ClassifierRule

from Core.FullSolution import FullSolution
from Core.PS import PS, STAR, contains


class CombinatorialString(FullSolution):

    def __init__(self, values):
        super().__init__(values)

class CombinatorialCondition(PS):
    def __init__(self, original_values: Iterable[int], mask: Iterable[bool]):
        # I don't handle all the other cases
        ps_values = [original if is_fixed else STAR
                     for (original, is_fixed) in zip(original_values, mask)]
        super().__init__(ps_values)


    @classmethod
    def from_ps_values(cls, ps_values):
        # takes an illegal shortcut
        to_return = cls(original_values=[], mask = [])
        to_return.values = ps_values
        return to_return


    @classmethod
    def cover(cls, original_values: Iterable[int], wildcard_probability):
        mask = np.random.random(size=len(original_values)) > wildcard_probability
        return cls(original_values, mask)

    @property
    def bits(self):
        return self.values  # I'll leave it like this for now, just in case

    @property
    def mask(self):
        return self.values != STAR


    def count(self):
        return self.fixed_count()


    def copy(self):
        return CombinatorialCondition.from_ps_values(self.values.copy())


    def __iter__(self):
        for value in self.values:
            if value == STAR:
                yield None
            else:
                yield value

    def __getitem__(self, index):
        return self.values[index]

    # hash is already implemented

    # eq is already implemented

    # ne should be implemented automatically


    def is_generalisation_of(self, other_values) -> bool:
        same_value = self.values == other_values
        i_have_wildcard = self.values == STAR
        # (own, other) A A -> True,  A B -> False, A * -> False, * A -> True, * * -> True
        return np.all(same_value | i_have_wildcard)


    def __floordiv__(self, other):
        same_value = self.values == other.values
        wildcard_in_either = (self.values == STAR) | (other.values == STAR)
        return np.logical_or(same_value,
                             wildcard_in_either)


    def __call__(self, other):  # other might be a np.ndarray or another CombinatorialCondition
        def check_against_values(other_values: np.ndarray) -> bool:
            same_value = self.values == other_values
            wildcard_in_either = (self.values == STAR) | (other_values == STAR)
            return np.all(np.logical_or(same_value,
                                 wildcard_in_either))

        if isinstance(other, np.ndarray):
            return check_against_values(other)
        elif isinstance(other, CombinatorialCondition):
            return self.is_generalisation_of(other.values)
        elif type(other) is tuple and len(other) == 2 and isinstance(other[0], FullSolution):
            # this is a nasty hack to circumvent the restrictions in ActionSet for the condition to match the scenario
            winner, loser = other
            return self.is_generalisation_of(winner.values) != self.is_generalisation_of(loser.values) # xor
        elif isinstance(other, FullSolution):
            return check_against_values(other.values)
        else:
            print(f"Received a __call__ where type(other) = {type(other)}")
            return check_against_values(np.array(other))



    def crossover_with(self, other, points = 2) -> (Any, Any):
        # I'd rather use uniform binary crossover, thank you
        daughter_values = self.values.copy()
        son_values = other.values.copy()

        for index in range(len(self)):
            if random.random() < 0.5:
                daughter_values[index], son_values[index] = son_values[index], son_values[index]

        daughter = CombinatorialCondition.from_ps_values(daughter_values)
        son = CombinatorialCondition.from_ps_values(son_values)
        return daughter, son


    def mutate(self, point_mutation_probability: float, background_solution: FullSolution):
        # Modification of XCSAlgorithm._mutate

        for index, old_value in enumerate(self.values):
            if random.random() < point_mutation_probability:
                if old_value == STAR:
                    self.values[index] = background_solution.values[index]
                else:
                    self.values[index] = STAR


    def matches_partial_solution(self, ps: PS) -> bool:
        """Return true if self and ps don't have disagreements"""
        """ While i am aware that this implementation is identical to that of matches_solution, but these might diverge in the future"""
        #  * * -> true,  * a -> true, a a -> true,  a b false

        for own_value, other_value in zip(self.values, ps.values):
            if own_value != STAR and other_value != STAR and own_value != other_value:
                return False
        return True






