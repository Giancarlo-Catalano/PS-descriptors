import itertools
import random
from typing import Iterable

import numpy as np

from Core.FullSolution import FullSolution
from Core.SearchSpace import SearchSpace
from Core.custom_types import ArrayOfInts

STAR = -1


class PS:
    values: ArrayOfInts

    def __init__(self, values: Iterable[int]):
        self.values = np.fromiter(values, dtype=int)

    def __len__(self):
        return len(self.values)

    def __repr__(self) -> str:
        def repr_single(cell_value: int) -> str:
            return f'{cell_value}' if cell_value != STAR else '*'

        return "[" + " ".join(map(repr_single, self.values)) + "]"

    def __hash__(self):
        # NOTE: using
        return hash(tuple(self.values))
        # also works well.
        # alternative is hash(np.bitwise_xor.reduce(self.values))

    @classmethod
    def empty(cls, search_space: SearchSpace):
        values = np.full(search_space.amount_of_parameters, STAR)
        return cls(values)

    @classmethod
    def random(cls, search_space: SearchSpace, half_chance_star=True):
        def random_value_with_half_chance(cardinality):
            return STAR if random.random() < 0.5 else random.randrange(cardinality)

        def random_value_with_uniform_chance(cardinality):
            # THIS STATEMENT DEPENDS ON THE IMPLEMENTATION OF STAR
            return random.randrange(cardinality + 1) - 1

        # and this is very dodgy...
        value_generator = random_value_with_half_chance if half_chance_star else random_value_with_uniform_chance
        return PS(value_generator(cardinality) for cardinality in search_space.cardinalities)


    @classmethod
    def random_with_fixed_size(cls, search_space: SearchSpace, size: int):
        if size > search_space.amount_of_parameters:
            raise ValueError(f"Trying to obtain a PS with size {size} from search space {search_space}")

        values = [STAR for _ in search_space.cardinalities]
        fixed_vars = random.sample(list(range(search_space.amount_of_parameters)), k=size)
        for fixed_var in fixed_vars:
            new_value = random.randrange(search_space.cardinalities[fixed_var])
            values[fixed_var] = new_value
        return PS(values)

    def is_fully_fixed(self) -> bool:
        return np.all(self.values != STAR)

    def to_FS(self) -> FullSolution:
        return FullSolution(self.values)

    @classmethod
    def from_FS(cls, fs: FullSolution):
        return cls(fs.values)

    def with_unfixed_value(self, variable_position: int):
        new_values = np.copy(self.values)
        new_values[variable_position] = STAR
        return PS(new_values)

    def with_fixed_value(self, variable_position: int, fixed_value: int):
        new_values = np.copy(self.values)
        new_values[variable_position] = fixed_value
        return PS(new_values)

    def get_fixed_variable_positions(self) -> list[int]:
        return [position for position, value in enumerate(self.values) if value != STAR]

    def get_unfixed_variable_positions(self) -> list[int]:
        return [position for position, value in enumerate(self.values) if value == STAR]

    def simplifications(self):
        return [self.with_unfixed_value(i) for i in self.get_fixed_variable_positions()]

    def specialisations(self, search_space: SearchSpace):
        return [self.with_fixed_value(position, value)
                for position in self.get_unfixed_variable_positions()
                for value in range(search_space.cardinalities[position])]

    def __eq__(self, other) -> bool:
        """ This had to be optimised"""
        return np.array_equal(self.values, other.values)

    @classmethod
    def mergeable(cls, a, b) -> bool:
        """In the places where both PSs have fixed params, the values are the same"""
        for v_a, v_b in zip(a.values, b.values):
            if (v_a != STAR and v_b != STAR) and (v_a != v_b):
                return False
        return True

    @classmethod
    def merge(cls, a, b):
        """assumes that mergeable(a, b) == True"""
        new_values = np.max((a.values, b.values), axis=0)
        return cls(new_values)

    def present_in(self, full_solution: FullSolution) -> bool:
        same_fixed_value = self.values == full_solution.values
        is_unfixed = self.values == STAR
        matching_cells = np.logical_or(same_fixed_value, is_unfixed)
        return np.all(matching_cells)

    def is_empty(self) -> bool:
        return np.all(self.values == STAR)

    def fixed_count(self) -> int:
        return int(np.sum(self.values != STAR, dtype=int))

    def copy(self):
        return PS(self.values)

    @classmethod
    def all_possible(cls, search_space: SearchSpace):
        levels = (range(-1, c) for c in search_space.cardinalities)
        values = itertools.product(*levels)
        return [cls(values) for values in values]


    def __getitem__(self, item) -> int:
        return self.values[item]

    def __setitem__(self, key, value):
        self.values[key] = value


    @classmethod
    def from_string(cls, string: str):
        # first remove any spaces
        without_spaces = "".join(string.split())
        def char_to_value(char: str) -> int:
            if char == "*":
                return STAR
            else:
                return int(char)
        return cls(map(char_to_value, without_spaces))


    def get_hamming_distance(self, other) -> int:
        return int(np.sum(self.values != other.values))

    def get_jaccard_distance(self, other) -> float:
        fixed_a = self.values != STAR
        fixed_b = other.values != STAR
        intersection = np.sum(fixed_a & fixed_b)
        union = np.sum(fixed_a | fixed_b)

        return float(intersection / union)

    def to_json(self):
        return {"values": self.values.tolist()}

    @classmethod
    def from_json(cls, json_dict: dict):
        return cls(json_dict["values"])

def contains(fs: FullSolution, ps: PS) -> bool:
    return all(x_psi_i in {STAR, x_i} for x_psi_i, x_i in zip(ps.values, fs.values))



