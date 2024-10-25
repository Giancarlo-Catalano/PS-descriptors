import random
from typing import Iterable

import numpy as np

from Core.SearchSpace import SearchSpace
from Core.custom_types import ArrayOfInts


class FullSolution:
    # a wrapper for a tuple
    values: ArrayOfInts

    def __init__(self, values: Iterable[int]):
        self.values = np.fromiter(values, dtype=int)
        self.values.setflags(write=False)

    def __repr__(self):
        return "(" + (" ".join([f"{val}" for val in self.values])) + ")"

    def __eq__(self, other):
        return np.array_equal(self.values, other.values)

    def __hash__(self):
        return hash(tuple(self.values))

    def __len__(self):
        return len(self.values)

    @classmethod
    def random(cls, search_space: SearchSpace):
        return cls(random.randrange(card) for card in search_space.cardinalities)

    def with_different_value(self, variable_index: int, new_value: int):
        new_values = self.values.copy()
        new_values[variable_index] = new_value
        return FullSolution(new_values)

    def copy(self):
        return FullSolution(self.values)

    def get_hamming_distance(self, other) -> int:
        return int(np.sum(self.values != other.values))

    def to_json(self):
        return {"values": self.values.tolist()}

    @classmethod
    def from_json(cls, json_dict: dict):
        return cls(json_dict["values"])