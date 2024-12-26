import random
from typing import Iterable, Sized

import numpy as np

from Core.custom_types import ArrayOfInts


class SearchSpace(Sized):
    cardinalities: ArrayOfInts
    precomputed_offsets: ArrayOfInts

    def __init__(self, cardinalities: Iterable[int]):
        self.cardinalities = np.fromiter(cardinalities, dtype=int)
        self.precomputed_offsets = np.concatenate(([0], np.cumsum(self.cardinalities)))

    @property
    def hot_encoded_length(self) -> int:
        return int(np.sum(self.cardinalities))

    @property
    def dimensions(self) -> int:
        return len(self.cardinalities)

    def __len__(self) -> int:
        return self.dimensions

    @property
    def amount_of_parameters(self) -> int:
        return self.dimensions

    def __repr__(self):
        return f"SearchSpace{tuple(self.cardinalities)}"

    def __eq__(self, other) -> bool:
        return np.array_equal(self.cardinalities, other.cardinalities)

    @classmethod
    def concatenate_search_spaces(cls, to_concat: Iterable):
        cardinalities = tuple(ss.cardinalities for ss in to_concat)
        return cls(np.concatenate(cardinalities))

    def random_digit(self, position: int) -> int:
        return random.randrange(self.cardinalities[position])

    @classmethod
    def from_permuation_of(cls, qty_nodes: int):
        # I don't include the last choice having cardinality 1 just to avoid
        # annoying edge cases in code
        return cls([i for i in range(qty_nodes, 1, -1)])
