import random
from math import ceil, sqrt
from typing import Iterable, Optional

import numpy as np

from Core.EvaluatedPS import EvaluatedPS
from Core.PS import PS, STAR
from Core.FullSolution import FullSolution
from Core.SearchSpace import SearchSpace


class PickAndMergeSampler:
    search_space: SearchSpace
    individuals: set[EvaluatedPS]
    merge_limit: int

    tournament_size = 3

    def __init__(self,
                 search_space: SearchSpace,
                 individuals: Iterable[EvaluatedPS],
                 merge_limit: Optional[int] = None):
        self.search_space = search_space
        self.individuals = set(individuals)

        if merge_limit is None:
            merge_limit = ceil(sqrt(search_space.amount_of_parameters))
        self.merge_limit = merge_limit

    def sample_ps_unsafe(self, starting_point: Optional[PS] = None) -> PS:
        """ this is unsafe in the sense that the result might not be complete"""

        available = set(self.individuals)

        def pick() -> EvaluatedPS:
            return max(random.choices(list(available), k=self.tournament_size))

        current = PS.empty(self.search_space) if starting_point is None else starting_point
        added_count = 0

        while (len(available) > 0) and (added_count < self.merge_limit) and not current.is_fully_fixed():
            to_add = pick()
            if PS.mergeable(current, to_add):
                current = PS.merge(current, to_add)
                added_count += 1
            available.remove(to_add)

        return current  # it might be incomplete!!

    def fill_in_the_gaps(self, incomplete_ps: PS):
        result_values = np.array(incomplete_ps.values)
        for index in incomplete_ps.get_unfixed_variable_positions():
            result_values[index] = random.randrange(self.search_space.cardinalities[index])
        return PS(result_values)


    def fill_in_the_gaps_avoid_previous(self, incomplete_ps: PS, previous_fs: FullSolution):
        def pick_random_value_for_var_thats_different_from_previous(index: int) -> int:
            allowed_values = list(range(self.search_space.cardinalities[index]))
            if len(allowed_values) == 0:
                raise Exception(f"While attempting to path the solution, var #{index} could not be filled, probabily because it has cardinality of 1")
            old_value = previous_fs.values[index]
            allowed_values.remove(old_value)
            return random.choice(allowed_values)
        result_values = np.array(incomplete_ps.values)
        for index in incomplete_ps.get_unfixed_variable_positions():
            result_values[index] = pick_random_value_for_var_thats_different_from_previous(index)

        return PS(result_values)

    def sample(self) -> FullSolution:
        produced_ps = self.sample_ps_unsafe()
        filled_ps = self.fill_in_the_gaps(produced_ps)
        return filled_ps.to_FS()


    def apply_patches(self, starting_point: PS, original_FS: FullSolution) -> FullSolution:
        produced_ps = self.sample_ps_unsafe(starting_point=starting_point)
        filled_ps = self.fill_in_the_gaps_avoid_previous(produced_ps, original_FS)
        return filled_ps.to_FS()

def test_pick_and_merge():
    amount_of_groups = 3
    size_of_groups = 5
    amount_of_variables = amount_of_groups * size_of_groups
    search_space = SearchSpace([2 for _ in range(amount_of_variables)])

    print(f"The search space is {search_space}")

    def group_starting_from(value: int, index: int, score: float) -> EvaluatedPS:
        result_values = np.full(amount_of_variables, STAR)
        result_values[index:(index + size_of_groups)] = value
        result = EvaluatedPS(PS(result_values))
        result.aggregated_score = score
        return result

    good_disjoint_groups = [group_starting_from(1, which_group * size_of_groups, 1)
                            for which_group in range(amount_of_groups)]

    meh_disjoint_groups = [group_starting_from(0, which_group * size_of_groups, 0.5)
                           for which_group in range(amount_of_groups)]

    annoying_groups = [group_starting_from(starting_index % 2, starting_index * 2, 0.7)
                       for starting_index in range(amount_of_variables // 2)]

    def test_with_basis(basis: list[EvaluatedPS]):
        sampler = PickAndMergeSampler(search_space, basis, merge_limit=ceil(sqrt(search_space.amount_of_parameters)))
        print("Initialised a sampler with the following partial solutions:")
        for individual in basis:
            print(individual)

        print("\n")
        for _ in range(12):
            print(sampler.sample())

    test_with_basis(good_disjoint_groups)

    test_with_basis(meh_disjoint_groups)

    test_with_basis(annoying_groups)
