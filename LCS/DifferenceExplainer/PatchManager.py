import random
from typing import TypeAlias

import xcs

from Core.EvaluatedFS import EvaluatedFS
from Core.FullSolution import FullSolution
from Core.PS import PS, STAR
from Core.SearchSpace import SearchSpace
from LCS.LCSManager import LCSManager

IncompleteSolution: TypeAlias = PS


class PatchManager:
    """This class will allow to edit incomplete solutions using different methods"""


    merge_limit: int
    search_space: SearchSpace
    lcs_manager: LCSManager

    def __init__(self,
                 lcs_manager: LCSManager,
                 search_space: SearchSpace,
                 merge_limit: int = 12):
        self.lcs_manager = lcs_manager
        self.search_space = search_space
        self.merge_limit = merge_limit





    def fix_solution(self, incomplete_solution: IncompleteSolution, method: str) -> FullSolution:
        match method:
            case "random": return self.fix_via_random(incomplete_solution)
            case "P&M": return self.fix_via_pick_and_merge(incomplete_solution)
            case "GA": return self.fix_via_GA(incomplete_solution)

    def fix_via_random(self, incomplete_solution: IncompleteSolution) -> FullSolution:
        new_values = incomplete_solution.values.copy()
        for index, value in enumerate(incomplete_solution.values):
            if value == STAR:
                new_values[index] = self.search_space.random_digit(position=index)

        solution = FullSolution(new_values)
        return solution


    def fix_via_pick_and_merge_unsafe(self, incomplete_solution: IncompleteSolution) -> IncompleteSolution:
        possible_patches = self.lcs_manager.get_matches_with_partial_solution(partial_solution = incomplete_solution)




        def select_and_pop(patches_left: list[xcs.XCSClassifierRule]) -> xcs.XCSClassifierRule:
            to_return = random.choice(patches_left)  #this should be weighted in some way... tournament selection on accuracy
            patches_left.remove(to_return)
            return to_return


        usable_patches = list(set(possible_patches))  # I would make it a set but i need to call random.choice
        current_result = incomplete_solution.copy()

        def apply_patch(patch: xcs.XCSClassifierRule) -> None:
            where_not_star = patch.condition.values != STAR
            current_result.values[where_not_star] = patch.condition.values[where_not_star]

        def candidate_is_compatible(patch: xcs.XCSClassifierRule) -> bool:
            return patch.condition.matches_partial_solution(current_result)



        merged = 0
        while (merged < self.merge_limit) and \
                (not current_result.is_fully_fixed()) and \
                len(usable_patches) > 0:
            candidate = select_and_pop(usable_patches)
            if candidate_is_compatible(candidate):
                apply_patch(candidate)


        return current_result

    def fix_via_pick_and_merge(self, incomplete_solution: IncompleteSolution) -> FullSolution:
        possibly_complete = self.fix_via_pick_and_merge_unsafe(incomplete_solution)
        # sometimes possibly complete will not need to have its gaps filled, and in those cases fix_via_random does nothing
        return self.fix_via_random(possibly_complete)




    @classmethod
    def remove_random_subset_from_solution(cls, solution: EvaluatedFS, amount_to_remove: int) -> PS:
        variables_to_remove = random.sample(range(len(solution)), amount_to_remove)
        new_values = solution.values.copy()
        new_values[variables_to_remove] = STAR
        return PS(new_values)



