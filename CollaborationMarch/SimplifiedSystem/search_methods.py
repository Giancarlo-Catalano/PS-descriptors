from typing import Optional

import numpy as np

from CollaborationMarch.SimplifiedSystem.GlobalPSSearchTask import find_ps_in_problem
from CollaborationMarch.SimplifiedSystem.LocalPSSearchTask import find_ps_in_solution
from CollaborationMarch.SimplifiedSystem.PSSearchSettings import PSSearchSettings
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS, STAR
from Core.SearchSpace import SearchSpace


def get_unexplained_parts_of_solution(solution: FullSolution, partial_solutions: list[PS]) -> np.ndarray:
    if len(partial_solutions) == 0:
        return np.ones(shape=len(solution), dtype=bool)
    ps_matrix = np.array([ps.values for ps in partial_solutions])
    return np.all(ps_matrix == STAR, axis=0)


def get_unexplained_pairs_of_search_space(search_space: SearchSpace, partial_solutions: list[PS]) -> np.ndarray:
    if len(partial_solutions) == 0:
        return np.ones(shape=search_space.amount_of_parameters, dtype=bool)
    ps_matrix = np.array([ps.values for ps in partial_solutions])
    return np.all(ps_matrix == STAR, axis=0)


def search_local_ps(solution_to_explain: FullSolution,
                    backgroundInformation: PRef,
                    search_settings: PSSearchSettings,
                    to_avoid: Optional[list[PS]] = None) -> [PS]:
    unexplained_vars = get_unexplained_parts_of_solution(solution_to_explain, [] if to_avoid is None else to_avoid)
    return find_ps_in_solution(pRef=backgroundInformation,
                               ps_budget=search_settings.ps_search_budget,
                               culling_method=search_settings.culling_method,
                               population_size=search_settings.ps_search_population,
                               to_explain=solution_to_explain,
                               unexplained_mask=unexplained_vars,
                               proportion_unexplained_that_needs_used=search_settings.proportion_unexplained_that_needs_used,
                               proportion_used_that_should_be_unexplained=search_settings.proportion_used_that_should_be_unexplained,
                               problem=search_settings.original_problem,
                               metrics=search_settings.metrics,
                               verbose=search_settings.verbose)


def search_global_ps(original_problem_search_space: SearchSpace,
                     backgroundInformation: PRef,
                     search_settings: PSSearchSettings,
                     to_avoid: Optional[list[PS]] = None) -> [PS]:
    unexplained_vars = get_unexplained_pairs_of_search_space(original_problem_search_space,
                                                             [] if to_avoid is None else to_avoid)
    return find_ps_in_problem(original_problem_search_space=original_problem_search_space,
                              pRef=backgroundInformation,
                              ps_budget=search_settings.ps_search_budget,
                              culling_method=search_settings.culling_method,
                              population_size=search_settings.ps_search_population,
                              unexplained_mask=unexplained_vars,
                              proportion_unexplained_that_needs_used=search_settings.proportion_unexplained_that_needs_used,
                              proportion_used_that_should_be_unexplained=search_settings.proportion_used_that_should_be_unexplained,
                              problem=search_settings.original_problem,
                              metrics=search_settings.metrics,
                              verbose=search_settings.verbose)
