from typing import Optional, Callable

import numpy as np

from CollaborationMarch.PolishSystem.PolishPSSearchTask import find_ps_in_polish_solution
from CollaborationMarch.SimplifiedSystem.GlobalPSSearchTask import find_ps_in_problem
from CollaborationMarch.SimplifiedSystem.LocalPSSearchTask import find_ps_in_solution
from CollaborationMarch.SimplifiedSystem.PSSearchSettings import PSSearchSettings
from CollaborationMarch.SimplifiedSystem.search_methods import get_unexplained_parts_of_solution
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS, STAR
from Core.SearchSpace import SearchSpace


def search_local_polish_ps(solution_to_explain: FullSolution,
                    search_settings: PSSearchSettings,
                    objectives: list[Callable],
                    to_avoid: Optional[list[PS]] = None) -> [PS]:
    unexplained_vars = get_unexplained_parts_of_solution(solution_to_explain, [] if to_avoid is None else to_avoid)
    return find_ps_in_polish_solution(to_explain=solution_to_explain,
                                      ps_budget=search_settings.ps_search_budget,
                                      culling_method=search_settings.culling_method,
                                      population_size=search_settings.ps_search_population,
                                      metrics_functions=objectives,
                                      unexplained_mask=unexplained_vars,
                                      proportion_unexplained_that_needs_used=search_settings.proportion_unexplained_that_needs_used,
                                      proportion_used_that_should_be_unexplained=search_settings.proportion_used_that_should_be_unexplained,
                                      verbose=search_settings.verbose)
