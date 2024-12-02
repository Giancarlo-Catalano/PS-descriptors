import itertools
from math import floor
from typing import Optional

import numpy as np

from BenchmarkProblems.RoyalRoad import RoyalRoad
from BenchmarkProblems.Trapk import Trapk
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS, STAR
from Core.SearchSpace import SearchSpace
from Explanation.PRefManager import PRefManager


class BivariateLinkage:
    pRef: Optional[PRef]

    def __init__(self):
        self.pRef = None

    def set_pRef(self, pRef: PRef) -> None:
        self.pRef = pRef

    @property
    def search_space(self) -> SearchSpace:
        return self.pRef.search_space

    @property
    def n_vars(self) -> int:
        return self.search_space.amount_of_parameters

    def get_bivariate_linkage_between_vars(self, var_a: int, var_b: int) -> float:
        raise NotImplemented

    def get_univariate_linkage_of_var(self, var: int) -> float:
        raise NotImplemented

    def every_var_iterator(self):
        return range(self.n_vars)

    def every_var_pair_iterator(self):
        return itertools.combinations(range(self.n_vars), r=2)

    def get_atomicity(self, ps: PS) -> float:
        fixed_vars = ps.get_fixed_variable_positions()

        def weakest_internal_linkage_for(var) -> float:
            return min(self.get_bivariate_linkage_between_vars(var, other)
                       for other in fixed_vars if other != var)

        if len(fixed_vars) > 1:
            weakest_links = np.array([weakest_internal_linkage_for(var) for var in fixed_vars])
            return np.average(weakest_links)
        elif len(fixed_vars) == 1:
            var = fixed_vars[0]
            return self.get_univariate_linkage_of_var(var)
        else:
            return 0

    def get_dependence(self, ps: PS) -> float:
        fixed_vars = ps.get_fixed_variable_positions()
        unfixed_vars = [index for index, val in enumerate(ps.values) if val == STAR]

        def strongest_external_linkage_for(var) -> float:
            return max(self.get_bivariate_linkage_between_vars(var, other)
                       for other in unfixed_vars)

        if (len(unfixed_vars) > 0) and (len(fixed_vars) > 0):  # maybe this should be zero?
            strongest_links = np.array([strongest_external_linkage_for(var) for var in fixed_vars])
            return np.average(strongest_links)
        elif len(fixed_vars) == 1:
            var = fixed_vars[0]
            return strongest_external_linkage_for(var)
        else:
            return 0