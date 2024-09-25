from typing import Optional

import numpy as np

from Core import SearchSpace
from Core.PRef import PRef
from Core.PS import PS, STAR
from Core.PSMetric.Metric import Metric
from Core.custom_types import ArrayOfFloats


class Atomicity(Metric):
    pRef: Optional[PRef]
    normalised_pRef: Optional[PRef]
    global_isolated_benefits: Optional[list[list[float]]]

    def __init__(self):
        super().__init__()
        self.pRef = None
        self.normalised_pRef = None
        self.global_isolated_benefits = None

    def set_pRef(self, pRef: PRef):
        self.pRef = pRef
        self.normalised_pRef = self.get_normalised_pRef(self.pRef)
        self.global_isolated_benefits = self.get_global_isolated_benefits()

    def __repr__(self):
        return "Atomicity"

    @staticmethod
    def get_isolated_in_search_space(search_space: SearchSpace.SearchSpace) -> list[PS]:
        empty: PS = PS.empty(search_space)
        return empty.specialisations(search_space)

    @staticmethod
    def get_normalised_pRef(pRef: PRef) -> PRef:
        min_fitness = np.min(pRef.fitness_array)
        normalised_fitnesses = pRef.fitness_array - min_fitness
        sum_fitness = np.sum(normalised_fitnesses, dtype=float)

        if sum_fitness == 0:
            raise Exception(f"The sum of fitnesses for {pRef} is 0, could not normalise")

        normalised_fitnesses /= sum_fitness

        return PRef(fitness_array=normalised_fitnesses,  # this is the only thing that changes
                    full_solution_matrix=pRef.full_solution_matrix,
                    search_space=pRef.search_space)

    def get_benefit(self, ps: PS) -> float:
        return float(np.sum(self.normalised_pRef.fitnesses_of_observations(ps)))

    def get_global_isolated_benefits(self) -> list[list[float]]:
        """Requires self.normalised_pRef"""
        ss = self.normalised_pRef.search_space
        empty: PS = PS.empty(ss)

        def benefit_when_isolating(var: int, val: int) -> float:
            isolated = empty.with_fixed_value(var, val)
            return self.get_benefit(isolated)

        return [[benefit_when_isolating(var, val)
                 for val in range(ss.cardinalities[var])]
                for var in range(ss.amount_of_parameters)]

    def get_isolated_benefits(self, ps: PS) -> ArrayOfFloats:
        return np.array([self.global_isolated_benefits[var][val]
                         for var, val in enumerate(ps.values)
                         if val != STAR])

    def get_excluded_benefits(self, ps: PS) -> ArrayOfFloats:
        exclusions = ps.simplifications()
        return np.array([self.get_benefit(excluded) for excluded in exclusions])

    def get_single_score(self, ps: PS):
        pAB = self.get_benefit(ps)
        if pAB == 0.0:
            return pAB

        isolated = self.get_isolated_benefits(ps)
        excluded = self.get_excluded_benefits(ps)

        if len(isolated) == 0:  # ie we have the empty ps
            return 0

        max_denominator = np.max(isolated * excluded)  # praying that they are always the same size!

        result = pAB * np.log(pAB / max_denominator)
        if np.isnan(result).any():
            raise Exception("There is a nan value returned in atomicity")
        return result
