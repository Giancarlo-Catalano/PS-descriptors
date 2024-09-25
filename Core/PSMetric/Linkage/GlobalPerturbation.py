import itertools
from typing import TypeAlias, Optional

import numpy as np

import utils
from Core.PRef import PRef
from Core.PS import PS, STAR
from Core.PSMetric.Linkage.OutdatedLinkage import OutdatedLinkage
from Core.PSMetric.Linkage.OutdatedLinkage import BivariateLocalPerturbation, UnivariateLocalPerturbation
from Core.PSMetric.Metric import Metric
from Core.custom_types import ArrayOfFloats

ImportanceArray: TypeAlias = np.ndarray
LinkageTable: TypeAlias = np.ndarray


class UnivariateGlobalPerturbation(Metric):
    importance_array: Optional[ImportanceArray]
    normalised_importance_array: Optional[ImportanceArray]

    def __init__(self):
        self.importance_array = None
        self.normalised_importance_array = None
        super().__init__()

    def __repr__(self):
        return "UnivariateGlobalLinkage"

    @staticmethod
    def get_importance_array(pRef: PRef) -> ImportanceArray:
        levels = [list(range(cardinality)) for cardinality in pRef.search_space.cardinalities]

        def get_mean_fitness_for_each_val(locus: int) -> ArrayOfFloats:
            def mean_fit(val):
                return np.mean(pRef.get_fitnesses_matching_var_val(locus, val))

            return np.array([mean_fit(val) for val in levels[locus]])

        def get_variance_in_var(locus: int) -> float:
            return float(np.var(get_mean_fitness_for_each_val(locus)))

        return np.array([get_variance_in_var(i) for i in range(pRef.search_space.amount_of_parameters)])

    @staticmethod
    def get_normalised_importance_array(importance_array: ImportanceArray) -> ImportanceArray:
        return utils.remap_array_in_zero_one(importance_array)

    def set_pRef(self, pRef: PRef):
        self.importance_array = self.get_importance_array(pRef)
        self.normalised_importance_array = self.get_normalised_importance_array(self.importance_array)

    def get_single_normalised_score(self, ps: PS) -> float:
        return np.min(self.normalised_importance_array, where=ps.values != STAR, initial=1)

    def get_single_score(self, ps: PS) -> float:
        return self.get_single_normalised_score(ps)


class BivariateGlobalPerturbation(Metric):
    linkage_table: Optional[ImportanceArray]
    normalised_linkage_table: Optional[ImportanceArray]

    def __init__(self):
        self.linkage_table = None
        self.normalised_linkage_table = None
        super().__init__()

    def __repr__(self):
        return "BivariateGlobalLinkage"

    @staticmethod
    def get_linkage_table(pRef: PRef) -> ImportanceArray:

        levels = [list(range(cardinality)) for cardinality in pRef.search_space.cardinalities]

        def get_mean_fitness_for_each_combination(locus_a: int, locus_b: int) -> ArrayOfFloats:
            def mean_fit(val_a, val_b):
                return np.mean(pRef.get_fitnesses_matching_var_val_pair(locus_a, val_a, locus_b, val_b))

            return np.array([mean_fit(val_a, val_b)
                             for val_a in levels[locus_a]
                             for val_b in levels[locus_b]])

        def get_variance_in_loci(locus_a: int, locus_b: int) -> float:
            return float(np.var(get_mean_fitness_for_each_combination(locus_a, locus_b)))

        linkage_table = np.zeros((pRef.search_space.amount_of_parameters, pRef.search_space.amount_of_parameters))
        for var_a in range(pRef.search_space.amount_of_parameters):
            for var_b in range(var_a + 1, pRef.search_space.amount_of_parameters):
                linkage_table[var_a][var_b] = get_variance_in_loci(var_a, var_b)

        univariate_variances = UnivariateGlobalPerturbation.get_importance_array(pRef)  # for the diagonal
        np.fill_diagonal(linkage_table, univariate_variances)
        # then we mirror it for convenience...
        upper_triangle = np.triu(linkage_table, k=1)
        linkage_table = linkage_table + upper_triangle.T
        return linkage_table

    def set_pRef(self, pRef: PRef):
        self.linkage_table = self.get_linkage_table(pRef)
        self.normalised_linkage_table = OutdatedLinkage.get_normalised_linkage_table(self.linkage_table, include_diagonal=True)

    def get_all_normalised_linkages(self, ps: PS, include_reflexive=False) -> list[float]:
        if include_reflexive:
            pairs = itertools.combinations_with_replacement(ps.get_fixed_variable_positions(), r=2)
        else:
            pairs = itertools.combinations(ps.get_fixed_variable_positions(), r=2)

        return [self.normalised_linkage_table[pair] for pair in pairs]

    def get_single_normalised_score(self, ps: PS) -> float:
        return np.min(self.get_all_normalised_linkages(ps, include_reflexive=True))


    def get_single_score(self, ps: PS) -> float:
        return self.get_single_normalised_score(ps)


class AlternativeBivariateGlobalLinkage(Metric):
    linkage_table: Optional[ImportanceArray]
    normalised_linkage_table: Optional[ImportanceArray]

    def __init__(self):
        self.linkage_table = None
        self.normalised_linkage_table = None
        super().__init__()

    def __repr__(self):
        return "AlternativeBivariateGlobalLinkage"

    @staticmethod
    def get_linkage_table(pRef: PRef) -> ImportanceArray:

        levels = [list(range(cardinality)) for cardinality in pRef.search_space.cardinalities]
        blp = BivariateLocalPerturbation()
        ulp = UnivariateLocalPerturbation()
        blp.set_pRef(pRef)
        ulp.set_pRef(pRef)

        def get_mean_fitness_for_each_combination(locus_a: int, locus_b: int) -> ArrayOfFloats:
            def mean_effect(val_a, val_b):
                ps = PS.empty(pRef.search_space).with_fixed_value(locus_a, val_a).with_fixed_value(locus_b, val_b)
                return blp.get_single_score(ps)

            return np.array([mean_effect(val_a, val_b)
                             for val_a in levels[locus_a]
                             for val_b in levels[locus_b]])

        def get_interaction_in_loci(locus_a: int, locus_b: int) -> float:
            return float(np.average(get_mean_fitness_for_each_combination(locus_a, locus_b)))

        def get_interaction_in_locus(locus) -> float:
            return float(np.average([ulp.get_single_score(PS.empty(pRef.search_space).with_fixed_value(locus, val))
                                     for val in levels[locus]]))

        linkage_table = np.zeros((pRef.search_space.amount_of_parameters, pRef.search_space.amount_of_parameters))
        for var_a in range(pRef.search_space.amount_of_parameters):
            for var_b in range(var_a, pRef.search_space.amount_of_parameters):
                if var_a == var_b:
                    linkage_table[var_a][var_b] = get_interaction_in_locus(var_a)
                else:
                    linkage_table[var_a][var_b] = get_interaction_in_loci(var_a, var_b)

        # then we mirror it for convenience...
        upper_triangle = np.triu(linkage_table, k=1)
        linkage_table = linkage_table + upper_triangle.T
        return linkage_table

    def set_pRef(self, pRef: PRef):
        self.linkage_table = self.get_linkage_table(pRef)
        self.normalised_linkage_table = OutdatedLinkage.get_normalised_linkage_table(self.linkage_table, include_diagonal=True)

    def get_all_normalised_linkages(self, ps: PS, include_reflexive=False) -> list[float]:
        if include_reflexive:
            pairs = itertools.combinations_with_replacement(ps.get_fixed_variable_positions(), r=2)
        else:
            pairs = itertools.combinations(ps.get_fixed_variable_positions(), r=2)

        return [self.normalised_linkage_table[pair] for pair in pairs]

    def get_single_normalised_score(self, ps: PS) -> float:
        return np.min(self.get_all_normalised_linkages(ps, include_reflexive=False))
