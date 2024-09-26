import itertools
import warnings
from typing import Optional, TypeAlias

import numpy as np

from Core.PRef import PRef
from Core.PS import PS, STAR
from Core.PSMetric.Metric import Metric
from Core.custom_types import ArrayOfBools, ArrayOfFloats

LinkageTable: TypeAlias = np.ndarray

class LocalPerturbationCalculator:
    pRef: PRef
    cached_value_locations: list[list[ArrayOfBools]]

    def __init__(self, pRef: PRef):
        self.pRef = pRef
        self.cached_value_locations = self.get_cached_value_locations(pRef)

    @staticmethod
    def get_cached_value_locations(pRef: PRef) -> list[list[ArrayOfBools]]:
        def get_where_var_val(var: int, val: int) -> ArrayOfBools:
            return pRef.full_solution_matrix[:, var] == val

        return [[get_where_var_val(var, val)
                 for val in range(cardinality)]
                for var, cardinality in enumerate(pRef.search_space.cardinalities)]

    def get_univariate_perturbation_fitnesses(self, ps: PS, locus: int) -> (ArrayOfFloats, ArrayOfFloats):
        """ The name is horrible, but essentially it returns
           (fitnesses of observations of ps,  fitnesses of observations which match ps but at locus it DOESN't match"""

        assert (ps.values[locus] != STAR)

        where_ps_matches_ignoring_locus = np.full(shape=self.pRef.sample_size, fill_value=True, dtype=bool)
        for var, val in enumerate(ps.values):
            if val != STAR and var != locus:
                where_ps_matches_ignoring_locus = np.logical_and(where_ps_matches_ignoring_locus,
                                                                 self.cached_value_locations[var][val])

        locus_val = ps.values[locus]

        where_locus = self.cached_value_locations[locus][locus_val]
        where_value_matches = np.logical_and(where_ps_matches_ignoring_locus, where_locus)
        where_complement_matches = np.logical_and(where_ps_matches_ignoring_locus, np.logical_not(where_locus))

        return (self.pRef.fitness_array[where_value_matches], self.pRef.fitness_array[where_complement_matches])

    def get_bivariate_perturbation_fitnesses(self, ps: PS, locus_a: int, locus_b) -> (ArrayOfFloats, ArrayOfFloats):
        """ returns the fitnesses of x(a, b), x(not a, b), x(a, not b), x(not a, not b)"""

        assert (ps.values[locus_a] != STAR)
        assert (ps.values[locus_b] != STAR)

        where_ps_matches_ignoring_loci = np.full(shape=self.pRef.sample_size, fill_value=True, dtype=bool)
        for var, val in enumerate(ps.values):
            if val != STAR and var != locus_a and var != locus_b:
                where_ps_matches_ignoring_loci = np.logical_and(where_ps_matches_ignoring_loci,
                                                                self.cached_value_locations[var][val])

        val_a = ps.values[locus_a]
        val_b = ps.values[locus_b]

        where_a = self.cached_value_locations[locus_a][val_a]
        where_b = self.cached_value_locations[locus_b][val_b]
        where_not_a = np.logical_not(where_a)
        where_not_b = np.logical_not(where_b)

        where_a_b = np.logical_and(where_a, where_b)
        where_not_a_b = np.logical_and(where_not_a, where_b)
        where_a_not_b = np.logical_and(where_a, where_not_b)
        where_not_a_not_b = np.logical_and(where_not_a, where_not_b)

        def fits(where_condition: ArrayOfBools):
            return self.pRef.fitness_array[np.logical_and(where_ps_matches_ignoring_loci, where_condition)]

        return fits(where_a_b), fits(where_not_a_b), fits(where_a_not_b), fits(where_not_a_not_b)

    def get_delta_f_of_ps_at_locus_univariate(self, ps: PS, locus: int) -> float:
        value_matches, complement_matches = self.get_univariate_perturbation_fitnesses(ps, locus)

        if len(value_matches) == 0 or len(complement_matches) == 0:
            warnings.warn(
                f"Encountered a PS with insufficient observations when calculating Univariate Local perturbation")
            return 0  # panic

        fs_y = np.average(value_matches)
        fs_n = np.average(complement_matches)
        return abs(fs_y - fs_n)

    def get_delta_f_of_ps_at_loci_bivariate(self, ps: PS, locus_a: int, locus_b: int) -> float:
        fs = self.get_bivariate_perturbation_fitnesses(ps, locus_a, locus_b)
        fs_yy, fs_ny, fs_yn, fs_nn = fs
        if any(len(fs) == 0 for fs in fs):
            # warnings.warn(
            #    f"Encountered a Core with insufficient observations ({ps}) when calculating bivLocal perturbation")
            return 0  # panic

        f_yy = np.average(fs_yy)
        f_yn = np.average(fs_yn)
        f_ny = np.average(fs_ny)
        f_nn = np.average(fs_nn)

        return f_yy + f_nn - f_yn - f_ny


class UnivariateLocalPerturbation(Metric):
    linkage_calculator: Optional[LocalPerturbationCalculator]

    def __init__(self):
        self.pRef = None
        super().__init__()

    def __repr__(self):
        return "UnivariateLocalPerturbation"

    def set_pRef(self, pRef: PRef):
        self.linkage_calculator = LocalPerturbationCalculator(pRef)

    def get_local_importance_array(self, ps: PS):
        fixed_loci = ps.get_fixed_variable_positions()
        return [self.linkage_calculator.get_delta_f_of_ps_at_locus_univariate(ps, i) for i in fixed_loci]

    def get_single_score(self, ps: PS) -> float:
        fixed_loci = ps.get_fixed_variable_positions()
        dfs = [self.linkage_calculator.get_delta_f_of_ps_at_locus_univariate(ps, i) for i in fixed_loci]
        return np.average(dfs)

    def get_single_normalised_score(self, ps: PS) -> float:
        return self.get_single_score(ps)


class BivariateLocalPerturbation(Metric):
    linkage_calculator: Optional[LocalPerturbationCalculator]


    fitness_range: float

    def __init__(self):
        self.pRef = None
        super().__init__()

    def __repr__(self):
        return "BivariateLocalPerturbation"

    def set_pRef(self, pRef: PRef):
        self.linkage_calculator = LocalPerturbationCalculator(pRef)
        self.fitness_range = np.max(pRef.fitness_array) - np.min(pRef.fitness_array)

    def get_single_score(self, ps: PS) -> float:
        if ps.fixed_count() < 2:
            if ps.fixed_count() == 1:
                fixed_locus = ps.get_fixed_variable_positions()[0]
                return self.linkage_calculator.get_delta_f_of_ps_at_locus_univariate(ps, fixed_locus)
            else:
                return 0
        fixed_loci = ps.get_fixed_variable_positions()
        pairs = list(itertools.combinations(fixed_loci, r=2))
        dfs = [self.linkage_calculator.get_delta_f_of_ps_at_loci_bivariate(ps, a, b) for a, b in pairs]
        return np.average(dfs)

    def get_single_normalised_score(self, ps: PS) -> float:
        if ps.fixed_count() < 2:
            if ps.fixed_count() == 1:
                fixed_locus = ps.get_fixed_variable_positions()[0]
                perturbation = self.linkage_calculator.get_delta_f_of_ps_at_locus_univariate(ps, fixed_locus)
                return (perturbation + self.fitness_range) / (2 * self.fitness_range)  # note how perturbation is not divided by 2, because it's univariate now
            else:
                return 0
        perturbation = self.get_single_score(ps)
        perturbation_normalised = ((perturbation / 2) + self.fitness_range) / (2 * self.fitness_range)
        return perturbation_normalised

    def get_local_linkage_table(self, ps: PS) -> np.ndarray:
        fixed_loci = ps.get_fixed_variable_positions()
        locus_index_within_loci = {locus: position for position, locus in enumerate(fixed_loci)}
        pairs = list(itertools.combinations(fixed_loci, r=2))

        linkage_table = np.zeros((ps.fixed_count(), ps.fixed_count()), dtype=float)
        for a, b in pairs:
            x = locus_index_within_loci[a]
            y = locus_index_within_loci[b]
            linkage_table[x, y] = self.linkage_calculator.get_delta_f_of_ps_at_loci_bivariate(ps, a, b)

        linkage_table += linkage_table.T
        return np.sqrt(linkage_table)

class OutdatedLinkage(Metric):
    linkage_table: Optional[LinkageTable]
    normalised_linkage_table: Optional[LinkageTable]

    def __init__(self):
        super().__init__()
        self.linkage_table = None
        self.normalised_linkage_table = None

    def __repr__(self):
        return "Linkage"

    def set_pRef(self, pRef: PRef):
        # print("Calculating linkages...", end="")
        self.linkage_table = self.get_linkage_table_fast(pRef)
        # self.normalised_linkage_table = self.get_quantized_linkage_table(self.linkage_table)
        # print("Finished")
        self.normalised_linkage_table = self.get_normalised_linkage_table(self.linkage_table)

    @staticmethod
    def get_linkage_table_fast(pRef: PRef) -> LinkageTable:
        overall_average = np.average(pRef.fitness_array)

        def get_mean_benefit_of_ps(ps: PS):
            return np.average(pRef.fitnesses_of_observations(ps)) - overall_average

        def one_fixed_var(var, val) -> PS:
            return PS.empty(pRef.search_space).with_fixed_value(var, val)

        def two_fixed_vars(var_x, val_x, var_y, val_y) -> PS:
            return (PS.empty(pRef.search_space)
                    .with_fixed_value(var_x, val_x)
                    .with_fixed_value(var_y, val_y))

        marginal_benefits = [[get_mean_benefit_of_ps(one_fixed_var(var, val))
                              for val in range(cardinality)]
                             for var, cardinality in enumerate(pRef.search_space.cardinalities)]

        def interaction_effect_between_vars(var_x: int, var_y: int) -> float:

            def addend(val_a, val_b):
                expected_conditional = marginal_benefits[var_x][val_a] + marginal_benefits[var_y][val_b]
                observed_conditional = get_mean_benefit_of_ps(two_fixed_vars(var_x, val_a, var_y, val_b))

                return abs(expected_conditional - observed_conditional)

            cardinality_x = pRef.search_space.cardinalities[var_x]
            cardinality_y = pRef.search_space.cardinalities[var_y]
            return sum(addend(val_a, val_b)
                       for val_a in range(cardinality_x)
                       for val_b in range(cardinality_y))

        linkage_table = np.zeros((pRef.search_space.amount_of_parameters, pRef.search_space.amount_of_parameters))
        for var_a in range(pRef.search_space.amount_of_parameters):
            for var_b in range(var_a, pRef.search_space.amount_of_parameters):
                linkage_table[var_a][var_b] = interaction_effect_between_vars(var_a, var_b)

        # then we mirror it for convenience...
        upper_triangle = np.triu(linkage_table, k=1)
        linkage_table = linkage_table + upper_triangle.T
        return linkage_table

    @staticmethod
    def get_linkage_table_using_chi_squared(pRef: PRef) -> LinkageTable:
        n = pRef.sample_size

        def get_p_of_ps(ps: PS):
            amount = len(pRef.fitnesses_of_observations(ps))
            return amount / n

        def one_fixed_var(var, val) -> PS:
            return PS.empty(pRef.search_space).with_fixed_value(var, val)

        def two_fixed_vars(var_x, val_x, var_y, val_y) -> PS:
            return (PS.empty(pRef.search_space)
                    .with_fixed_value(var_x, val_x)
                    .with_fixed_value(var_y, val_y))

        marginal_probabilities = [[get_p_of_ps(one_fixed_var(var, val))
                                   for val in range(cardinality)]
                                  for var, cardinality in enumerate(pRef.search_space.cardinalities)]

        def interaction_effect_between_vars(var_x: int, var_y: int) -> float:
            """Returns the chi squared value between x and y"""

            def chi_square_addend(val_a, val_b):
                expected_conditional = marginal_probabilities[var_x][val_a] * marginal_probabilities[var_y][val_b]
                observed_conditional = get_p_of_ps(two_fixed_vars(var_x, val_a, var_y, val_b))

                return ((n * observed_conditional - n * expected_conditional) ** 2) / (n * expected_conditional)

            cardinality_x = pRef.search_space.cardinalities[var_x]
            cardinality_y = pRef.search_space.cardinalities[var_y]
            return sum(chi_square_addend(val_a, val_b)
                       for val_a in range(cardinality_x)
                       for val_b in range(cardinality_y))

        linkage_table = np.zeros((pRef.search_space.amount_of_parameters, pRef.search_space.amount_of_parameters))
        for var_a in range(pRef.search_space.amount_of_parameters):
            for var_b in range(var_a, pRef.search_space.amount_of_parameters):
                linkage_table[var_a][var_b] = interaction_effect_between_vars(var_a, var_b)

        # then we mirror it for convenience...
        upper_triangle = np.triu(linkage_table, k=1)
        linkage_table = linkage_table + upper_triangle.T
        return linkage_table

    @staticmethod
    def get_linkage_table(pRef: PRef) -> LinkageTable:
        """TODO this is incredibly slow..."""
        overall_avg_fitness = np.average(pRef.fitness_array)

        empty = PS.empty(pRef.search_space)
        trivial_pss = [[empty.with_fixed_value(var_index, val)
                        for val in range(cardinality)]
                       for var_index, cardinality in enumerate(pRef.search_space.cardinalities)]

        def interaction_effect_between_pss(ps_a, ps_b) -> float:
            mean_a = np.average(pRef.fitnesses_of_observations(ps_a))
            mean_b = np.average(pRef.fitnesses_of_observations(ps_b))
            mean_both = np.average(pRef.fitnesses_of_observations(PS.merge(ps_a, ps_b)))

            benefit_a = mean_a - overall_avg_fitness
            benefit_b = mean_b - overall_avg_fitness
            benefit_both = mean_both - overall_avg_fitness
            return abs(benefit_both - benefit_a - benefit_b)

        def interaction_effect_of_value(ps_a) -> float:
            mean_a = np.average(pRef.fitnesses_of_observations(ps_a))
            benefit_a = mean_a - overall_avg_fitness
            return abs(benefit_a)

        def interaction_effect_between_vars(var_a: int, var_b: int) -> float:
            if var_a == var_b:
                return sum([interaction_effect_of_value(ps_a)
                            for ps_a in trivial_pss[var_a]])

            return sum([interaction_effect_between_pss(ps_a, ps_b)
                        for ps_b in trivial_pss[var_b]
                        for ps_a in trivial_pss[var_a]])

        linkage_table = np.zeros((pRef.search_space.amount_of_parameters, pRef.search_space.amount_of_parameters))
        for var_a in range(pRef.search_space.amount_of_parameters):
            for var_b in range(var_a, pRef.search_space.amount_of_parameters):
                linkage_table[var_a][var_b] = interaction_effect_between_vars(var_a, var_b)

        # then we mirror it for convenience...
        upper_triangle = np.triu(linkage_table, k=1)
        linkage_table = linkage_table + upper_triangle.T
        return linkage_table

    @staticmethod
    def get_normalised_linkage_table(linkage_table: LinkageTable, include_diagonal=False):
        where_to_consider = np.triu(np.full_like(linkage_table, True, dtype=bool), k=0 if include_diagonal else 1)
        triu_min = np.min(linkage_table, where=where_to_consider, initial=np.inf)
        triu_max = np.max(linkage_table, where=where_to_consider, initial=-np.inf)
        normalised_linkage_table: LinkageTable = (linkage_table - triu_min) / triu_max

        return normalised_linkage_table

    @staticmethod
    def get_quantized_linkage_table(linkage_table: LinkageTable):
        where_to_consider = np.triu(np.full_like(linkage_table, True, dtype=bool), k=1)
        average = np.average(linkage_table[where_to_consider])
        quantized_linkage_table: LinkageTable = np.array(linkage_table >= average, dtype=float)
        return quantized_linkage_table

    def get_linkage_scores(self, ps: PS) -> np.ndarray:
        fixed = ps.values != STAR
        fixed_combinations: np.array = np.outer(fixed, fixed)
        fixed_combinations = np.triu(fixed_combinations, k=1)
        return self.linkage_table[fixed_combinations]

    def get_normalised_linkage_scores(self, ps: PS) -> np.ndarray:
        fixed = ps.values != STAR
        fixed_combinations: np.array = np.outer(fixed, fixed)
        fixed_combinations = np.triu(fixed_combinations, k=0)
        return self.normalised_linkage_table[fixed_combinations]

    def get_single_score_using_avg(self, ps: PS) -> float:
        if ps.fixed_count() < 2:
            return 0
        else:
            return np.min(self.get_linkage_scores(ps))

    def get_single_normalised_score(self, ps: PS) -> float:
        self.used_evaluations += 1
        if ps.fixed_count() < 2:
            return 0
        else:
            return np.average(self.get_normalised_linkage_scores(ps))

    def get_single_score(self, ps: PS) -> float:
        return self.get_single_score_using_avg(ps)