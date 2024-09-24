import itertools
import warnings
from collections import defaultdict
from typing import Optional

import numpy as np

import utils
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS, STAR
from Core.PSMetric.Metric import Metric
from Core.custom_types import ArrayOfBools, ArrayOfFloats


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
                return (perturbation + self.fitness_range) / (
                            2 * self.fitness_range)  # note how perturbation is not divided by 2, because it's univariate now
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


class PerturbationOfSolution(Metric):
    pRef: Optional[PRef]

    current_linkage_table: Optional[np.ndarray]
    current_solution: Optional[FullSolution]

    def __init__(self):
        super().__init__()
        self.pRef = None
        self.current_linkage_table = None
        self.current_solution = None

    def set_pRef(self, pRef: PRef):
        self.pRef = pRef

    def set_solution(self, solution: FullSolution):
        if self.current_solution is not None and solution == self.current_solution:
            return  # saves repeated calculations
        self.current_solution = solution
        self.current_linkage_table = self.get_linkage_table_for_solution(self.current_solution,
                                                                             difference_upper_bound=len(solution) // 2)
        return  # just to set a break point

    def get_linkage_table_for_solution(self, solution: FullSolution, difference_upper_bound: int) -> np.ndarray:
        n = len(solution)
        no_difference_fitnesses = []
        one_diffence_fitnesses = [[] for i in range(n)]
        two_difference_fitnesses = {(a, b): [] for a, b in itertools.combinations(range(n), r=2)}

        difference_matrix: np.ndarray = self.pRef.full_solution_matrix != solution.values
        for difference_row, fitness in zip(difference_matrix, self.pRef.fitness_array):
            diff_count = sum(difference_row)
            if diff_count >= difference_upper_bound:
                continue

            for a, is_different in enumerate(difference_row):
                if is_different:
                    one_diffence_fitnesses[a].append(fitness)
            for a, b in itertools.combinations(range(n), r=2):
                a_is_different = difference_row[a]
                b_is_different = difference_row[b]
                if a_is_different and b_is_different:
                    two_difference_fitnesses[(a, b)].append(fitness)
                elif not a_is_different and not b_is_different:
                    no_difference_fitnesses.append(fitness)

        def safe_mean(values):
            if len(values) < 1:
                return -100000
            return np.average(values)

        no_difference_mean = safe_mean(no_difference_fitnesses)
        one_difference_means = [safe_mean(values) for values in one_diffence_fitnesses]
        two_difference_means = {key: safe_mean(values) for key, values in two_difference_fitnesses.items()}

        def get_linkage(a: int, b: int) -> float:
            return np.abs(no_difference_mean + two_difference_means[(a, b)] - one_difference_means[a] - one_difference_means[b])

        def get_importance(a: int) -> float:
            return np.abs(no_difference_mean - one_difference_means[a]) / 2

        def safe_variance(values):
            if len(values) < 2:
                return 0
            else:
                return np.var(values)


        # no_difference_variance = safe_variance(no_difference_fitnesses)
        # one_difference_variance = [safe_variance(values) for values in one_diffence_fitnesses]
        # two_difference_variance = {key: safe_variance(values) for key, values in two_difference_fitnesses.items()}
        #
        # diff_counts = np.sum(difference_matrix, axis=1)
        # eligible_rows = diff_counts < difference_upper_bound
        # background_variance = safe_variance(self.pRef.fitness_array[eligible_rows])
        #
        # def get_importance(a: int) -> float:
        #     return one_difference_variance[a]
        #
        # def get_linkage(a: int, b: int) -> float:
        #      return np.abs(notwo_difference_variance[(a, b)] - one_difference_variance[a] - one_difference_variance[b])

        table = np.zeros(shape=(n, n))
        for a, b in itertools.combinations(range(n), r=2):
            table[a, b] = get_linkage(a, b)

        for a in range(n):
            table[a, a] = get_importance(a)

        table += table.T

        # table /= background_variance

        return table


    def get_traditional_atomicity(self, ps: PS) -> float:
        if ps.is_empty():
            return 0

        fixed_positions = ps.get_fixed_variable_positions()
        if len(fixed_positions) > 1:
            linkages = [self.current_linkage_table[a, b]
                        for a, b in itertools.combinations(fixed_positions, r=2)]
            return np.average(linkages)
        else:
            singleton = fixed_positions[0]
            return self.current_linkage_table[singleton, singleton]


    def get_atomicity(self, ps: PS) -> float:
        if ps.is_empty():
            return 0
        fixed_positions = ps.get_fixed_variable_positions()

        def min_linkage_with_fixed(fixed: int) -> float:
            return min(self.current_linkage_table[fixed, other_fixed] for other_fixed in fixed_positions)

        if len(fixed_positions) > 1:
            linkages = [min_linkage_with_fixed(fixed) for fixed in fixed_positions]
            return np.average(linkages)
        else:
            singleton = fixed_positions[0]
            return self.current_linkage_table[singleton, singleton]


    def get_linkage_threshold(self) -> float:
        values_to_check = self.current_linkage_table[np.triu_indices(len(self.current_solution), 1)]
        values_to_check = list(values_to_check)
        values_to_check.sort()

        differences = [(index, values_to_check[index] - values_to_check[index-1])
                       for index in range(len(values_to_check)//2, len(values_to_check))]
        best_index, best_difference = max(differences, key=utils.second)
        return np.average(values_to_check[(best_index-1):(best_index+1)])


    def get_dependence(self, ps: PS) -> float:
        fixed_vars = ps.get_fixed_variable_positions()
        unfixed_vars = [index for index, val in enumerate(ps.values) if val == STAR]

        def max_linkage_with_unfixed(fixed: int) -> float:
            return max(self.current_linkage_table[fixed, unfixed] for unfixed in unfixed_vars)

        if (len(unfixed_vars) > 0) and (len(fixed_vars) > 0):  # maybe this should be zero?
            outwards_linkages = [max_linkage_with_unfixed(fixed) for fixed in fixed_vars]
            return np.average(outwards_linkages)
        else:
            return 10000


