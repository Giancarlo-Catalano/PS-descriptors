import itertools
from typing import Optional

import numpy as np

import utils
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS, STAR
from Core.PSMetric.Metric import Metric


class PerturbationOfSolution(Metric):
    # best metric as of 25/09/24
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
                                                                         difference_upper_bound=len(solution) - 1)

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

        assert(len(no_difference_fitnesses) > 0)
        no_difference_mean = np.average(no_difference_fitnesses)

        def safe_mean(values):
            if len(values) < 1:
                return no_difference_mean   # so that the linkage at the end will be zero in theory. Returning -10000 causes lots of issues...
            return np.average(values)


        one_difference_means = [safe_mean(values) for values in one_diffence_fitnesses]
        two_difference_means = {key: safe_mean(values) for key, values in two_difference_fitnesses.items()}

        def get_linkage(a: int, b: int) -> float:
            return np.abs(
                no_difference_mean + two_difference_means[(a, b)] - one_difference_means[a] - one_difference_means[b])

        def get_importance(a: int) -> float:
            return np.abs(no_difference_mean - one_difference_means[a])

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

        table += table.T

        for a in range(n):
            table[a, a] = get_importance(a)

        # table /= background_variance

        return table

    def get_atomicity(self, ps: PS) -> float:
        def get_from_minimum_internal():
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

        def get_from_average_internal():
            if ps.is_empty():
                return 0
            fixed_positions = ps.get_fixed_variable_positions()

            if len(fixed_positions) > 1:
                linkages = [self.current_linkage_table[fixed, other_fixed]
                            for fixed, other_fixed in itertools.combinations(fixed_positions, r=2)]
                return np.average(linkages)
            else:
                singleton = fixed_positions[0]
                return self.current_linkage_table[singleton, singleton]

        def get_from_NDSGAII_method():
            if ps.is_empty():
                return 0
            fixed_positions = ps.get_fixed_variable_positions()

            if len(fixed_positions) > 1:
                linkages = [self.current_linkage_table[fixed, other_fixed]
                            for fixed, other_fixed in itertools.combinations(fixed_positions, r=2)]
                return np.sum(linkages) / (ps.fixed_count())
            else:
                singleton = fixed_positions[0]
                return self.current_linkage_table[singleton, singleton]

        return get_from_average_internal()

    def get_linkage_threshold(self) -> float:
        values_to_check = self.current_linkage_table[np.triu_indices(len(self.current_solution), 1)]
        values_to_check = list(values_to_check)

        def get_at_greatest_slope():
            values_to_check.sort()

            differences = [(index, values_to_check[index] - values_to_check[index - 1])
                           for index in range(len(values_to_check) // 2, len(values_to_check))]
            best_index, best_difference = max(differences, key=utils.second)
            return np.average(values_to_check[(best_index - 1):(best_index + 1)])

        def get_average() -> float:
            return np.average(values_to_check)

        return get_average()

    def get_dependence(self, ps: PS) -> float:
        def get_from_max_external():
            fixed_vars = ps.get_fixed_variable_positions()
            unfixed_vars = [index for index, val in enumerate(ps.values) if val == STAR]

            def max_linkage_with_unfixed(fixed: int) -> float:
                return max(self.current_linkage_table[fixed, unfixed] for unfixed in unfixed_vars)

            if (len(unfixed_vars) > 0) and (len(fixed_vars) > 0):  # maybe this should be zero?
                outwards_linkages = [max_linkage_with_unfixed(fixed) for fixed in fixed_vars]
                return np.average(outwards_linkages)
            else:
                return 10000

        def get_from_average_external():
            fixed_vars = ps.get_fixed_variable_positions()
            unfixed_vars = [index for index, val in enumerate(ps.values) if val == STAR]

            if (len(unfixed_vars) > 0) and (len(fixed_vars) > 0):  # maybe this should be zero?
                outwards_linkages = [self.current_linkage_table[fixed, unfixed]
                                     for fixed in fixed_vars
                                     for unfixed in unfixed_vars]
                return np.average(outwards_linkages)
            else:
                return 10000

        return get_from_average_external()
