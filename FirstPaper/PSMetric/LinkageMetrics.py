import itertools
import random
from typing import Optional

import numpy as np
from numba import njit

import utils
from FirstPaper.PRef import PRef
from FirstPaper.PS import PS, STAR
from FirstPaper.PSMetric.Metric import Metric


class Influence(Metric):
    """This metric is good for ordering PSs in terms of atomicity, but
    * it's painfully slow...
    * it's unstable for pss which are too fixed or too simple"""
    pRef: Optional[PRef]
    trivial_means: Optional[list[list[float]]]
    overall_mean: Optional[float]

    def __init__(self):
        super().__init__()
        self.pRef = None

    def __repr__(self):
        return "Influence"


    def mf(self, ps: PS) -> float:
        fitnesses = self.pRef.fitnesses_of_observations(ps)
        if len(fitnesses) < 1:
            return self.overall_mean
        else:
            return np.average(fitnesses)

    def calculate_trivial_means(self) -> list[list[float]]:
        def value_for_combination(var, val) -> float:
            ps = PS.empty(self.pRef.search_space).with_fixed_value(var, val)
            return self.mf(ps)
        return [[value_for_combination(var, val)
                 for val in range(self.pRef.search_space.cardinalities[var])]
                for var in range(self.pRef.search_space.amount_of_parameters)]

    def set_pRef(self, pRef: PRef):
        self.pRef = pRef
        self.overall_mean = np.average(pRef.fitness_array)
        self.trivial_means = self.calculate_trivial_means()


    def get_external_internal_influence(self, ps: PS) -> (float, float):
        empty_ps = PS.empty(search_space= self.pRef.search_space)
        empty_ps_mf = self.mf(empty_ps)
        ps_mf = self.mf(ps)

        if ps.is_empty():
            return (100, 0)
        if ps.is_fully_fixed():
            return (100,0)

        def absence_influence_for_var_val(var: int, val: int) -> int:
            trivial_mf = self.trivial_means[var][val]
            ps_with_trivial_mf = self.mf(ps.with_fixed_value(var, val))
            effect_on_empty = trivial_mf - empty_ps_mf
            effect_on_ps = ps_with_trivial_mf - ps_mf
            return np.abs(effect_on_ps - effect_on_empty)
        def absence_influence_for_var(var: int) -> float:
            influences = [absence_influence_for_var_val(var, val)
                          for val in range(self.pRef.search_space.cardinalities[var])]
            return np.max(influences)

        def presence_influence_for_var(var: int) -> int:
            trivial_mf = self.trivial_means[var][ps[var]]
            without_trivial_mf = self.mf(ps.with_unfixed_value(var))
            effect_on_empty = trivial_mf - empty_ps_mf
            effect_on_ps = ps_mf - without_trivial_mf
            return np.abs(effect_on_ps - effect_on_empty)


        unfixed_vars = [index for index, value in enumerate(ps.values) if value == STAR]
        absence_influences = np.array([absence_influence_for_var(var) for var in unfixed_vars])
        presence_influences = np.array([presence_influence_for_var(var) for var in ps.get_fixed_variable_positions()])

        presence_score = np.average(presence_influences)
        absence_score = np.average(absence_influences)
        return (absence_score, presence_score)


    def get_single_score(self, ps: PS) -> float:
        external_influence, internal_influence = self.get_external_internal_influence(ps)
        # internal variables should be important, external variables should be not important
        return internal_influence - external_influence




def sort_by_influence(pss: list[PS], pRef: PRef) -> list[PS]:
    evaluator = Influence()
    evaluator.set_pRef(pRef)
    return sorted(pss, key=lambda x:evaluator.get_single_score(x), reverse=True)


class MutualInformation(Metric):
    """This is the metric discussed in the paper, which replaces LegacyAtomicity"""
    """ we measure the mutual information regarding the probability of winning in a binary tournament selection"""
    sorted_pRef: Optional[PRef]

    univariate_probability_table: Optional[list]
    bivariate_probability_table: Optional[list]

    linkage_table: Optional[np.ndarray]

    def __init__(self):
        super().__init__()
        self.sorted_pRef = None
        self.univariate_probability_table = None
        self.bivariate_probability_table = None
        self.linkage_table = None

    def __repr__(self):
        return "MutualInformation"

    @classmethod
    def get_sorted_pRef(cls, pRef: PRef) -> PRef:
        indexed_fitnesses = list(enumerate(pRef.fitness_array))
        indexed_fitnesses.sort(key=utils.second, reverse=True)
        indexes, fitnesses = utils.unzip(indexed_fitnesses)

        new_matrix = pRef.full_solution_matrix[indexes]
        new_matrix = np.array(new_matrix, dtype=np.uint8) # so that it takes less space
        return PRef(fitnesses, new_matrix, search_space=pRef.search_space)
    def set_pRef(self, pRef: PRef):
        self.sorted_pRef = self.get_sorted_pRef(pRef)

        self.univariate_probability_table, self.bivariate_probability_table = self.calculate_probability_tables()
        self.linkage_table = self.get_linkage_table()

    def calculate_probability_tables(self) -> (list, list):
        indexes = list(range(len(self.sorted_pRef.fitness_array)))
        amount_of_solutions = len(self.sorted_pRef.fitness_array)
        fsm = self.sorted_pRef.full_solution_matrix
        def tournament_selection(tournament_size: int) -> np.ndarray:
            picks = random.choices(indexes, k=tournament_size)
            winner_index = min(picks)
            return self.sorted_pRef.full_solution_matrix[winner_index]

        def binary_tournament_selection() -> np.ndarray:
            first, second = random.randrange(amount_of_solutions), random.randrange(amount_of_solutions)
            winning_index = min(first, second)
            return fsm[winning_index]


        univariate_counts = [np.zeros(card) for card in self.sorted_pRef.search_space.cardinalities]
        cs = self.sorted_pRef.search_space.cardinalities
        bivariate_count_table = [[np.zeros((c2, c1), dtype=int)
                                  for c1 in cs]
                                 for c2 in cs]
        n = len(cs)

        def register_batch(solution_matrix: np.ndarray) -> None:

            for var_a in range(n):
                col_a = solution_matrix[:, var_a]

                # register univariate counts
                a_counts = np.unique(col_a, return_counts=True)
                for val_a, count_for_val in zip(*a_counts):
                    univariate_counts[var_a][val_a] += count_for_val

                for var_b in range(var_a+1, n):
                    cols_together = solution_matrix[:, [var_a, var_b]]
                    raw_counts = np.unique(cols_together, axis=0, return_counts=True)
                    for (val_a, val_b), count in zip(*raw_counts):
                        bivariate_count_table[var_a][var_b][val_a, val_b] += count

        with utils.announce("Getting information for Mutual Information", verbose=False):
            batch_size = 12 * 1024 // (fsm.itemsize * n)
            remaining_samples = amount_of_solutions
            while remaining_samples > 0:
                current_batch_size = min(remaining_samples, batch_size)
                samples_matrix = np.array([binary_tournament_selection() for _ in range(current_batch_size)])
                register_batch(samples_matrix)
                remaining_samples -= current_batch_size



        def counts_to_probabilities(counts: np.ndarray):
            """ used for both arrays and matrices"""
            return (counts / np.sum(counts))


        univariate_probabilities = [counts_to_probabilities(var_counts) for var_counts in univariate_counts]
        bivariate_probabilities = [[counts_to_probabilities(bivariate_count_table[var_a][var_b])
                                    if var_b > var_a else None
                                    for var_b in range(n)]
                                   for var_a in range(n)]
        return univariate_probabilities, bivariate_probabilities

    def get_linkage_between_vars(self, var_a:int, var_b:int) -> float:

        def mutual_information(value_a: int, value_b: int) -> float:
            p_a = self.univariate_probability_table[var_a][value_a]
            p_b = self.univariate_probability_table[var_b][value_b]

            p_a_b = self.bivariate_probability_table[var_a][var_b][value_a, value_b]
            return p_a_b * np.log(p_a_b/(p_a * p_b))


        ss = self.sorted_pRef.search_space
        return sum(mutual_information(value_a, value_b)
                   for value_a in range(ss.cardinalities[var_a])
                   for value_b in range(ss.cardinalities[var_b]))

    def get_linkage_table(self) -> np.ndarray:
        param_count = self.sorted_pRef.search_space.amount_of_parameters
        table = np.zeros((param_count, param_count), dtype=float)
        for var_a in range(param_count):
            for var_b in range(var_a+1, param_count):
                table[var_a][var_b] = self.get_linkage_between_vars(var_a, var_b)

        table += table.T

        return table

    def get_linkages_in_ps(self, ps: PS) -> list[float]:
        fixed_vars = ps.get_fixed_variable_positions()
        return [self.linkage_table[var_a, var_b] for var_a, var_b in itertools.combinations(fixed_vars, r=2)]

    def get_single_score(self, ps: PS) -> float:
        linkages = self.get_linkages_in_ps(ps)
        if len(linkages) == 0:
            return 0
        return np.median(linkages)