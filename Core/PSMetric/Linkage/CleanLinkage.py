import itertools
import random
from typing import Optional

import numpy as np

import utils
from Core.PRef import PRef
from Core.PS import PS
from Core.PSMetric.Linkage.Additivity import MutualInformation


class CleanLinkage(MutualInformation):
    sorted_pRef: Optional[PRef]

    univariate_probability_table: Optional[list]
    bivariate_probability_table: Optional[list]

    importance_array: Optional[np.ndarray]
    linkage_table: Optional[np.ndarray]

    def __init__(self):
        super().__init__()
        self.sorted_pRef = None
        self.univariate_probability_table = None
        self.bivariate_probability_table = None
        self.linkage_table = None
        self.importance_array = None

    def __repr__(self):
        return "CleanLinkage"



    def get_amount_of_params(self) -> int:
        return self.sorted_pRef.search_space.amount_of_parameters
    @classmethod
    def get_sorted_pRef(cls, pRef: PRef) -> PRef:
        indexed_fitnesses = list(enumerate(pRef.fitness_array))
        indexed_fitnesses.sort(key=utils.second, reverse=True)
        indexes, fitnesses = utils.unzip(indexed_fitnesses)

        new_matrix = pRef.full_solution_matrix[indexes]
        return PRef(fitnesses, new_matrix, search_space=pRef.search_space)
    def set_pRef(self, pRef: PRef):
        self.sorted_pRef = self.get_sorted_pRef(pRef) # I don't think I need to sort the pRef... TODO

        self.univariate_probability_table, self.bivariate_probability_table = self.calculate_probability_tables()

        self.importance_array = self.get_univariate_importance_values()
        self.importance_array = utils.remap_array_in_zero_one(self.importance_array)

        self.linkage_table = self.get_linkage_table()
        self.linkage_table = self.get_normalised_bivariate(self.linkage_table)

        np.fill_diagonal(self.linkage_table, self.importance_array)

    def calculate_probability_tables(self) -> (list, list):
        indexes = list(range(len(self.sorted_pRef.fitness_array)))
        def tournament_selection(tournament_size: int) -> np.ndarray:
            picks = random.choices(indexes, k=tournament_size)
            winner_index = min(picks)
            return self.sorted_pRef.full_solution_matrix[winner_index]


        univariate_counts = [np.zeros(card) for card in self.sorted_pRef.search_space.cardinalities]
        cs = self.sorted_pRef.search_space.cardinalities
        bivariate_count_table = [[np.zeros((c2, c1), dtype=int)
                                  for c1 in cs]
                                 for c2 in cs]
        def register_solution_for_univariate(solution: np.ndarray):
            for var, value in enumerate(solution):
                univariate_counts[var][value] += 1


        def register_solution_for_bivariate(solution: np.ndarray):
            for var_a, value_a in enumerate(solution):
                for var_b in range(var_a+1, len(solution)):
                    value_b = solution[var_b]
                    bivariate_count_table[var_a][var_b][value_a, value_b] += 1


        amount_of_samples = min(len(self.sorted_pRef.fitness_array), 10000)
        for sample_number in range(amount_of_samples):
            sample = tournament_selection(2)
            register_solution_for_univariate(sample)
            register_solution_for_bivariate(sample)
            if sample_number%(amount_of_samples // 100) == 0:
                print(f"CMI data gathering progress: {100*sample_number/amount_of_samples:.2f}%")

        def counts_to_probabilities(counts: np.ndarray):
            """ used for both arrays and matrices"""
            return (counts / np.sum(counts))


        univariate_probabilities = [counts_to_probabilities(var_counts) for var_counts in univariate_counts]
        bivariate_probabilities = [[counts_to_probabilities(bivariate_count_table[var_a][var_b])
                                    if var_b > var_a else None
                                    for var_b in range(len(cs))]
                                   for var_a in range(len(cs))]
        return univariate_probabilities, bivariate_probabilities

    def get_linkage_between_vars(self, var_a:int, var_b:int) -> float:

        def mutual_information(value_a: int, value_b: int) -> float:
            p_a = self.univariate_probability_table[var_a][value_a]
            p_b = self.univariate_probability_table[var_b][value_b]

            p_a_b = self.bivariate_probability_table[var_a][var_b][value_a, value_b]

            if p_a_b == 0:
                return 0
            return p_a_b * np.log(p_a_b/(p_a * p_b))


        ss = self.sorted_pRef.search_space
        return sum(mutual_information(value_a, value_b)
                   for value_a in range(ss.cardinalities[var_a])
                   for value_b in range(ss.cardinalities[var_b]))

    def get_linkage_table(self) -> np.ndarray:
        param_count = self.get_amount_of_params()
        table = np.zeros((param_count, param_count), dtype=float)
        for var_a in range(param_count):
            for var_b in range(var_a+1, param_count):
                table[var_a][var_b] = self.get_linkage_between_vars(var_a, var_b)

        table += table.T

        return table

    def get_normalised_bivariate(self, old_table: np.ndarray) -> np.ndarray:
        indices_to_modify = np.triu_indices(self.get_amount_of_params(), k=1)
        area_to_normalise = old_table[indices_to_modify]
        new_values = utils.remap_array_in_zero_one(area_to_normalise)

        result = np.zeros_like(old_table)
        result[indices_to_modify] = new_values
        result += result.T
        return result
    def get_univariate_importance_values(self) -> np.ndarray:
        def get_entropy_for_var(var_index: int) -> float:
            probabilities = self.univariate_probability_table[var_index]
            return -np.sum(probabilities * np.log(probabilities))
        return np.array([get_entropy_for_var(i) for i in range(self.sorted_pRef.search_space.amount_of_parameters)])

    def get_single_score(self, ps: PS) -> float:
        if ps.is_empty():
            return 0

        if ps.fixed_count() < 2:
            fixed_position = ps.get_fixed_variable_positions()[0]
            return float(self.importance_array[fixed_position])

        pairs = itertools.combinations(ps.get_fixed_variable_positions(), r=2)
        linkages = [float(self.linkage_table[var_a, var_b])
                    for var_a, var_b in pairs]

        return np.average(linkages)
