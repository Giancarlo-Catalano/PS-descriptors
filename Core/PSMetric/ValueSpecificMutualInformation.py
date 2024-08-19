import itertools
import random
from typing import Optional

import numpy as np

import utils
from Core.PRef import PRef
from Core.PS import PS
from Core.PSMetric.Metric import Metric


class ValueSpecificMutualInformation(Metric):
    pRef: Optional[PRef]

    univariate_probability_table: Optional[list]
    bivariate_probability_table: Optional[list]

    linkage_dict: Optional[dict[(int, int, int, int), float]]

    def __init__(self):
        super().__init__()
        self.pRef = None
        self.univariate_probability_table = None
        self.bivariate_probability_table = None
        self.linkage_dict = None

    def __repr__(self):
        return "ValueSpecificMutualInformation"

    def set_pRef(self, pRef: PRef):
        self.pRef = pRef

        self.univariate_probability_table, self.bivariate_probability_table = self.calculate_probability_tables()
        self.linkage_dict = self.get_linkage_dict()

    def calculate_probability_tables(self) -> (list, list):

        amount_of_samples = 10000

        indexes = np.random.randint(self.pRef.sample_size, size=amount_of_samples)
        fitnesses = self.pRef.fitness_array[indexes]
        who_won = fitnesses > np.roll(fitnesses, 1)  # note > and not >=. This is preferred because some problems have heavy fitness collisions
        winning_indexes = indexes[who_won]
        winning_solutions = self.pRef.full_solution_matrix[winning_indexes, :]


        univariate_counts = [np.zeros(card) for card in self.pRef.search_space.cardinalities]
        cs = self.pRef.search_space.cardinalities
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


        for sample_number, sample in enumerate(winning_solutions):
            register_solution_for_univariate(sample)
            register_solution_for_bivariate(sample)
            if sample_number%(len(winning_solutions) // 100) == 0:
                print(f"MI data gathering progress: {100*sample_number/len(winning_solutions):.2f}%")

        def counts_to_probabilities(counts: np.ndarray):
            """ used for both arrays and matrices"""
            return (counts / np.sum(counts))


        univariate_probabilities = [counts_to_probabilities(var_counts) for var_counts in univariate_counts]
        bivariate_probabilities = [[counts_to_probabilities(bivariate_count_table[var_a][var_b])
                                    if var_b > var_a else None
                                    for var_b in range(len(cs))]
                                   for var_a in range(len(cs))]
        return univariate_probabilities, bivariate_probabilities

    def mutual_information(self, var_a, val_a, var_b, val_b) -> float:
        p_a = self.univariate_probability_table[var_a][val_a]
        p_b = self.univariate_probability_table[var_b][val_b]

        p_a_b = self.bivariate_probability_table[var_a][var_b][val_a, val_b]

        if p_a_b == 0:
            return 0
        return p_a_b * np.log(p_a_b/(p_a * p_b))

    def get_linkage_dict(self) -> dict[(int, int, int, int), float]:
        ss = self.pRef.search_space
        cs = ss.cardinalities
        n = ss.amount_of_parameters

        return {(var_a, val_a, var_b, val_b): self.mutual_information(var_a, val_a, var_b, val_b)
                for var_a, var_b in itertools.combinations(range(n), r=2)
                for val_a in range(cs[var_a])
                for val_b in range(cs[var_b])}

    def get_linkages_in_ps(self, ps: PS) -> list[float]:
        fixed_vars = ps.get_fixed_variable_positions()
        return [self.linkage_dict[(var_a, ps[var_a], var_b, ps[var_b])]
                for var_a, var_b in itertools.combinations(fixed_vars, r=2)]

    def get_single_score(self, ps: PS) -> float:
        fixed_count = ps.fixed_count()
        if fixed_count >= 2:
            linkages = self.get_linkages_in_ps(ps)
            return np.average(linkages)
        else:
            return 0
        # elif fixed_count == 1:
        #     [fixed_position] = ps.get_fixed_variable_positions()
        #     return self.linkage_table[fixed_position][fixed_position]
