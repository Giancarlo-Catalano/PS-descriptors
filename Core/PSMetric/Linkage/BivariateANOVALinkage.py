import itertools
import warnings
from typing import TypeAlias, Optional

import numpy as np
from scipy.stats import f

from Core.PRef import PRef
from Core.PS import PS, STAR
from Core.PSMetric.Linkage.OutdatedLinkage import OutdatedLinkage
from Core.PSMetric.Metric import Metric

LinkageTable: TypeAlias = np.ndarray


class BivariateANOVALinkage(Metric):
    linkage_table: Optional[LinkageTable]
    normalised_linkage_table: Optional[LinkageTable]

    def __init__(self):
        super().__init__()
        self.linkage_table = None
        self.normalised_linkage_table = None

    def __repr__(self):
        return "BiVariateANOVALinkage"

    def set_pRef(self, pRef: PRef):
        # print("Calculating linkages...", end="")
        self.linkage_table = self.get_linkage_table(pRef)
        self.normalised_linkage_table = OutdatedLinkage.get_normalised_linkage_table(self.linkage_table)
        # print("Finished")

    def get_ANOVA_interaction_table(self, pRef: PRef) -> LinkageTable:
        """every entry in this table will be a p-value, so in theory smaller values have stronger linkage"""
        solutions = pRef.full_solution_matrix
        fitnesses = pRef.fitness_array

        grand_mean = np.mean(fitnesses)
        n = pRef.sample_size
        dof_total = n - 1
        amount_of_variables = pRef.search_space.amount_of_parameters

        levels = pRef.search_space.cardinalities

        if n == 0:
            raise Exception("0 samples in ANOVA when calculating linkage table.")

        where_values = [[(solutions[:, i] == level_i) for level_i in range(levels[i])]
                        for i in range(amount_of_variables)]

        def interaction_test(i: int, j: int):
            """ Perform a 2-factor ANOVA test with interaction term """

            where_values_i = where_values[i]
            where_values_j = where_values[j]

            # Calculating the sum of squares for the interaction
            # (Normally we'd also calculate the marginal sum of squares, but we don't need them here.

            # debug
            warnings.filterwarnings("error")
            try:
                sum_sq_interaction = np.sum([(np.mean(fitnesses[where_val_i & where_val_j]) -
                                              np.mean(fitnesses[where_val_i]) -
                                              np.mean(fitnesses[where_val_j]) +
                                              grand_mean) ** 2 for where_val_i, where_val_j in
                                             itertools.product(where_values_i, where_values_j)])
            except RuntimeWarning as w:  # sometimes we get a mean of empty slice error
                # print(f"Received the warning {w} when calculating the sum_sq_interaction")
                sum_sq_interaction = 0

            warnings.resetwarnings()

            # Calculate error sum of squares
            ss_error = np.sum((fitnesses - np.mean(fitnesses)) ** 2)

            # Calculate degrees of freedom
            dof_factor_i = pRef.search_space.cardinalities[i] - 1
            dof_factor_j = pRef.search_space.cardinalities[j] - 1
            dof_interaction = dof_factor_i * dof_factor_j
            dof_error = dof_total - (dof_factor_i + dof_factor_j + dof_interaction)

            # Calculate mean squares
            ms_interaction = sum_sq_interaction / dof_interaction
            ms_error = ss_error / dof_error

            # Calculate F statistic
            f_statistic = (ms_interaction / ms_error) if ms_error != 0 else np.inf

            # Calculate p-value
            p_value = 1 - f.cdf(f_statistic, dof_interaction, dof_error)
            return p_value

        def calculate_interaction(data: np.ndarray):
            num_features = data.shape[1]
            interaction_table = np.zeros((num_features, num_features))
            for i, j in itertools.combinations(range(num_features), 2):
                interaction_table[i, j] = interaction_test(i, j)
            return interaction_table + interaction_table.T  # Make the table symmetric

        return calculate_interaction(solutions)

    def get_linkage_table(self, pRef: PRef):
        # from_anova = self.get_ANOVA_interaction_table(pRef)
        table = 1 - self.get_ANOVA_interaction_table(pRef)
        # for debugging purposes, just so that it looks prettier in the PyCharm debugging window.
        np.fill_diagonal(table, 0)
        return table

    def get_normalised_linkage_scores(self, ps: PS, include_reflexive=False) -> np.ndarray:
        fixed = ps.values != STAR
        fixed_combinations: np.array = np.outer(fixed, fixed)
        fixed_combinations = np.triu(fixed_combinations, k=0 if include_reflexive else 1)
        return self.normalised_linkage_table[fixed_combinations]

    def get_single_normalised_score(self, ps: PS) -> float:
        self.used_evaluations += 1
        if ps.fixed_count() < 2:
            return 0
        else:
            return np.average(self.get_normalised_linkage_scores(ps))


    def get_single_score(self, ps: PS) -> float:
        return self.get_single_normalised_score(ps)

    def get_quantized_linkage_table(self, linkage_table: LinkageTable):
        in_zero_one_range = OutdatedLinkage.get_normalised_linkage_table(linkage_table)
        return np.array(in_zero_one_range > 0.5, dtype=float)
