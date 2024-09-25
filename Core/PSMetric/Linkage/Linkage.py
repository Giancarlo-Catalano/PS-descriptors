from typing import TypeAlias, Optional

import numpy as np

from Core.PRef import PRef
from Core.PS import PS, STAR
from Core.PSMetric.Metric import Metric

LinkageTable: TypeAlias = np.ndarray


class Linkage(Metric):
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
