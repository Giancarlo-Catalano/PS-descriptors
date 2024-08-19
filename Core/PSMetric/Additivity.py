import itertools
import random
from typing import Optional

import numpy as np

import utils
from Core.PRef import PRef
from Core.PS import PS, STAR
from Core.PSMetric.Metric import Metric
from Core.custom_types import ArrayOfFloats


class Additivity(Metric):
    pRef: Optional[PRef]


    def __init__(self, which: int):
        super().__init__()
        self.pRef = None
        self.which = which

    def set_pRef(self, pRef: PRef):
        self.pRef = pRef

    def __repr__(self):
            return f"Additivity({self.which})"

    def get_fitnesses_split_by_error(self, ps: PS) -> (float, float, float):
        fixed_vars = ps.get_fixed_variable_positions()
        relevant_rows_in_pRef = self.pRef.full_solution_matrix[:, fixed_vars]
        correct_values = ps.values[fixed_vars]
        error_counts = np.sum(relevant_rows_in_pRef != correct_values, axis=1)

        def mean_with_amount_of_errors(error_count: int) -> np.ndarray:
            return np.mean(self.pRef.fitness_array[error_counts == error_count])

        return mean_with_amount_of_errors(0), mean_with_amount_of_errors(1), mean_with_amount_of_errors(2)


    def get_single_score(self, ps: PS) -> float:
        order = ps.fixed_count()
        if order == 0:
            return 0.0
        no_err, one_err, two_err = self.get_fitnesses_split_by_error(ps)

        if order == 1:
            return 0.0  # but maybe it should be no_err - one_err

        if np.isnan(no_err) or np.isnan(one_err) or np.isnan(two_err):
            return 0.0


        alpha = 2 * (no_err - one_err)
        beta = (no_err - two_err)

        var_a = alpha - beta
        var_b = abs(alpha - beta)
        var_c = abs(alpha) - abs(beta)
        var_d = abs(abs(alpha) - abs(beta))
        # the below formula could be in an abs(.), but that would promote hindrance effects I think
        return [var_a, var_b, var_c, var_d][self.which]  # it could be simplified





class MeanError(Metric):
    pRef: Optional[PRef]

    def __init__(self):
        super().__init__()
        self.pRef = None

    def __repr__(self):
        return "SimplePerturbation"
    def set_pRef(self, pRef: PRef):
        self.pRef = pRef


    def get_perturbation_items_at_loci(self, ps: PS, a:int, b:int) -> (PS, PS, PS, PS):
        if ps[a] == -1 or ps[b] == -1:
            raise Exception("I need fixed vars!!!")

        p_ss = ps.copy()
        p_cs = ps.with_fixed_value(a, 1-ps[a])
        p_sc = ps.with_fixed_value(b, 1-ps[b])
        p_cc = p_cs.with_fixed_value(b, 1-ps[b])
        return p_ss, p_cs, p_sc, p_cc


    def get_perturbation_at_loci(self, ps: PS, a: int, b: int) -> float:
        p_ss, p_cs, p_sc, p_cc = self.get_perturbation_items_at_loci(ps, a, b)

        def m(input_ps: PS) -> float:
            return np.average(self.pRef.fitnesses_of_observations(input_ps))

        return abs(m(p_ss) + m(p_cc) - m(p_sc) - m(p_cs))


    def get_single_score(self, ps: PS) -> float:
        return -utils.get_mean_error(self.pRef.fitnesses_of_observations(ps))



class Influence(Metric):
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
        return PRef(fitnesses, new_matrix, search_space=pRef.search_space)
    def set_pRef(self, pRef: PRef):
        self.sorted_pRef = self.get_sorted_pRef(pRef)

        self.univariate_probability_table, self.bivariate_probability_table = self.calculate_probability_tables()
        self.linkage_table = self.get_linkage_table()

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
                print(f"MI data gathering progress: {100*sample_number/amount_of_samples:.2f}%")

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

    def get_univariate_entropies(self) -> np.ndarray:
        as_array = np.array(self.univariate_probability_table)
        return np.sum(-1 * as_array * np.log(as_array), axis=1)

    def get_linkage_table(self) -> np.ndarray:
        param_count = self.sorted_pRef.search_space.amount_of_parameters
        table = np.zeros((param_count, param_count), dtype=float)
        for var_a in range(param_count):
            for var_b in range(var_a+1, param_count):
                table[var_a][var_b] = self.get_linkage_between_vars(var_a, var_b)

        table += table.T

        # def univariate_joint_entropy(var_index: int) -> float:
        #     values = self.univariate_probability_table[var_index]
        #     individual_mutual_infos = values * np.log(1/values)
        #     return np.sum(individual_mutual_infos)
        #
        # diagonal_values = np.array([univariate_joint_entropy(i)
        #                             for i in range(self.sorted_pRef.search_space.amount_of_parameters)])
        # np.fill_diagonal(table, diagonal_values)
        #
        #
        # table += np.identity(self.sorted_pRef.search_space.amount_of_parameters) * self.get_univariate_entropies()

        return table

    def get_linkages_in_ps(self, ps: PS) -> list[float]:
        fixed_vars = ps.get_fixed_variable_positions()
        return [self.linkage_table[var_a, var_b] for var_a, var_b in itertools.combinations(fixed_vars, r=2)]

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
