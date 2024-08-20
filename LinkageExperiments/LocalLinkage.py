import itertools
import random
from typing import Optional, TypeAlias

import numpy as np

import utils
from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from BenchmarkProblems.RoyalRoad import RoyalRoad
from BenchmarkProblems.RoyalRoadWithOverlaps import RoyalRoadWithOverlaps
from BenchmarkProblems.Trapk import Trapk
from Core.EvaluatedPS import EvaluatedPS
from Core.PRef import PRef
from Core.PS import PS, STAR
from Core.PSMetric.Metric import Metric
from Core.SearchSpace import SearchSpace

LinkageHolder: TypeAlias = dict[tuple, float]

class LocalLinkage(Metric):
    sorted_pRef: Optional[PRef]

    univariate_probability_table: Optional[list]
    bivariate_probability_table: Optional[list]

    linkage_dict: Optional[LinkageHolder]

    def __init__(self):
        super().__init__()
        self.sorted_pRef = None
        self.univariate_probability_table = None
        self.bivariate_probability_table = None
        self.linkage_dict = None

    def __repr__(self):
        return "LocalLinkage"



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
        self.linkage_dict = self.get_linkage_holder()

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

    def get_linkage_between_vals(self,
                                 var_a:int,
                                 val_a:int,
                                 var_b:int,
                                 val_b:int) -> float:
        p_a = self.univariate_probability_table[var_a][val_a]
        p_b = self.univariate_probability_table[var_b][val_b]

        p_a_b = self.bivariate_probability_table[var_a][var_b][val_a, val_b]

        if p_a_b == 0:
            return 0
        return p_a_b * np.log(p_a_b / (p_a * p_b))

    def get_linkage_holder(self) -> LinkageHolder:
        linkage_holder = dict()
        ss = self.sorted_pRef.search_space

        for var_a, var_b in itertools.combinations(range(ss.amount_of_parameters), r=2):
            for val_a in range(ss.cardinalities[var_a]):
                for val_b in range(ss.cardinalities[var_b]):
                    key = (var_a, val_a, var_b, val_b)
                    linkage_holder[key] = self.get_linkage_between_vals(var_a, val_a, var_b, val_b)
        return linkage_holder

    def get_linkages_in_ps(self, ps: PS) -> list[float]:
        fixed_vars = ps.get_fixed_variable_positions()
        linkages = []
        for var_a, var_b in itertools.combinations(fixed_vars, r=2):
            val_a = ps[var_a]
            val_b = ps[var_b]
            linkages.append(self.linkage_dict[(var_a, val_a, var_b, val_b)])
        return linkages

    def get_single_score(self, ps: PS) -> float:
        fixed_count = ps.fixed_count()
        if fixed_count > 1:
            linkages = self.get_linkages_in_ps(ps)
            return np.average(linkages)
        else:
            return 0


class JustVariance(Metric):

    pRef: PRef
    global_variance: float

    def __init__(self,
                 pRef: PRef):
        super().__init__()
        self.pRef = pRef
        self.global_variance = self.get_variance(PS.empty(self.pRef.search_space))

    def get_variance(self, ps: PS) -> float:
        return np.var(self.pRef.fitnesses_of_observations(ps))

    def get_single_score(self, ps: PS) -> float:

        if ps.is_empty():
            return 0

        if ps.fixed_count() == 1:
            return self.global_variance-self.get_variance(ps)

        univariate_pss = [PS.empty(self.pRef.search_space).with_fixed_value(var, val)
                          for var, val in enumerate(ps.values)
                          if val != STAR]

        univariate_variances = [self.get_variance(u_ps) for u_ps in univariate_pss]
        local_variance = self.get_variance(ps)

        return self.global_variance - np.sum(univariate_variances) + local_variance


class BivariateVariance(Metric):
    global_variance: Optional[float]
    univariate_variance_dict: Optional[dict[(int, int), float]]
    bivariate_variance_dict: Optional[dict[(int, int, int, int), float]]

    pRef: Optional[PRef]

    univariate_importances: Optional[dict[(int, int), float]]
    bivariate_linkages: Optional[dict[(int, int, int, int), float]]


    def __init__(self):
        self.global_variance = None
        self.univariate_variance_dict = None
        self.bivariate_variance_dict = None
        self.pRef = None

        self.univariate_importances = None
        self.bivariate_linkages = None

        super().__init__()

    def set_pRef(self, pRef: PRef):
        self.pRef = pRef
        self.global_variance = float(np.var(self.pRef.fitness_array))
        self.univariate_variance_dict, self.bivariate_variance_dict = self.get_variance_dicts()
        self.univariate_importances = self.get_univariate_importance()
        self.bivariate_linkages = self.get_bivariate_linkages()

    def get_variance_dicts(self) -> (dict, dict):
        sols = self.pRef.full_solution_matrix
        fitnesses = self.pRef.fitness_array
        cardinalities = self.pRef.search_space.cardinalities
        def get_rows_with_var_val(var: int, val: int) -> np.ndarray:
            return sols[:, var] == val

        univariate_rows = {(var, val): get_rows_with_var_val(var, val)
                           for var, cardinality in enumerate(cardinalities)
                           for val in range(cardinality)}

        def get_variance_of_rows(rows: np.ndarray) -> float:
            return float(np.var(fitnesses[rows]))

        univariate_variances: dict[(int, int), float] = {var_val: get_variance_of_rows(univariate_rows[var_val])
                                for var_val in univariate_rows}

        def get_bivariate_rows(var_a:int, val_a:int, var_b:int, val_b:int) -> np.ndarray:
            var_a_rows = univariate_rows[(var_a, val_a)]
            var_b_rows = univariate_rows[(var_b, val_b)]
            return np.logical_and(var_a_rows, var_b_rows)

        bivariate_variances = {(var_a, val_a, var_b, val_b):
                                   get_variance_of_rows(get_bivariate_rows(var_a, val_a, var_b, val_b))
                               for (var_a, val_a) in univariate_variances
                               for (var_b, val_b) in univariate_variances
                               if var_a < var_b}

        return univariate_variances, bivariate_variances

    def get_univariate_importance(self):
        #return self.global_variance - self.univariate_variance_dict  # I'm not sure how to preprocess it to match the scales of the linkage dict
        return {key: self.global_variance - value
                for key, value in self.univariate_variance_dict.items()}

    def get_bivariate_linkages(self):
        def get_linkage(var_a, val_a, var_b, val_b) -> float:
            variance_a = self.univariate_variance_dict[(var_a, val_a)]
            variance_b = self.univariate_variance_dict[(var_b, val_b)]
            return self.global_variance - variance_a - variance_b + self.bivariate_variance_dict[(var_a, val_a, var_b, val_b)]

        return {key: get_linkage(*key)
                for key in self.bivariate_variance_dict}


    def get_single_score(self, ps: PS) -> float:
        fixed_count = ps.fixed_count()
        fixed_vars = ps.get_fixed_variable_positions()
        if fixed_count > 1:
            def get_linkage_of_pair(var_a, var_b) -> float:
                if var_a == var_b:
                    return self.univariate_importances[(var_a, ps[var_a])]
                else:
                    return self.bivariate_linkages[(var_a, ps[var_a], var_b, ps[var_b])]

            return np.average([get_linkage_of_pair(var_a, var_b)
                               for var_a, var_b in itertools.combinations(fixed_vars, r=2)])
        else:
            return 0
        # elif fixed_count ==1:
        #     return 0
        #     # var = fixed_vars[0]
        #     # return (-self.global_variance+self.univariate_importances[(var, ps[var])])/2
        # else:
        #     return self.global_variance


class MarkovianSampler:
    transition_matrix: np.ndarray
    iterations: int
    search_space: SearchSpace

    def __init__(self,
                 transition_matrix: np.ndarray,
                 iterations: int,
                 search_space: SearchSpace):
        self.transition_matrix = transition_matrix
        self.iterations = iterations
        self.search_space = search_space

    @classmethod
    def from_linkage_metric(cls, linkage_metric: BivariateVariance, iterations: int = 5):
        transition_matrix = cls.get_transition_matrix(linkage_metric)
        return cls(transition_matrix, iterations, linkage_metric.pRef.search_space)

    @classmethod
    def get_transition_matrix(cls, linkage_metric: BivariateVariance) -> np.ndarray:
        ss = linkage_metric.pRef.search_space
        cumulative_starts = np.cumsum(ss.cardinalities)-ss.cardinalities[0] # so that it starts from 0
        var_positions = {(var, val): cumulative_starts[var]+val
                         for var, card in enumerate(ss.cardinalities)
                         for val in range(card)}
        table_size = sum(ss.cardinalities)

        transition_matrix = np.zeros(shape= (table_size, table_size))

        for var_a, var_b in itertools.combinations_with_replacement(range(ss.amount_of_parameters), r=2):
            for val_a in range(ss.cardinalities[var_a]):
                position_a = var_positions[(var_a, val_a)]
                for val_b in range(ss.cardinalities[var_b]):
                    position_b = var_positions[(var_b, val_b)]

                    ps = PS.empty(ss).with_fixed_value(var_a, val_a).with_fixed_value(var_b, val_b)
                    variance = linkage_metric.get_single_score(ps)
                    transition_matrix[position_a, position_b] = variance

        transition_matrix += np.triu(transition_matrix).T

        univariate_importances = [linkage_metric.univariate_importances[(var, val)]
                                  for var, card in enumerate(ss.cardinalities)
                                  for val in range(card)]

        np.fill_diagonal(transition_matrix, univariate_importances)

        # normalise the columns
        for column_index in range(table_size):
            column = transition_matrix[:, column_index]
            column -= np.min(column)
            transition_matrix[:, column_index] = column / np.sum(column)

        return transition_matrix



def test_local_linkage():
    problem = RoyalRoad(4, 4)

    pRef = problem.get_reference_population(sample_size=10000)

    n = problem.search_space.amount_of_parameters
    linkage_table = np.zeros(shape=(n, n), dtype=float)

    #linkage_metric = LocalLinkage()
    #linkage_metric.set_pRef(pRef)

    linkage_metric = BivariateVariance()
    linkage_metric.set_pRef(pRef)
    for var_a, var_b in itertools.product(range(n), range(n)):
        ps = PS.empty(problem.search_space)
        ps = ps.with_fixed_value(var_a, 1)
        ps = ps.with_fixed_value(var_b, 1)
        linkage = linkage_metric.get_single_score(ps)
        linkage_table[var_a, var_b] = linkage



    random_pss = [PS.random(half_chance_star=True, search_space=problem.search_space) for _ in range(10000)]
    random_pss = [EvaluatedPS(ps.values, metric_scores=(linkage_metric.get_single_score(ps),))
                  for ps in random_pss]
    for ps in random_pss:
        ps.aggregated_score = ps.metric_scores[0]

    random_pss.sort(key=lambda x: x.metric_scores, reverse=True)
    random_pss = random_pss[:100]


    sampler = MarkovianSampler.from_linkage_metric(linkage_metric, iterations=4)

    print("All done!")
