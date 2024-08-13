import itertools
import random
from typing import Optional

import numpy as np

import utils
from BenchmarkProblems.RoyalRoad import RoyalRoad
from BenchmarkProblems.RoyalRoadWithOverlaps import RoyalRoadWithOverlaps
from BenchmarkProblems.Trapk import Trapk
from Core.PRef import PRef
from Core.PS import PS, STAR
from Core.PSMetric.Metric import Metric

type LinkageHolder = dict[tuple, float]

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
            return self.global_variance - self.get_variance(ps)

        univariate_pss = [PS.empty(self.pRef.search_space).with_fixed_value(var, val)
                          for var, val in enumerate(ps.values)
                          if val != STAR]

        univariate_variances = [self.get_variance(u_ps) for u_ps in univariate_pss]
        local_variance = self.get_variance(ps)

        return self.global_variance - np.sum(univariate_variances) + local_variance


def test_local_linkage():
    problem = RoyalRoadWithOverlaps(5, 5, 20)
    pRef = problem.get_reference_population(sample_size=10000)

    n = problem.search_space.amount_of_parameters
    linkage_table = np.zeros(shape=(n, n), dtype=float)

    #linkage_metric = LocalLinkage()
    #linkage_metric.set_pRef(pRef)

    linkage_metric = JustVariance(pRef)
    for var_a, var_b in (itertools.combinations(range(n), r=2)):
        ps = PS.empty(problem.search_space)
        ps = ps.with_fixed_value(var_a, 1)
        ps = ps.with_fixed_value(var_b, 1)
        linkage = linkage_metric.get_single_score(ps)
        linkage_table[var_a, var_b] = linkage

    linkage_table += linkage_table.T

    print("All done!")