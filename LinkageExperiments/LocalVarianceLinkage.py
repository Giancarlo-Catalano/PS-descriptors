import itertools
from math import floor
from typing import Optional

import numpy as np

from BenchmarkProblems.RoyalRoad import RoyalRoad
from BenchmarkProblems.Trapk import Trapk
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS, STAR
from Core.SearchSpace import SearchSpace
from Explanation.PRefManager import PRefManager


class BivariateLinkage:
    pRef: Optional[PRef]

    def __init__(self):
        self.pRef = None

    def set_pRef(self, pRef: PRef) -> None:
        self.pRef = pRef

    @property
    def search_space(self) -> SearchSpace:
        return self.pRef.search_space

    @property
    def n_vars(self) -> int:
        return self.search_space.amount_of_parameters

    def get_bivariate_linkage_between_vars(self, var_a: int, var_b: int) -> float:
        raise NotImplemented

    def get_univariate_linkage_of_var(self, var: int) -> float:
        raise NotImplemented

    def every_var_iterator(self):
        return range(self.n_vars)

    def every_var_pair_iterator(self):
        return itertools.combinations(range(self.n_vars), r=2)

    def get_atomicity(self, ps: PS) -> float:
        fixed_vars = ps.get_fixed_variable_positions()

        def weakest_internal_linkage_for(var) -> float:
            return min(self.get_bivariate_linkage_between_vars(var, other)
                       for other in fixed_vars if other != var)

        if len(fixed_vars) > 1:
            weakest_links = np.array([weakest_internal_linkage_for(var) for var in fixed_vars])
            return np.average(weakest_links)
        elif len(fixed_vars) == 1:
            var = fixed_vars[0]
            return self.get_univariate_linkage_of_var(var)
        else:
            return 0

    def get_independence(self, ps: PS) -> float:
        fixed_vars = ps.get_fixed_variable_positions()
        unfixed_vars = [index for index, val in enumerate(ps.values) if val == STAR]

        def strongest_external_linkage_for(var) -> float:
            return max(self.get_bivariate_linkage_between_vars(var, other)
                       for other in unfixed_vars)

        if (len(unfixed_vars) > 0) and (len(fixed_vars) > 0):  # maybe this should be zero?
            strongest_links = np.array([strongest_external_linkage_for(var) for var in fixed_vars])
            return np.average(strongest_links)
        elif len(fixed_vars) == 1:
            var = fixed_vars[0]
            return strongest_external_linkage_for(var)
        else:
            return 0


class LocalVarianceLinkage(BivariateLinkage):
    similarity_threshold: float

    linkage_table: Optional[np.ndarray]
    current_solution: Optional[FullSolution]

    def __init__(self, similarity_threshold: float = 0.5):
        super().__init__()
        self.linkage_table = None
        self.current_solution = None

        self.similarity_threshold = similarity_threshold

    def set_solution(self, solution: FullSolution) -> None:
        self.current_solution = solution

        similarity_rows, fitnesses = self.get_close_solution_data()
        univariate_variances, bivariate_variances = self.get_linkage_structures(similarity_rows, fitnesses)
        self.linkage_table = self.get_linkage_tables_from_variances(univariate_variances,
                                                                    bivariate_variances)

    def get_close_solution_data(self) -> (np.ndarray, np.ndarray):
        """ finds all the solutions that are similar enough to the current solution,
            returns a matrix of their difference from the solution (booleans),
            and their fitnesses"""

        assert (self.current_solution is not None)
        assert (self.pRef is not None)

        similarity_matrix = self.pRef.full_solution_matrix == self.current_solution.values
        similarity_counts = np.sum(similarity_matrix, axis=1)
        similarity_count_threshold = round(self.similarity_threshold * self.n_vars)

        eligible_rows = similarity_counts >= similarity_count_threshold

        rows_returned = similarity_matrix[eligible_rows]
        fitnesses_returned = self.pRef.fitness_array[eligible_rows]

        return rows_returned, fitnesses_returned

    def get_linkage_structures(self,
                               similarity_rows: np.ndarray,
                               fitnesses: np.ndarray) -> (dict[int, float], dict[(int, int), float]):
        univariate_fitnesses: dict[int, list[float]] = {var: [] for var in range(self.n_vars)}
        bivariate_fitnesses: dict[(int, int), list[float]] = {(var_a, var_b): []
                                                              for var_a, var_b in self.every_var_pair_iterator()}

        def register_row_univariate(similarity_row: np.ndarray, fitness: float) -> None:
            for index, value in enumerate(similarity_row):
                if value:
                    univariate_fitnesses[index].append(fitness)

        def register_row_bivariate(similarity_row: np.ndarray, fitness: float) -> None:
            similarities = [bool(value) for value in similarity_row]  # because bool_ values behave weirdly
            for var_a, var_b in self.every_var_pair_iterator():
                if similarities[var_a] and similarities[var_b]:
                    bivariate_fitnesses[(var_a, var_b)].append(fitness)

        def register_row(similarity_row: np.ndarray, fitness: float) -> None:
            register_row_univariate(similarity_row, fitness)
            register_row_bivariate(similarity_row, fitness)

        for row, fitness in zip(similarity_rows, fitnesses):
            register_row(row, fitness)

        univariate_variances = {var: self.get_variance_of_fitnesses(observations)
                                for var, observations in univariate_fitnesses.items()}

        bivariate_variances = {var_pair: self.get_variance_of_fitnesses(observations)
                               for var_pair, observations in bivariate_fitnesses.items()}

        return univariate_variances, bivariate_variances

    @classmethod
    def get_variance_of_fitnesses(cls, fitnesses: list[float]) -> Optional[float]:
        if len(fitnesses) < 2:
            return None
        else:
            return np.var(fitnesses)

    def get_linkage_tables_from_variances(self,
                                          univariate_variances: dict[int, float],
                                          bivariate_variances: dict[(int, int), float]) -> np.ndarray:
        def get_bivariate_linkage(var_a: int, var_b: int) -> float:
            """assumes var_a < var_b"""
            return bivariate_variances[(var_a, var_b)] - univariate_variances[var_a] - univariate_variances[
                var_b]  # TODO investigate

        bivariate_table = np.zeros(shape=(self.n_vars, self.n_vars))
        for var_a, var_b in self.every_var_pair_iterator():
            bivariate_table[var_a, var_b] = get_bivariate_linkage(var_a, var_b)

        # the next steps are just to populate the diagonal
        bivariate_table += bivariate_table.T

        # then we get the average of each column, excluding the middle diagonal
        univariate_importances = np.sum(bivariate_table, axis=0) / (self.n_vars - 1)
        np.fill_diagonal(bivariate_table, univariate_importances)

        return bivariate_table

    def get_bivariate_linkage_between_vars(self, var_a: int, var_b: int) -> float:
        return self.linkage_table[var_a, var_b]

    def get_univariate_linkage_of_var(self, var: int) -> float:
        return self.linkage_table[var, var]


def test_local_variance_linkage():
    # optimisation_problem = RoyalRoad(4, 4)
    optimisation_problem = Trapk(6, 5)
    pRef = PRefManager.generate_pRef(problem=optimisation_problem,
                                     sample_size=10000,  # these are the Solution evaluations
                                     which_algorithm="uniform",
                                     verbose=False)

    metric = LocalVarianceLinkage()
    metric.set_pRef(pRef)

    solutions_to_check = [FullSolution([1 for _ in range(metric.n_vars)]),
                          FullSolution([0 for _ in range(metric.n_vars)])]
    # solutions_to_check.append(FullSolution([0, 0, 0, 0, 0]+[0, 0, 0, 0, 1]+[0, 0, 0, 1, 1]+[0, 0, 1, 1, 1]+[0, 1, 1, 1, 1]+[1, 1, 1, 1, 1]))

    for solution in solutions_to_check:
        metric.set_solution(solution)
        linkage_table = metric.linkage_table
        print("You should be debugging right now")


test_local_variance_linkage()
