import itertools
from typing import Optional

import numpy as np

import utils
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS, STAR
from Core.PSMetric.Linkage.BivariateLinkage import BivariateLinkage
from Core.PSMetric.Metric import Metric


class ValueSpecificMutualInformation(Metric):
    pRef: Optional[PRef]

    univariate_probability_table: Optional[list]
    bivariate_probability_table: Optional[list]

    linkage_dict: Optional[dict[(int, int, int, int), float]]
    univariate_dict: Optional[dict[(int, int), float]]

    def __init__(self):
        super().__init__()
        self.pRef = None
        self.univariate_probability_table = None
        self.bivariate_probability_table = None
        self.linkage_dict = None
        self.univariate_dict = None

    def __repr__(self):
        return "ValueSpecificMutualInformation"

    def set_pRef(self, pRef: PRef):
        self.pRef = pRef

        self.univariate_probability_table, self.bivariate_probability_table = self.calculate_probability_tables()
        self.linkage_dict = self.get_linkage_dict()
        self.univariate_dict = self.get_univariate_dict()

    def calculate_probability_tables(self) -> (list, list):

        amount_of_samples = 10000

        indexes = np.random.randint(self.pRef.sample_size, size=amount_of_samples)
        fitnesses = self.pRef.fitness_array[indexes]
        who_won = fitnesses > np.roll(fitnesses,
                                      1)  # note > and not >=. This is preferred because some latest_material have heavy fitness collisions
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
                for var_b in range(var_a + 1, len(solution)):
                    value_b = solution[var_b]
                    bivariate_count_table[var_a][var_b][value_a, value_b] += 1

        for sample_number, sample in enumerate(winning_solutions):
            register_solution_for_univariate(sample)
            register_solution_for_bivariate(sample)
            if sample_number % (len(winning_solutions) // 100) == 0:
                print(f"MI data gathering progress: {100 * sample_number / len(winning_solutions):.2f}%")

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
        return p_a_b * np.log(p_a_b / (p_a * p_b))

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

    def get_univariate_dict(self) -> dict[(int, int), float]:
        all_varvals = [(var, val)
                       for var, card in enumerate(self.pRef.search_space.cardinalities)
                       for val in range(card)]

        def get_linkage_unordered(var_a, val_a, var_b, val_b) -> float:
            if var_a > var_b:
                return self.linkage_dict[(var_b, val_b, var_a, val_a)]
            else:
                return self.linkage_dict[(var_a, val_a, var_b, val_b)]

        def univariate_for_varval(var, val) -> float:
            return np.average([get_linkage_unordered(var, val, o_var, o_val)
                               for o_var, o_val in all_varvals
                               if o_var != var])

        return {(var, val): univariate_for_varval(var, val)
                for (var, val) in all_varvals}

    def get_single_score(self, ps: PS) -> float:
        fixed_count = ps.fixed_count()
        if fixed_count >= 2:
            linkages = self.get_linkages_in_ps(ps)
            return np.average(linkages)
        elif fixed_count == 1:
            [fixed_position] = ps.get_fixed_variable_positions()
            return self.univariate_dict[(fixed_position, ps[fixed_position])]
        else:
            return 0

    def get_linkage_table_for_solution(self, fs: FullSolution) -> np.ndarray:
        # this is mainly for debug
        n = len(fs)
        result = np.zeros(shape=(n, n), dtype=float)
        for a, b in itertools.combinations(range(n), r=2):
            result[a, b] = self.linkage_dict[(a, fs.values[a], b, fs.values[b])]

        result += result.T

        for var, val in enumerate(fs.values):
            result[var, var] = self.univariate_dict[(var, val)]

        return result


class SolutionSpecificMutualInformation(Metric):
    pRef: Optional[PRef]
    current_solution: Optional[FullSolution]

    univariate_probability_table: Optional[list]
    bivariate_probability_table: Optional[list]

    linkage_table: Optional[np.ndarray]

    def __init__(self):
        super().__init__()
        self.current_solution = None
        self.pRef = None
        self.univariate_probability_table = None
        self.bivariate_probability_table = None
        self.linkage_table = None
        # self.univariate_dict = self.get_univariate_dict()

    def __repr__(self):
        return "SolutionSpecificMutualInformation"

    def set_pRef(self, pRef: PRef):
        self.pRef = pRef
        self.univariate_probability_table, self.bivariate_probability_table = self.calculate_probability_tables()

    def set_solution(self, solution: FullSolution):
        assert self.pRef is not None
        self.current_solution = solution
        self.linkage_table = self.get_linkage_table_for_solution()

    def calculate_probability_tables(self) -> (list, list):

        amount_of_samples = 1000

        indexes = np.random.randint(self.pRef.sample_size, size=amount_of_samples)
        fitnesses = self.pRef.fitness_array[indexes]
        who_won = fitnesses > np.roll(fitnesses,
                                      1)  # self.solution.fitness  # note > and not >=. This is preferred because some toy problems have heavy fitness collisions
        winning_indexes = indexes[who_won]
        winning_solutions = self.pRef.full_solution_matrix[winning_indexes, :]
        # wins_for_main_solution = np.sum(~who_won)

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
                for var_b in range(var_a + 1, len(solution)):
                    value_b = solution[var_b]
                    bivariate_count_table[var_a][var_b][value_a, value_b] += 1

        for sample_number, sample in enumerate(winning_solutions):
            register_solution_for_univariate(sample)
            register_solution_for_bivariate(sample)
            # if sample_number%((len(winning_solutions) // 100)+1) == 0:
            #    print(f"MI data gathering progress: {100*sample_number/len(winning_solutions):.2f}%")

        # for _ in range(wins_for_main_solution):
        #     register_solution_for_univariate(self.solution.values)
        #     register_solution_for_bivariate(self.solution.values)

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
        return p_a_b * np.log(p_a_b / (p_a * p_b))

    def get_linkage_table_for_solution(self) -> np.ndarray:
        n = self.pRef.search_space.amount_of_parameters
        result = np.zeros(shape=(n, n), dtype=float)
        for a, b in itertools.combinations(range(n), r=2):
            result[a, b] = self.mutual_information(a,
                                                   self.current_solution.values[a],
                                                   b,
                                                   self.current_solution.values[b])

        result += result.T

        sums_of_linkages = np.sum(result, axis=0)
        averages = sums_of_linkages / (n - 1)
        np.fill_diagonal(result, averages)

        return result

    def get_linkages_in_ps(self, ps: PS) -> list[float]:
        fixed_vars = ps.get_fixed_variable_positions()
        return [self.linkage_table[var_a, var_b]
                for var_a, var_b in itertools.combinations(fixed_vars, r=2)]

    def get_univariate_dict(self) -> dict[(int, int), float]:
        all_varvals = [(var, val)
                       for var, card in enumerate(self.pRef.search_space.cardinalities)
                       for val in range(card)]

        def get_linkage_unordered(var_a, val_a, var_b, val_b) -> float:
            if var_a > var_b:
                return self.linkage_dict[(var_b, val_b, var_a, val_a)]
            else:
                return self.linkage_dict[(var_a, val_a, var_b, val_b)]

        def univariate_for_varval(var, val) -> float:
            return np.average([get_linkage_unordered(var, val, o_var, o_val)
                               for o_var, o_val in all_varvals
                               if o_var != var])

        return {(var, val): univariate_for_varval(var, val)
                for (var, val) in all_varvals}

    def get_single_score(self, ps: PS) -> float:
        fixed_count = ps.fixed_count()
        if fixed_count >= 2:
            linkages = self.get_linkages_in_ps(ps)
            return np.average(linkages)
        elif fixed_count == 1:
            [fixed_position] = ps.get_fixed_variable_positions()
            return self.linkage_table[fixed_position, fixed_position]
        else:
            return 0

    def get_atomicity(self, ps: PS) -> float:
        fixed_vars = ps.get_fixed_variable_positions()

        def weakest_internal_linkage_for(var) -> float:
            return min(self.linkage_table[var, other] for other in fixed_vars if other != var)

        if len(fixed_vars) > 1:
            weakest_links = np.array([weakest_internal_linkage_for(var) for var in fixed_vars])
            return np.average(weakest_links)
        elif len(fixed_vars) == 1:
            var = fixed_vars[0]
            return self.linkage_table[var, var]
        else:
            return 0

    def get_dependence(self, ps: PS) -> float:
        fixed_vars = ps.get_fixed_variable_positions()
        unfixed_vars = [index for index, val in enumerate(ps.values) if val == STAR]

        def strongest_external_linkage_for(var) -> float:
            return max(self.linkage_table[var, other] for other in unfixed_vars)

        if (len(unfixed_vars) > 0) and (len(fixed_vars) > 0):  # maybe this should be zero?
            strongest_links = np.array([strongest_external_linkage_for(var) for var in fixed_vars])
            return np.average(strongest_links)
        elif len(fixed_vars) == 1:
            var = fixed_vars[0]
            return strongest_external_linkage_for(var)
        else:
            return 0

# use this one instead of the previous two
class FasterSolutionSpecificMutualInformation(SolutionSpecificMutualInformation,BivariateLinkage):

    def __init__(self):
        super().__init__()

    def set_pRef(self, pRef: PRef):
        self.pRef = pRef
        self.univariate_probability_table, self.bivariate_probability_table = self.calculate_probability_tables()

    def calculate_probability_tables_old(self) -> (list, list):

        solution_matrix = self.pRef.get_sorted(reverse=False).full_solution_matrix

        ss = self.pRef.search_space
        cs = ss.cardinalities
        univariate_counts = [np.zeros(card, dtype=int) for card in cs]
        bivariate_count_table = [[np.zeros((c2, c1), dtype=int)
                                  for c1 in cs]
                                 for c2 in cs]

        # from here on, rank is a high value when the solution is good, a low value when it's bad
        # ie global optima = size_of_pRef, global minima = 0 for a maximisation problem
        sample_size = self.pRef.sample_size

        def register_solution_for_univariate(solution: np.ndarray, rank: int):
            for var, value in enumerate(solution):
                univariate_counts[var][value] += rank

        def register_solution_for_bivariate(solution: np.ndarray, rank: int):
            for var_a, value_a in enumerate(solution):
                for var_b in range(var_a + 1, len(solution)):
                    value_b = solution[var_b]
                    bivariate_count_table[var_a][var_b][value_a, value_b] += rank

        for rank, sample in enumerate(solution_matrix):
            register_solution_for_univariate(sample, rank)
            register_solution_for_bivariate(sample, rank)

        def counts_to_probabilities(counts: np.ndarray):
            """ used for both arrays and matrices"""
            return (counts / np.sum(counts))

        univariate_probabilities = [counts_to_probabilities(var_counts) for var_counts in univariate_counts]
        bivariate_probabilities = [[counts_to_probabilities(bivariate_count_table[var_a][var_b])
                                    if var_b > var_a else None
                                    for var_b in range(len(cs))]
                                   for var_a in range(len(cs))]
        return univariate_probabilities, bivariate_probabilities


    def calculate_probability_tables(self) -> (list, list):

        solution_matrix = self.pRef.full_solution_matrix
        fitness_array = self.pRef.fitness_array

        ss = self.pRef.search_space
        cs = ss.cardinalities
        univariate_counts = [np.zeros(card, dtype=float) for card in cs]
        bivariate_count_table = [[np.zeros((c2, c1), dtype=float)
                                  for c1 in cs]
                                 for c2 in cs]

        # from here on, rank is a high value when the solution is good, a low value when it's bad
        # ie global optima = size_of_pRef, global minima = 0 for a maximisation problem
        sample_size = self.pRef.sample_size


        def get_rank_of_fitness(fitness: float) -> float:
            # count the amount of times a fitness like this would win in a binary tournament, if there were (n*n-1) total tournaments
            normal_wins = np.sum(fitness_array < fitness)
            tie_break_wins = np.sum(fitness == fitness) / 2
            all_wins = float(normal_wins + tie_break_wins)

            return all_wins ** 2

        def register_solution_for_univariate(solution: np.ndarray, rank: float):
            for var, value in enumerate(solution):
                univariate_counts[var][value] += rank

        def register_solution_for_bivariate(solution: np.ndarray, rank: float):
            for var_a, value_a in enumerate(solution):
                for var_b in range(var_a + 1, len(solution)):
                    value_b = solution[var_b]
                    bivariate_count_table[var_a][var_b][value_a, value_b] += rank

        for sample, fitness in zip(solution_matrix, fitness_array):
            rank = get_rank_of_fitness(fitness)
            register_solution_for_univariate(sample, rank)
            register_solution_for_bivariate(sample, rank)

        def counts_to_probabilities(counts: np.ndarray):
            """ used for both arrays and matrices"""
            return (counts / np.sum(counts))

        univariate_probabilities = [counts_to_probabilities(var_counts) for var_counts in univariate_counts]
        bivariate_probabilities = [[counts_to_probabilities(bivariate_count_table[var_a][var_b])
                                    if var_b > var_a else None
                                    for var_b in range(len(cs))]
                                   for var_a in range(len(cs))]
        return univariate_probabilities, bivariate_probabilities

    # def get_linkage_table_for_solution(self) -> np.ndarray:
    #     #  this is temporary, TODO remove me
    #     def get_linkage_between_vars(var_a: int, var_b: int) -> float:
    #         ss = self.pRef.search_space
    #         return max(self.mutual_information(var_a, value_a, var_b, value_b)
    #                    for value_a in range(ss.cardinalities[var_a])
    #                    for value_b in range(ss.cardinalities[var_b]))
    #
    #     n = self.pRef.search_space.amount_of_parameters
    #     result = np.zeros(shape=(n, n), dtype=float)
    #     for a, b in itertools.combinations(range(n), r=2):
    #         result[a, b] = get_linkage_between_vars(a, b)
    #
    #     result += result.T
    #
    #     sums_of_linkages = np.sum(result, axis=0)
    #     averages = sums_of_linkages / (n - 1)
    #     np.fill_diagonal(result, averages)
    #
    #     return result


    def get_linkage_threshold(self) -> float:
        values_to_check = self.linkage_table[np.triu_indices(len(self.current_solution), 1)]
        values_to_check = list(values_to_check)
        values_to_check.sort()

        differences = [(index, values_to_check[index] - values_to_check[index-1])
                       for index in range(len(values_to_check)//2, len(values_to_check))]
        best_index, best_difference = max(differences, key=utils.second)
        return np.average(values_to_check[(best_index-1):(best_index+1)])

