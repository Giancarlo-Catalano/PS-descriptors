import itertools
import random
from collections import defaultdict
from typing import Tuple

import numpy as np

from BenchmarkProblems.BinVal import BinVal
from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from BenchmarkProblems.RoyalRoad import RoyalRoad
from BenchmarkProblems.WithLinkage import WithLinkage
from Core.EvaluatedFS import EvaluatedFS
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.SearchSpace import SearchSpace


def select_n_from_pRef(pRef: PRef, n: int = 2) -> list[EvaluatedFS]:
    random_indexes = [random.randrange(pRef.sample_size) for _ in range(n)]

    def get_solution_from_index(index: int) -> EvaluatedFS:
        return EvaluatedFS(FullSolution(pRef.full_solution_matrix[index]),
                           fitness=float(pRef.fitness_array[index]))

    return list(map(get_solution_from_index, random_indexes))


def binary_tournament_selection_on_pRef(pRef: PRef) -> (EvaluatedFS, EvaluatedFS):
    first, second = select_n_from_pRef(pRef, 2)
    if first < second:
        first, second = second, first
    return first, second


def get_variable_importance(pRef: PRef, tournaments: int = 1000) -> (list[int], list[np.ndarray]):
    search_space = pRef.search_space

    def make_win_table_for_var(index: int):
        cardinality = search_space.cardinalities[index]
        return np.zeros(shape=(cardinality, cardinality), dtype=int)

    win_tables = [make_win_table_for_var(index)
                  for index in range(search_space.amount_of_parameters)]

    def register_one_tournament(input_win_tables: list[np.ndarray]) -> None:
        winner, loser = binary_tournament_selection_on_pRef(pRef)
        for win_table, winner_value, loser_value in zip(input_win_tables, winner.values, loser.values):
            win_table[winner_value][loser_value] += 1

    for _ in range(tournaments):
        register_one_tournament(win_tables)

    def remove_diagonal(win_table: np.ndarray):
        np.fill_diagonal(win_table, 0)

    for win_table in win_tables:
        remove_diagonal(win_table)

    def get_total_score_in_table(win_table: np.ndarray):
        return np.sum(np.abs(win_table - win_table.T))

    variable_importances = list(map(get_total_score_in_table, win_tables))
    return variable_importances, win_tables


def get_linkage_information(pRef: PRef, tournaments: int = 1000) -> np.ndarray:
    variable_importances, univariate_win_tables = get_variable_importance(pRef, tournaments)
    search_space = pRef.search_space

    variable_importances = np.array(variable_importances) / tournaments

    expectation_table = np.zeros(shape=(search_space.amount_of_parameters,
                                        search_space.amount_of_parameters),
                                 dtype=float)

    def get_prediction_table(win_table) -> np.ndarray:
        return (win_table - win_table.T) / tournaments

    prediction_tables = list(map(get_prediction_table, univariate_win_tables))

    def register_one_tournament(input_expectation_table) -> None:
        winner, loser = binary_tournament_selection_on_pRef(pRef)
        for index_a in range(search_space.amount_of_parameters):
            w_v_a = winner.values[index_a]
            l_v_a = loser.values[index_a]

            pred_a = prediction_tables[index_a][w_v_a][l_v_a]
            for index_b in range(index_a + 1, search_space.amount_of_parameters):
                w_v_b = winner.values[index_b]
                l_v_b = loser.values[index_b]
                pred_b = prediction_tables[index_b][w_v_b][l_v_b]

                input_expectation_table[index_a][index_b] += pred_a * pred_b

    for _ in range(tournaments):
        register_one_tournament(expectation_table)

    expectation_table += expectation_table.T

    return expectation_table


def test_variable_importance():
    problem = WithLinkage()
    pRef = problem.get_reference_population(sample_size=1000)

    variable_importances, _ = get_variable_importance(pRef)
    print(variable_importances)


def test_interaction():
    # problem = EfficientBTProblem.from_default_files()
    problem = WithLinkage()
    pRef = problem.get_reference_population(sample_size=10000)
    expectation_table = get_linkage_information(pRef, tournaments=10000)
    print(expectation_table)


class ImportanceForSubset:
    subset: list[int]
    win_dict: dict[(tuple, tuple), float]

    def __init__(self,
                 subset: list[int]):
        self.subset = subset
        self.win_dict = defaultdict(float)

    def register_win(self, winner: EvaluatedFS, loser: EvaluatedFS) -> None:
        winning_values = tuple(winner.values[list(self.subset)])
        losing_values = tuple(loser.values[list(self.subset)])

        self.win_dict[(winning_values, losing_values)] += 1

    def normalise(self) -> None:
        total = sum(self.win_dict.values())
        for key in self.win_dict:
            self.win_dict[key] /= total

    def get_importance(self) -> float:
        used_keys: set[(tuple, tuple)] = set()
        current_sum = 0


        # the next 8 lines are to add missing reverse keys
        keys_to_add = list()
        for key in self.win_dict:
            reverse_key = (key[1], key[0])
            if reverse_key not in self.win_dict:
                keys_to_add.append(reverse_key)

        for new_key in keys_to_add:
            self.win_dict[new_key] = 0

        # this is the interesting part
        accuracies = []
        for key in self.win_dict:
            if key not in used_keys:
                reverse_key = (key[1], key[0])
                used_keys.add(key)
                used_keys.add(reverse_key)
                current_sum += abs(self.win_dict[key] - self.win_dict[reverse_key])

                ab = self.win_dict[key]
                ba = self.win_dict[reverse_key]
                accuracy = max(ab, ba) / (ab+ba) # debug
                accuracies.append(accuracy)


        tournament_samples = sum(self.win_dict.values())

        average_accuracy = np.max(accuracies) # debug
        return average_accuracy # debug
        # return current_sum / tournament_samples





class MultivariateImportance:
    search_space: SearchSpace
    n: int

    win_dicts: dict[tuple[int], ImportanceForSubset]


    def get_iterable_for_subsets(self):
        return itertools.combinations_with_replacement(range(self.search_space.amount_of_parameters), r=self.n)
    def __init__(self,
                 search_space: SearchSpace,
                 n: int):
        self.search_space = search_space
        self.n = n
        self.win_dicts = {subset: ImportanceForSubset(subset)
                          for subset in self.get_iterable_for_subsets()}

    @classmethod
    def tournament_sample_from_pRef(cls, pRef: PRef) -> (EvaluatedFS, EvaluatedFS):
        return binary_tournament_selection_on_pRef(pRef)

    def train_from_pRef(self,
                        pRef: PRef,
                        tournament_samples: int) -> None:
        for _ in range(tournament_samples):
            winner, loser = self.tournament_sample_from_pRef(pRef)
            self.register_tournament(winner, loser)

        # for subset, win_dict in self.win_dicts.items():
        #     win_dict.normalise()

    def register_tournament(self, winner: EvaluatedFS, loser: EvaluatedFS) -> None:
        for win_dict in self.win_dicts.values():
            win_dict.register_win(winner, loser)

    def get_importances(self) -> dict[tuple[int], float]:
        return {subset: win_dict.get_importance()
                for subset, win_dict in self.win_dicts.items()}


    def show_univariate_importances(self) -> np.ndarray:
        assert(self.n == 1)

        result = np.zeros(shape=self.search_space.amount_of_parameters, dtype=float)
        for var_index in range(self.search_space.amount_of_parameters):
            result[var_index] = self.win_dicts[(var_index,)].get_importance()

        return result

    def get_importance_array(self) -> np.ndarray:
        # assert(self.n == 2)

        result = np.zeros(shape=[self.search_space.amount_of_parameters for _ in range(self.n)], dtype=float)
        iterable = [range(self.search_space.amount_of_parameters) for _ in range(self.n)]
        for subset in itertools.combinations(*iterable):
            result[subset] = self.win_dicts[subset].get_importance()

        return result



def test_multivariate_importance():
    problem = RoyalRoad(2, 4)
    pRef = problem.get_reference_population(sample_size=1000)
    importance_calculator = MultivariateImportance(search_space=problem.search_space, n=2)
    importance_calculator.train_from_pRef(pRef, tournament_samples=10000)
    print(importance_calculator.get_importances())
    bivariate_importances = importance_calculator.get_importance_array()


    univariate_importance_calculator = MultivariateImportance(search_space=problem.search_space, n=1)
    univariate_importance_calculator.train_from_pRef(pRef, tournament_samples=100000)
    univariate_importances = univariate_importance_calculator.get_importance_array()


    adjusted_bivariate = bivariate_importances - univariate_importances - univariate_importances.reshape((-1, 1))

    def make_colours_more_visible(table) -> np.ndarray:
        without_lowr_tri = np.triu(table, k=1)
        lower_tri_filled = without_lowr_tri + without_lowr_tri.T
        return lower_tri_filled + np.identity(table.shape[0])*np.average(lower_tri_filled)

    to_show = bivariate_importances + bivariate_importances.T + np.identity(problem.search_space.amount_of_parameters)*univariate_importances



    print(make_colours_more_visible(adjusted_bivariate))