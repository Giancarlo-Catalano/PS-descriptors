import heapq
import itertools
import random
from math import ceil

import numpy as np
from xcs.bitstrings import BitString
from xcs.scenarios import Scenario

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.EvaluatedFS import EvaluatedFS
from Core.PRef import PRef


class SolutionDifferenceScenario(Scenario):
    """ This class is the main interface between the optimisation problem and the LCS"""
    """ It takes the solutions generated while solving the optimisation problem (the pRef),
    and shows in alternation the best and worst"""

    """This class is not actually that interesting"""

    input_size: int
    possible_actions: tuple
    initial_training_cycles: int
    remaining_cycles: int

    original_problem: BenchmarkProblem

    all_solutions: list[EvaluatedFS]

    current_winner: EvaluatedFS
    current_loser: EvaluatedFS

    solution_pairs_to_consider: list[(EvaluatedFS, EvaluatedFS)]
    current_solution_pair_index: int

    verbose: bool

    def __init__(self,
                 original_problem: BenchmarkProblem,
                 pRef: PRef,
                 training_cycles: int = 1000,
                 verbose: bool = False):
        # store the original optimisation problem, for convenience
        self.original_problem = original_problem

        self.input_size = self.original_problem.search_space.amount_of_parameters
        self.possible_actions = (True,)
        self.initial_training_cycles = training_cycles
        self.remaining_cycles = training_cycles

        self.solution_pairs_to_consider = self.get_solution_pairs(all_solutions=pRef.get_evaluated_FSs(),
                                                                  amount_required = training_cycles)

        # reset the internal index
        self.current_solution_pair_index = -1
        self.current_winner = None
        self.current_loser = None

        self.verbose = verbose

    @classmethod
    def get_solution_pairs(cls, all_solutions: list[EvaluatedFS], amount_required: int) -> list[
        (EvaluatedFS, EvaluatedFS)]:

        # remove duplicates
        all_solutions = list(set(all_solutions))

        all_solutions.sort(reverse=True)

        all_pairs = []

        def add_layer(layer_index: int):
            if layer_index * 2 <= len(all_solutions):
                winner_index = 0
                loser_index = layer_index + 1
            else:
                winner_index = layer_index-len(all_solutions)
                loser_index = len(all_solutions)-1

            while loser_index - winner_index > 0:
                winner = all_solutions[winner_index]
                loser = all_solutions[loser_index]

                if winner.fitness != loser.fitness:
                    all_pairs.append((winner, loser))
                winner_index += 1
                loser_index -= 1

        for layer in range(2 * len(all_solutions) - 3):
            if len(all_pairs) >= amount_required:
                break
            add_layer(layer)

        def solution_difference(sol_pair: (EvaluatedFS, EvaluatedFS)) -> int:
            sol_a, sol_b = sol_pair
            return int(np.sum(sol_a.values != sol_b.values))

        all_pairs.sort(key=solution_difference)

        # def rearrange_pair_if_necessary(sol_pair: (EvaluatedFS, EvaluatedFS)) -> (EvaluatedFS, EvaluatedFS):
        #     a, b = sol_pair
        #     return sol_pair if a > b else (b, a)

        # return list(map(rearrange_pair_if_necessary, all_pairs))

        return all_pairs

    @property
    def is_dynamic(self):
        return False

    def get_possible_actions(self):
        return self.possible_actions

    def reset_internal_index(self):
        # reset the internal index
        self.current_solution_pair_index = -1
        self.current_winner = None
        self.current_loser = None

    def reset(self):
        self.reset_internal_index()
        self.remaining_cycles = self.initial_training_cycles

    def more(self):
        return self.remaining_cycles > 0

    def obtain_new_solution_pair(self):
        self.current_solution_pair_index += 1
        self.current_winner, self.current_loser = self.solution_pairs_to_consider[self.current_solution_pair_index]

    def sense(self) -> (EvaluatedFS, EvaluatedFS):
        # return the current solution
        # the tutorial stores the "fitness" of the solution as well..?
        self.remaining_cycles -= 1
        self.current_solution_pair_index+=1
        self.current_winner, self.current_loser = self.solution_pairs_to_consider[self.current_solution_pair_index]

        if self.verbose and ((self.remaining_cycles + 1) % 100 == 0):
            print(f"remaining: {self.remaining_cycles}")

        return (self.current_winner, self.current_loser)

    def execute(self, is_in_winner: bool) -> float:
        # this returns the payoff for the action, it should be in the range [0, 1]

        # TODO make a comment here when mature
        return bool(is_in_winner)
