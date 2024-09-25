import heapq
import random
from typing import Optional

import numpy as np
from xcs.scenarios import Scenario

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.EvaluatedFS import EvaluatedFS
from Core.FullSolution import FullSolution
from Core.PRef import PRef


class GenericSolutionDifferenceScenario(Scenario):
    input_size: int
    possible_actions: tuple
    initial_training_cycles: int
    remaining_cycles: int

    original_problem: BenchmarkProblem

    current_winner: Optional[EvaluatedFS]
    current_loser: Optional[EvaluatedFS]

    solution_pairs_to_consider: Optional[list[(EvaluatedFS, EvaluatedFS)]]
    current_solution_pair_index: int

    verbose: bool

    def __init__(self,
                 initial_training_cycles: int,
                 original_problem: BenchmarkProblem,
                 verbose: bool):
        self.original_problem = original_problem

        self.input_size = self.original_problem.search_space.amount_of_parameters
        self.possible_actions = (True,)
        self.initial_training_cycles = initial_training_cycles
        self.remaining_cycles = initial_training_cycles

        self.solution_pairs_to_consider = None

        # reset the internal index
        self.current_solution_pair_index = -1
        self.current_winner = None
        self.current_loser = None

        self.verbose = verbose

    #@abc.abstractmethod
    def get_solution_pairs(self, pRef: PRef, amount_required: int) -> list[(EvaluatedFS, EvaluatedFS)]:
        raise Exception("The class needs to implement the method .get_solution_pairs")

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
        self.current_solution_pair_index += 1
        self.current_winner, self.current_loser = self.solution_pairs_to_consider[self.current_solution_pair_index]

        return (self.current_winner, self.current_loser)

    def execute(self, is_in_winner: bool) -> float:
        # this returns the payoff for the action, it should be in the range [0, 1]
        # this function shouldn't actually get used, but it's here just in case
        return bool(is_in_winner)


class SolutionDifferenceScenario(GenericSolutionDifferenceScenario):
    """ This class is the main interface between the optimisation problem and the LCS"""
    """This class is not actually that interesting"""

    def __init__(self,
                 original_problem: BenchmarkProblem,
                 pRef: PRef,
                 training_cycles: int = 1000,
                 verbose: bool = False):

        super().__init__(initial_training_cycles=training_cycles,
                         original_problem=original_problem,
                         verbose=verbose)

        self.solution_pairs_to_consider = self.get_solution_pairs(pRef=pRef, amount_required=training_cycles)

    @classmethod
    def get_solution_pairs(cls, pRef: PRef, amount_required: int) -> list[
        (EvaluatedFS, EvaluatedFS)]:

        # remove duplicates
        all_solutions = pRef.get_evaluated_FSs()

        all_solutions.sort(reverse=True)

        all_pairs = []

        def add_layer(layer_index: int):
            if layer_index * 2 <= len(all_solutions):
                winner_index = 0
                loser_index = layer_index + 1
            else:
                winner_index = layer_index - len(all_solutions)
                loser_index = len(all_solutions) - 1

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


class RandomPairsScenario(GenericSolutionDifferenceScenario):
    def __init__(self,
                 original_problem: BenchmarkProblem,
                 pRef: PRef,
                 training_cycles: int = 1000,
                 verbose: bool = False):

        super().__init__(initial_training_cycles=training_cycles,
                         original_problem=original_problem,
                         verbose=verbose)

        self.solution_pairs_to_consider = self.get_solution_pairs(pRef=pRef, amount_required=training_cycles)

    @classmethod
    def get_solution_pairs(cls, pRef: PRef, amount_required: int) -> list[
        (EvaluatedFS, EvaluatedFS)]:

        all_solutions = pRef.get_evaluated_FSs()

        all_pairs = []

        while len(all_pairs) < amount_required:
            first = random.choice(all_solutions)
            second = random.choice(all_solutions)
            if first > second:
                all_pairs.append((first, second))
            elif second > first:
                all_pairs.append((second, first))
            # else it doesn't get added

        return all_pairs


class OneAtATimeSolutionDifferenceScenario(GenericSolutionDifferenceScenario):
    current_solution: Optional[EvaluatedFS]
    pRef: PRef

    def __init__(self,
                 original_problem: BenchmarkProblem,
                 pRef: PRef,
                 training_cycles: int = 1000,
                 verbose: bool = False):
        self.current_solution = None
        self.pRef = pRef

        super().__init__(initial_training_cycles=training_cycles,
                         original_problem=original_problem,
                         verbose=verbose)

    def set_solution_to_investigate(self, to_investigate: EvaluatedFS):
        self.reset()  # reset the pair index and put the training cycles to 0
        self.current_solution = to_investigate
        self.solution_pairs_to_consider = self.get_solution_pairs(self.pRef, self.initial_training_cycles)
        #self.initial_training_cycles = min(len(self.solution_pairs_to_consider), self.initial_training_cycles)
        self.remaining_cycles = min(len(self.solution_pairs_to_consider), self.initial_training_cycles)

    def get_solution_pairs(self, pRef: PRef, amount_required: int) -> list[(EvaluatedFS, EvaluatedFS)]:
        """returns the pairs in order of similarity (the most similar pairs are first)"""
        difference_matrix = pRef.full_solution_matrix != self.current_solution.values
        difference_counts = np.sum(difference_matrix, axis=1)

        row_number_and_differences = [(row_index, difference)
                                      for row_index, difference, fitness_diff in zip(range(len(difference_counts)), difference_counts, self.pRef.fitness_array-self.current_solution.fitness)
                                      if difference > 0
                                      if fitness_diff != 0]
        row_number_and_differences = heapq.nsmallest(n=amount_required,
                                                     iterable = row_number_and_differences,
                                                     key=utils.second)

        def get_pair_at_index(index: int) -> (EvaluatedFS, EvaluatedFS):
            other_solution = EvaluatedFS(full_solution=FullSolution(pRef.full_solution_matrix[index]),
                                         fitness=pRef.fitness_array[index])
            if self.current_solution > other_solution:
                return self.current_solution, other_solution
            else:
                return other_solution, self.current_solution

        return [get_pair_at_index(index)
                for index, _ in row_number_and_differences[:self.initial_training_cycles]]
