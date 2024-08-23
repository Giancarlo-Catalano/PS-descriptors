import random

from xcs.bitstrings import BitString
from xcs.scenarios import Scenario

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.EvaluatedFS import EvaluatedFS
from Core.PRef import PRef


class XCSProblemTopAndBottom(Scenario):
    """ This class is the main interface between the optimisation problem and the LCS"""
    """ It takes the solutions generated while solving the optimisation problem (the pRef),
    and shows in alternation the best and worst"""

    """This class is not actually that interesting"""


    input_size: int
    possible_actions: tuple
    initial_training_cycles: int

    original_problem: BenchmarkProblem

    all_solutions: list[EvaluatedFS]

    currently_showing_a_good_solution: bool
    current_index: int
    current_solution: EvaluatedFS

    met_the_mid_point: bool

    remaining_cycles: int
    total_training_cycles: int
    tail_size: int

    def __init__(self,
                 original_problem: BenchmarkProblem,
                 pRef: PRef,
                 training_cycles: int = 1000,
                 tail_size: int = 1000):
        # store the original optimisation problem, for convenience
        self.original_problem = original_problem

        self.input_size = self.original_problem.search_space.amount_of_parameters
        self.possible_actions = (True, False)
        self.initial_training_cycles = training_cycles
        self.remaining_cycles = training_cycles

        self.all_solutions = pRef.get_evaluated_FSs()
        self.all_solutions = list(set(self.all_solutions))  # remove duplicates
        self.all_solutions.sort(reverse=True)  # sort by fitness

        # reset the internal index
        self.current_index = -1
        self.current_solution = None
        self.currently_showing_a_good_solution = False
        self.met_the_mid_point = False

        self.remaining_cycles = training_cycles
        self.total_training_cycles = training_cycles
        self.tail_size = tail_size

    @property
    def is_dynamic(self):
        return False

    def get_possible_actions(self):
        return self.possible_actions

    def reset_internal_index(self):
        self.current_index = -1
        self.current_solution = None
        self.currently_showing_a_good_solution = False
        self.met_the_mid_point = False

    def reset(self):
        self.reset_internal_index()
        self.remaining_cycles = self.total_training_cycles

    def more(self):
        return self.remaining_cycles > 0

    def obtain_new_solution(self):
        if self.currently_showing_a_good_solution:
            new_solution = self.all_solutions[-(self.currently_showing_a_good_solution + 1)]
            self.currently_showing_a_good_solution = False
        else:
            self.current_index += 1
            new_solution = self.all_solutions[self.current_index]
            self.currently_showing_a_good_solution = True

        if self.current_solution is not None and new_solution.fitness == self.current_solution.fitness:
            print("Met the midpoint, resetting the internal index")
            self.reset_internal_index()

        if self.current_index > self.tail_size:
            print("Reached the end of extreme solutions, resetting the index")
            self.reset_internal_index()

        self.current_solution = new_solution

    def sense(self):
        # the tutorial stores the "fitness" of the solution as well..?
        self.remaining_cycles -= 1

        if (self.remaining_cycles + 1) % 100 == 0:
            print(f"remaining: {self.remaining_cycles}")

        self.obtain_new_solution()  # this updates also the "action", ie is_current_better
        bitstring = BitString(self.current_solution.values)

        return bitstring

    def get_current_outcome(self):
        return self.currently_showing_a_good_solution

    def execute(self, action):
        return float(self.currently_showing_a_good_solution == action)
