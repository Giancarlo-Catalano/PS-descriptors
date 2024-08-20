import random

from xcs.bitstrings import BitString
from xcs.scenarios import Scenario

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.EvaluatedFS import EvaluatedFS
from Core.PRef import PRef


class XSCProblemTournamenter(Scenario):

    input_size: int
    possible_actions: tuple
    initial_training_cycles: int
    remaining_cycles: int

    original_problem: BenchmarkProblem

    all_solutions: list[EvaluatedFS]
    previous_solution: EvaluatedFS
    current_solution: EvaluatedFS


    def randomly_pick_solution(self) -> EvaluatedFS:
        return random.choice(self.all_solutions)

    def __init__(self,
                 original_problem: BenchmarkProblem,
                 pRef: PRef,
                 training_cycles: int = 1000):
        self.original_problem = original_problem

        self.input_size = self.original_problem.search_space.amount_of_parameters
        self.possible_actions = (True, False)
        self.initial_training_cycles = training_cycles
        self.remaining_cycles = training_cycles

        self.all_solutions = pRef.get_evaluated_FSs()
        self.previous_solution = self.randomly_pick_solution()
        self.current_solution = self.randomly_pick_solution()


    @property
    def is_dynamic(self):
        return False

    def get_possible_actions(self):
        return self.possible_actions

    def reset(self):
        self.remaining_cycles = self.initial_training_cycles

    def more(self):
        return self.remaining_cycles > 0

    def sense(self):
        # the tutorial stores the "fitness" of the solution as well..?
        bitstring = BitString(self.current_solution.values)
        return bitstring

    def execute(self, action):
        self.remaining_cycles -= 1

        is_better = self.current_solution > self.previous_solution

        new_solution = self.randomly_pick_solution()
        self.previous_solution = self.current_solution
        self.current_solution = new_solution
        return is_better