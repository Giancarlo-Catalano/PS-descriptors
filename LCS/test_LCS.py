from typing import Optional

import xcs

import logging

from xcs.bitstrings import BitString
from xcs.scenarios import ScenarioObserver, MUXProblem, Scenario

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from BenchmarkProblems.RoyalRoad import RoyalRoad
from Core.FullSolution import FullSolution


""" this file is just to check if XCS all works as intended"""

def test_if_library_works():
    logging.root.setLevel(logging.INFO)
    xcs.test()


def test_manually():
    scenario = ScenarioObserver(MUXProblem(50000))
    algorithm = xcs.XCSAlgorithm()

    algorithm.exploration_probability = .1
    algorithm.discount_factor = 0
    algorithm.do_ga_subsumption = True
    algorithm.do_action_set_subsumption = True

    model = algorithm.new_model(scenario)

    # training
    model.run(scenario, learn=True)

    print(model)


class XSCProblemWrapper(Scenario):
    input_size: int
    possible_actions: tuple
    initial_training_cycles: int
    remaining_cycles: int

    original_problem: BenchmarkProblem

    last_fitness = Optional[float]

    def __init__(self,
                 original_problem: BenchmarkProblem,
                 training_cycles: int = 1000):
        self.original_problem = original_problem

        self.input_size = self.original_problem.search_space.amount_of_parameters
        self.possible_actions = (True, False)
        self.initial_training_cycles = training_cycles
        self.remaining_cycles = training_cycles

        self.last_fitness = None

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
        full_solution = FullSolution.random(self.original_problem.search_space)
        bitstring = BitString(full_solution.values)
        fitness = self.original_problem.fitness_function(full_solution)
        self.last_fitness = fitness
        return bitstring

    def execute(self, action):
        self.remaining_cycles -= 1

        fitness_threshold = 0
        fitness_is_good = self.last_fitness > fitness_threshold
        return action == fitness_is_good


def test_custom_problem():
    original_problem = RoyalRoad(3, 4)
    wrapped_problem = XSCProblemWrapper(original_problem, training_cycles=50000)
    scenario = ScenarioObserver(wrapped_problem)
    algorithm = xcs.XCSAlgorithm()

    algorithm.exploration_probability = .5
    algorithm.discount_factor = 0
    algorithm.do_ga_subsumption = True
    algorithm.do_action_set_subsumption = True

    model = algorithm.new_model(scenario)

    # training
    model.run(scenario, learn=True)

    print(model)
