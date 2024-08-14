import random
from typing import Optional

import numpy as np
import xcs
from xcs.bitstrings import BitString, BitCondition
from xcs.scenarios import Scenario, ScenarioObserver

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from BenchmarkProblems.RoyalRoad import RoyalRoad
from BenchmarkProblems.Trapk import Trapk
from Core.EvaluatedFS import EvaluatedFS
from Core.PRef import PRef
from Core.PS import PS, STAR, contains
from LightweightLocalPSMiner.FastPSEvaluator import FastPSEvaluator
from LightweightLocalPSMiner.LocalPSSearch import local_ps_search
from PSMiners.Mining import get_history_pRef

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


def run_LCS_as_archive():
    problem = Trapk(4, 5)
    pRef = get_history_pRef(benchmark_problem=problem,
                            sample_size=10000,
                            which_algorithm="SA")

    ps_evaluator = FastPSEvaluator(pRef)

    xcs_problem = XSCProblemTournamenter(problem, pRef = pRef)

    scenario = ScenarioObserver(xcs_problem)
    algorithm = xcs.XCSAlgorithm()

    algorithm.exploration_probability = .5
    algorithm.discount_factor = 0
    algorithm.do_ga_subsumption = True
    algorithm.do_action_set_subsumption = True

    model = algorithm.new_model(scenario)

    def rule_to_ps(bitcondition: BitCondition) -> PS:
        bits = bitcondition.bits
        mask = bitcondition.mask

        ps_values = np.array(bits)
        where_unset = np.logical_not(np.array(mask, dtype=bool))
        ps_values[where_unset] = STAR
        return PS(ps_values)


    def get_match_set(model: xcs.ClassifierSet, solution: EvaluatedFS) -> (list[PS], xcs.MatchSet):
        situation=BitString(solution.values)
        # by_action = {}
        # for condition, actions in model._population.items():
        #     if not condition(situation):
        #         continue
        #
        #     for action, rule in actions.items():
        #         if action in by_action:
        #             by_action[action][condition] = rule
        #         else:
        #             by_action[action] = {condition: rule}
        #
        # # Construct the match set.
        # match_set = xcs.MatchSet(model, situation, by_action)
        #
        rule_population = [condition
                           for condition in model._population
                            if condition(situation)]

        # rule_population = match_set._model._population
        # all_rules = list(rule_population)


        pss = [rule_to_ps(bc) for bc in rule_population]
        # all_rules_match = all([contains(solution, ps) for ps in pss])
        return pss #, match_set



    def ps_to_rule(ps: PS) -> xcs.ClassifierRule:
        bits = ps.values.copy()
        mask = ps.values != STAR
        bits[~mask] = 0

        return xcs.XCSClassifierRule(
            BitCondition(bits, mask),
            True,
            algorithm,
            0
        )
    def look_at_new_good_solution(solution: EvaluatedFS):
        pss = get_match_set(model, solution)
        covered_spots = np.zeros(shape=len(solution), dtype=bool)
        if len(pss) > 0:
            covered_spots = np.sum([ps.values != STAR for ps in pss], axis=0, dtype=bool)
        coverage = np.average(np.array(covered_spots, dtype=float))
        print("The coverage is ", coverage)

        if coverage < 0.5:
            new_pss = local_ps_search(to_explain = solution,
                            to_avoid=pss,
                            population_size=50,
                            ps_evaluator=ps_evaluator,
                            ps_budget=3000,
                            verbose=False)

            for new_ps in new_pss:
                new_rule = ps_to_rule(new_ps)
                model.add(new_rule)

    solutions_to_look_at = set(xcs_problem.all_solutions)
    solutions_to_look_at = sorted(list(solutions_to_look_at), reverse=True)[:6]

    for good_solution in solutions_to_look_at:
        #model.run(scenario = scenario, learn=True)
        look_at_new_good_solution(good_solution)

    print("This is the model before calling .run")

    #model.run(scenario = scenario, learn=True)


    print("The model at the end is")
    print(model)




