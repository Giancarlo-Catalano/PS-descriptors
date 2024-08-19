import random
from typing import Optional, Any

import numpy as np
import xcs
from xcs.bitstrings import BitString, BitCondition
from xcs.scenarios import Scenario, ScenarioObserver

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from BenchmarkProblems.RoyalRoad import RoyalRoad
from BenchmarkProblems.Trapk import Trapk
from Core.EvaluatedFS import EvaluatedFS
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS, STAR, contains
from Explanation.PRefManager import PRefManager
from LightweightLocalPSMiner.FastPSEvaluator import FastPSEvaluator
from LightweightLocalPSMiner.LocalPSSearch import local_ps_search
from PSMiners.Mining import get_history_pRef
from utils import announce


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


def rule_to_ps(bitcondition: BitCondition) -> PS:
    bits = bitcondition.bits
    mask = bitcondition.mask

    ps_values = np.array(bits)
    where_unset = np.logical_not(np.array(mask, dtype=bool))
    ps_values[where_unset] = STAR
    return PS(ps_values)


def get_match_set(model: xcs.ClassifierSet, solution: EvaluatedFS) -> list[PS]:
    situation=BitString(solution.values)
    rule_population = [condition
                       for condition in model._population
                        if condition(situation)]

    pss = [rule_to_ps(bc) for bc in rule_population]
    return pss


def add_pss_to_model_as_rules(model: xcs.ClassifierSet,
                              algorithm,
                              pss: list[PS],
                              action):
    def ps_to_rule(ps: PS) -> xcs.ClassifierRule:
        bits = ps.values.copy()
        mask = ps.values != STAR
        bits[~mask] = 0

        return xcs.XCSClassifierRule(
            BitCondition(bits, mask),
            action,
            algorithm,
            0
        )

    for new_ps in pss:
        new_rule = ps_to_rule(new_ps)
        model.add(new_rule)


def get_solution_coverage(solution: FullSolution, pss: list[PS]) -> float:
    covered_spots = np.zeros(shape=len(solution), dtype=bool)
    if len(pss) > 0:
        covered_spots = np.sum([ps.values != STAR for ps in pss], axis=0, dtype=bool)
    coverage = np.average(np.array(covered_spots, dtype=float))
    return coverage


def get_rules_in_model(model: xcs.ClassifierSet) -> list[(PS, Any)]:
    result = model._population

    pss = map(rule_to_ps, result)  # result is a dictionary, where the keys are bitconditions. We convert each to a ps
    actions = [list(assigned_actions) for assigned_actions in result.values()]

    return list(zip(pss, actions))


def check_linkage_metric(ps_evaluator: FastPSEvaluator):
    atomicity_metric = ps_evaluator.atomicity_metric

    nnny = PS(([-1] * 12)+([1]*4))
    ynnn = PS(([1]*4)+([-1] * 12))
    ynny = PS(([1]*4)+([-1] * 8)+([1]*4))

    nnno = PS(([-1] * 12)+([0]*4))
    onnn = PS(([0]*4)+([-1] * 12))
    onno = PS(([0]*4)+([-1] * 8)+([0]*4))

    single_o = PS(([0]*1)+([-1] * 11))


    def print_linkage(ps: PS):
        atomicity = atomicity_metric.get_single_score(ps)
        print(f"The atomicity for {ps} is {atomicity}")

    for ps in [nnny, ynnn, ynny, nnno, onnn, onno, single_o]:
        print_linkage(ps)


def run_model_without_disruption(model: xcs.ClassifierSet, scenario: Scenario):
    # this is a modification of xsc.model.run(scenario, learn=True), where dynamic is false

    # Repeat until the scenario has run its course.
    while scenario.more():
        # Gather information about the current state of the environment.
        situation = scenario.sense()

        # Determine which rules match the current situation.
        match_set = model.match(situation)

        # Select the best action for the current situation (or a random one, if we are on an exploration step).
        match_set.select_action()

        # Perform the selected action and find out what the received reward was.
        reward = scenario.execute(match_set.selected_action)

        match_set.payoff = reward
        match_set.apply_payoff()


def run_LCS_as_archive():
    optimisation_problem = RoyalRoad(5, 4)
    pRef = PRefManager.generate_pRef(problem=optimisation_problem,
                                    sample_size=10000,
                                    which_algorithm="SA uniform")

    ps_evaluator = FastPSEvaluator(pRef)

    # check_linkage_metric(ps_evaluator)

    def get_lcs_objects(optimisation_problem: BenchmarkProblem, pRef: PRef):
        xcs_problem = XSCProblemTournamenter(optimisation_problem, pRef = pRef, training_cycles=10000)
        scenario = ScenarioObserver(xcs_problem)
        algorithm = xcs.XCSAlgorithm()

        algorithm.crossover_probability = 0
        algorithm.deletion_threshold = 10000
        algorithm.discount_factor = 0
        algorithm.do_action_set_subsumption = True
        algorithm.do_ga_subsumption = False
        algorithm.exploration_probability = 0

        # algorithm.exploration_probability = 0
        # algorithm.discount_factor = 0
        # algorithm.do_ga_subsumption = True
        # algorithm.do_action_set_subsumption = True
        # algorithm.mutation_probability = False

        model = algorithm.new_model(scenario)

        return xcs_problem, scenario, algorithm, model


    xcs_problem, scenario, algorithm, model = get_lcs_objects(optimisation_problem, pRef)

    def look_at_new_solution(solution: EvaluatedFS, action: bool):
        matched_pss = get_match_set(model, solution)

        coverage = get_solution_coverage(solution, matched_pss)
        if coverage > 0.8:
            print(f"Coverage for {solution} is {int(coverage*100)}%, I can't learn anything from this!")
            return

        if action:
            ps_evaluator.set_to_maximise_fitness()
        else:
            ps_evaluator.set_to_minimise_fitness()

        pss = local_ps_search(to_explain = solution,
                              to_avoid=matched_pss,
                              population_size=50,
                              ps_evaluator=ps_evaluator,
                              ps_budget = 300,
                              verbose=False)

        print("---------------------------------------------------The produced pss are ")
        pss.sort(key=lambda x: x.metric_scores[2])
        pss = pss[:10]
        for ps in pss:
            print(ps)

        add_pss_to_model_as_rules(model,
                                  action=action,
                                  algorithm=algorithm,
                                  pss=pss)


    # model.run(scenario = scenario, learn=True)
    #
    # print("This is the model before insemination")
    # print(model)

    solutions_to_look_at = set(xcs_problem.all_solutions)
    solutions_to_look_at = sorted(list(solutions_to_look_at), reverse=True)

    amount_of_solutions_to_look_at = 12
    amount_of_solutions_to_look_at = min(amount_of_solutions_to_look_at, len(solutions_to_look_at))

    for index in range(amount_of_solutions_to_look_at // 2):
        print(f"Checking index {index}")
        good_solution = solutions_to_look_at[index]
        bad_solution = solutions_to_look_at[-(index+1)]
        look_at_new_solution(good_solution, action=True)
        look_at_new_solution(bad_solution, action=False)

    print("This is the model before calling .run")
    print(model)


    with announce("Running the model"):
        run_model_without_disruption(model, scenario)  # TODO: understand why this gets rid of all the nice rules we inseminated

    print("The model at the end is")
    print(model)



run_LCS_as_archive()


