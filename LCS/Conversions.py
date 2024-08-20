import heapq
from typing import Any

import numpy as np
import xcs
from xcs.bitstrings import BitString, BitCondition
from xcs.scenarios import Scenario, ScenarioObserver

from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core.EvaluatedFS import EvaluatedFS
from Core.FullSolution import FullSolution
from Core.PRef import PRef
from Core.PS import PS, STAR
from LCS.XCSProblemTournamenter import XCSProblemTournamenter
from LightweightLocalPSMiner.FastPSEvaluator import FastPSEvaluator
from LightweightLocalPSMiner.LocalPSSearch import local_ps_search


def condition_to_ps(bitcondition: BitCondition) -> PS:
    bits = bitcondition.bits
    mask = bitcondition.mask

    ps_values = np.array(bits)
    where_unset = np.logical_not(np.array(mask, dtype=bool))
    ps_values[where_unset] = STAR
    return PS(ps_values)

def rule_to_ps(rule: xcs.ClassifierRule) -> PS:
    return condition_to_ps(rule.condition)


def ps_to_condition(ps: PS) -> BitCondition:
    bits = ps.values.copy()
    mask = ps.values != STAR
    bits[~mask] = 0

    return BitCondition(bits, mask)
def ps_to_rule(algorithm,
                  ps :PS,
                  action) -> xcs.ClassifierRule:
    return xcs.XCSClassifierRule(
        ps_to_condition(ps),
        action,
        algorithm,
        0)

def situation_to_fs(situation) -> FullSolution:
    return FullSolution(situation)


def get_match_set(model: xcs.ClassifierSet, situation) -> xcs.MatchSet:
    # the function in the normal implementation also applies covering
    by_action = {}
    for condition, actions in model._population.items():
        if not condition(situation):
            continue

        for action, rule in actions.items():
            if action in by_action:
                by_action[action][condition] = rule
            else:
                by_action[action] = {condition: rule}

    # Construct the match set.
    return xcs.MatchSet(model, situation, by_action)



def get_pss_from_action_set(action_set: xcs.ActionSet) -> list[PS]:
    rules = action_set._rules
    return list(map(condition_to_ps, rules))

def get_lcs_objects(optimisation_problem: BenchmarkProblem, pRef: PRef):
    xcs_problem = XCSProblemTournamenter(optimisation_problem, pRef = pRef, training_cycles=pRef.sample_size)
    scenario = ScenarioObserver(xcs_problem)
    algorithm = xcs.XCSAlgorithm()

    algorithm.crossover_probability = 0
    algorithm.deletion_threshold = 10000
    algorithm.discount_factor = 0
    algorithm.do_action_set_subsumption = True
    algorithm.do_ga_subsumption = False
    algorithm.exploration_probability = 0
    algorithm.max_population_size = 50
    algorithm.exploration_probability = 0
    algorithm.minimum_actions = 1
    # algorithm.exploration_probability = 0
    # algorithm.discount_factor = 0
    # algorithm.do_ga_subsumption = True
    # algorithm.do_action_set_subsumption = True
    # algorithm.mutation_probability = False

    model = algorithm.new_model(scenario)

    return xcs_problem, scenario, algorithm, model


def run_model_assuming_not_dynamic(model: xcs.ClassifierSet, scenario: Scenario):
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


def get_rules_in_model(model: xcs.ClassifierSet) -> list[(PS, Any)]:
    result = model._population

    pss = map(condition_to_ps, result)  # result is a dictionary, where the keys are bitconditions. We convert each to a ps
    actions = [list(assigned_actions) for assigned_actions in result.values()]

    return list(zip(pss, actions))

def own_rules_for_covering(model: xcs.ClassifierSet,
                       situation,
                       action,
                       match_set: xcs.MatchSet,
                       action_set: xcs.ActionSet,
                       ps_evaluator: FastPSEvaluator):
    already_found_pss = get_pss_from_action_set(action_set)
    solution = FullSolution(situation)
    if action:
        ps_evaluator.set_to_maximise_fitness()
    else:
        ps_evaluator.set_to_minimise_fitness()
    pss = local_ps_search(to_explain = solution,
                          to_avoid=already_found_pss,
                          population_size=50,
                          ps_evaluator=ps_evaluator,
                          ps_budget = 300,
                          verbose=False)

    pss = heapq.nlargest(10, iterable=pss, key=lambda x: x.metric_scores[2])

    return [ps_to_rule(algorithm=model.algorithm, action=action, ps = ps) for ps in pss]







# unused
def get_matched_pss(model: xcs.ClassifierSet, solution: EvaluatedFS) -> list[PS]:
    situation=BitString(solution.values)
    rule_population = [condition
                       for condition in model._population
                        if condition(situation)]

    pss = [condition_to_ps(bc) for bc in rule_population]
    return pss




def get_solution_coverage(match_set: xcs.MatchSet, action) -> float:
    action_set = get_action_set(match_set, action)
    rules = list(action_set._rules)  # these are conditions, for some reason
    masks = np.array([np.array(condition.mask) for condition in rules])
    if len(masks) == 0:
        return 0
    covered_vars = np.sum(masks, axis=0, dtype=bool)
    return np.average(covered_vars)


def get_action_set(match_set: xcs.MatchSet, action) -> xcs.ActionSet:
    def make_empty_action_set():
        return xcs.ActionSet(model = match_set.model,
                             situation=match_set.situation,
                             action=action,
                             rules = dict())
    return match_set._action_sets.get(action, make_empty_action_set())

def get_lcs_objects(optimisation_problem: BenchmarkProblem, pRef: PRef):
    xcs_problem = XCSProblemTournamenter(optimisation_problem, pRef = pRef, training_cycles=10000)
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











# def look_at_new_solution(solution: EvaluatedFS, action: bool):
#     matched_pss = get_match_set(model, solution)
#
#     coverage = get_solution_coverage(solution, matched_pss)
#     print(f"Learning from {solution}, coverage is {int(coverage*100)}%")
#     if coverage > 0.5:
#         print("Skipping")
#         return
#
#
#     if action:
#         ps_evaluator.set_to_maximise_fitness()
#     else:
#         ps_evaluator.set_to_minimise_fitness()
#
#     pss = local_ps_search(to_explain = solution,
#                           to_avoid=matched_pss,
#                           population_size=50,
#                           ps_evaluator=ps_evaluator,
#                           ps_budget = 300,
#                           verbose=False)
#
#     # print("---------------------------------------------------The produced pss are ")
#     pss.sort(key=lambda x: x.metric_scores[2])
#     pss = pss[:10]
#     # for ps in pss:
#     #     print(ps)
#
#     add_pss_to_model_as_rules(model,
#                               action=action,
#                               algorithm=algorithm,
#                               pss=pss)
#
#
# # model.run(scenario = scenario, learn=True)
# #
# # print("This is the model before insemination")
# # print(model)
#
# solutions_to_look_at = set(xcs_problem.all_solutions)
# solutions_to_look_at = sorted(list(solutions_to_look_at), reverse=True)
#
# amount_of_solutions_to_look_at = 20
# amount_of_solutions_to_look_at = min(amount_of_solutions_to_look_at, len(solutions_to_look_at))
#
# for index in range(amount_of_solutions_to_look_at // 2):
#     print(f"Checking index {index}")
#     good_solution = solutions_to_look_at[index]
#     bad_solution = solutions_to_look_at[-(index+1)]
#     if good_solution.fitness == bad_solution.fitness:
#         print("The fitnesses are now the same, breaking")
#         break
#     look_at_new_solution(good_solution, action=True)
#     look_at_new_solution(bad_solution, action=False)
#
# print("This is the model before calling .run")
# print(model)
#
#
# with announce("Running the model"):
#     run_model_assuming_not_dynamic(model, scenario)  # TODO: understand why this gets rid of all the nice rules we inseminated
#
# print("The model at the end is")
# print(model)
#
