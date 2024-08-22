import random

import numpy as np
import xcs
from xcs.bitstrings import BitString
from xcs.scenarios import ScenarioObserver

from BenchmarkProblems.BinVal import BinVal
from BenchmarkProblems.Checkerboard import CheckerBoard
from BenchmarkProblems.GraphColouring import GraphColouring
from BenchmarkProblems.MultiPlexerProblem import MediumMultiPlexerProblem
from BenchmarkProblems.RoyalRoad import RoyalRoad
from BenchmarkProblems.Trapk import Trapk
from Core.EvaluatedFS import EvaluatedFS
from Core.PS import PS
from Explanation.PRefManager import PRefManager
from LCS.CustomXCSAlgorithm import CustomXCSAlgorithm
from LCS.XCSProblemTopAndBottom import XCSProblemTopAndBottom
from LCS.XCSProblemTournamenter import XCSProblemTournamenter
from LightweightLocalPSMiner.FastPSEvaluator import FastPSEvaluator
from LightweightLocalPSMiner.TwoMetrics import TMEvaluator
from utils import announce
from Core.FullSolution import FullSolution

def check_linkage_metric(ps_evaluator: FastPSEvaluator):
    atomicity_metric = ps_evaluator.atomicity_metric
    atomicity_metric.set_solution(EvaluatedFS(FullSolution([1]*16), 16))

    to_test = ["1111 **** **** ****",
               "111* **** **** ****",
               "11** **** **** ****",
               "1*** **** **** ****",
               "1111 **** **** 1111",
               "1111 **** **** 11**",
               "11** 11** 11** ****",
               "1*** 1*** 1*** 1***",
               "0000 **** **** ****",
               "000* **** **** ****",
               "00** **** **** ****",
               "0*** **** **** ****",
               "0000 **** **** 0000",
               "0000 **** **** 00**",
               "00** 00** 00** ****",
               "0*** 0*** 0*** 0***",
               ]

    to_test = [PS.from_string(s) for s in to_test]


    def print_linkage(ps: PS):
        atomicity = atomicity_metric.get_atomicity_score(ps)
        independence = atomicity_metric.get_independence_score(ps)
        print(f"{ps}\t{atomicity}\t{independence}")

    for ps in to_test:
        print_linkage(ps)

def show_categories_of_linkage(ps_evaluator: FastPSEvaluator):
    atomicity_metric = ps_evaluator.atomicity_metric
    solution = EvaluatedFS(FullSolution([1]*8 + [0]*8), 16)
    atomicity_metric.set_solution(solution)

    def random_subset() -> PS:
        threshold = 0.7 #random.random()
        mask = np.random.random(len(solution)) < threshold
        values = solution.values.copy()
        values[mask] = -1
        return PS(values)

    sorted_by_signs = {(atom, indep): []
                       for atom in (True, False)
                       for indep in (True, False)}

    for _ in range(120):
        ps = random_subset()
        atomicity = atomicity_metric.get_atomicity_score(ps)
        independence = atomicity_metric.get_independence_score(ps)

        if atomicity == 0 or independence == 0:
            print(f"discarding {ps} because {atomicity = }, {independence =}")
        else:
            sorted_by_signs[(atomicity > 0, independence>0)].append(ps)


    for atom, indep in sorted_by_signs:
        print(f"{atom = }, {indep = }")
        for ps in sorted_by_signs[(atom, indep)]:
            print(ps)


def use_model_for_prediction(model: xcs.ClassifierSet, solution: FullSolution) -> dict:
    as_input = BitString(solution.values)
    match_set = model.match(as_input)
    selected_action = match_set.select_action()
    rules = list(match_set[selected_action])

    def get_rule_quality(rule):
        return rule.prediction_weight

    rules.sort(key=get_rule_quality)

    result = {"selected_action": selected_action,
              "rules": rules}

    return result



def run_LCS_as_archive():
    # optimisation_problem = RoyalRoad(4, 4) #GraphColouring.random(amount_of_colours=3, amount_of_nodes=6, chance_of_connection=0.4)
    # optimisation_problem = GraphColouring.random(amount_of_colours=3, amount_of_nodes=14, chance_of_connection=0.4)
    # optimisation_problem = Trapk(4, 5)
    optimisation_problem = CheckerBoard(4, 4)

    if isinstance(optimisation_problem, GraphColouring):
        optimisation_problem.view()
    pRef = PRefManager.generate_pRef(problem=optimisation_problem,
                                    sample_size=10000,
                                    which_algorithm="uniform SA")

    ps_evaluator = TMEvaluator(pRef)

    xcs_problem = XCSProblemTopAndBottom(optimisation_problem, pRef = pRef, training_cycles=3000, tail_size = 1000)
    scenario = ScenarioObserver(xcs_problem)
    algorithm = CustomXCSAlgorithm(ps_evaluator, xcs_problem)

    # algorithm.crossover_probability = 0
    # algorithm.deletion_threshold = 10000
    # algorithm.discount_factor = 0
    # algorithm.do_action_set_subsumption = True
    # algorithm.do_ga_subsumption = False
    # algorithm.exploration_probability = 0
    # algorithm.ga_threshold = 100000
    # algorithm.max_population_size = 50
    # algorithm.exploration_probability = 0
    # algorithm.minimum_actions = 1

    model = algorithm.new_model(scenario)

    with announce("Running the model"):
        model.run(scenario, learn=True)

    print("The model is")
    print(model)




    # solutions_to_evaluate = [FullSolution.random(optimisation_problem.search_space) for _ in range(120)]
    solutions_to_evaluate = [xcs_problem.all_solutions[0], xcs_problem.all_solutions[-1]]

    for solution in solutions_to_evaluate:
        as_input = BitString(solution.values)
        match_set = model.match(as_input)
        selected_action = match_set.select_action()

        actual_fitness = optimisation_problem.fitness_function(solution)

        result_dict = use_model_for_prediction(model, solution)
        rules = result_dict["rules"]
        for rule in rules:
            print(rule)

        print("The match set is")
        # for item in match_set:
        #     print(f"\t{item}")
        #     for rule in match_set[item]:
        #         print(rule)
        print(f"Solution: {solution}, predicted: {selected_action}, fitness:{actual_fitness}")

        # print("The actual result is")
        # result_dict = use_model_for_prediction(model, solution)
        # for key in result_dict:
        #     print(f"{key}:\n\t{result_dict[key]}")




run_LCS_as_archive()


