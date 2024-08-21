import random

import numpy as np
from xcs.scenarios import ScenarioObserver

from BenchmarkProblems.BinVal import BinVal
from BenchmarkProblems.MultiPlexerProblem import MediumMultiPlexerProblem
from BenchmarkProblems.RoyalRoad import RoyalRoad
from BenchmarkProblems.Trapk import Trapk
from Core.EvaluatedFS import EvaluatedFS
from Core.PS import PS
from Explanation.PRefManager import PRefManager
from LCS.CustomXCSAlgorithm import CustomXCSAlgorithm
from LCS.XCSProblemTournamenter import XCSProblemTournamenter
from LightweightLocalPSMiner.FastPSEvaluator import FastPSEvaluator
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


def run_LCS_as_archive():
    optimisation_problem = RoyalRoad(4, 4)
    pRef = PRefManager.generate_pRef(problem=optimisation_problem,
                                    sample_size=10000,
                                    which_algorithm="uniform")

    ps_evaluator = FastPSEvaluator(pRef)

    show_categories_of_linkage(ps_evaluator)
    check_linkage_metric(ps_evaluator)

    xcs_problem = XCSProblemTournamenter(optimisation_problem, pRef = pRef, training_cycles=1000)
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


run_LCS_as_archive()


