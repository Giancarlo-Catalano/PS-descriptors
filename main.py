#!/usr/bin/env python3
import heapq
import logging
import os
import sys
import traceback
import warnings

import numpy as np

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from BenchmarkProblems.EfficientBTProblem.ManuallyConstructedBTInstances import get_bad_week_instance
from BenchmarkProblems.GraphColouring import GraphColouring
from BenchmarkProblems.RoyalRoad import RoyalRoad
from Core import TerminationCriteria
from Core.ArchivePSMiner import ArchivePSMiner
from Core.EvaluatedPS import EvaluatedPS
from Core.Explainer import Explainer
from Core.PRef import PRef
from Core.PS import PS
from Core.PSMetric.Additivity import Additivity, MeanError, ExternalInfluence
from Core.PSMetric.Atomicity import Atomicity
from Core.PSMetric.BivariateANOVALinkage import BivariateANOVALinkage
from Core.PSMetric.Classic3 import Classic3PSEvaluator
from Core.PSMetric.Linkage import Linkage
from Core.PSMetric.LocalPerturbation import BivariateLocalPerturbation, UnivariateLocalPerturbation
from Core.PSMetric.Metric import Metric
from Core.TerminationCriteria import PSEvaluationLimit
from Explanation.Detector import Detector
from Explanation.HyperparameterEvaluator import HyperparameterEvaluator
from Explanation.PRefManager import PRefManager
from FSStochasticSearch.Operators import SinglePointFSMutation
from FSStochasticSearch.SA import SA
from PSMiners.DEAP.DEAPPSMiner import DEAPPSMiner
from PSMiners.Mining import get_history_pRef
from utils import announce, indent, decode_data_from_islets


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

def show_overall_system(benchmark_problem: BenchmarkProblem):
    """
    This function gives an overview of the system:
        1. Generate a reference population (a PRef)
        2. Generate a Core Catalog using the Core Miner
        3. Sample new solutions from the catalog using Pick & Merge
        4. Explain those new solutions using the catalog

    :param benchmark_problem: a benchmark problem, find more in the BenchmarkProblems directory
    :return: Nothing! Just printing
    """

    print(f"The problem is {benchmark_problem}")

    # 1. Generating the reference population
    pRef_size = 10000
    with announce("Generating Reference Population"):
        pRef = get_history_pRef(benchmark_problem, sample_size=pRef_size, which_algorithm="SA")
    pRef.describe_self()

    # 2. Obtaining the Core catalog
    ps_miner = DEAPPSMiner.with_default_settings(pRef)
    ps_evaluation_budget = 10000
    termination_criterion = TerminationCriteria.PSEvaluationLimit(ps_evaluation_budget)

    with announce("Running the PS Miner"):
        ps_miner.run(termination_criterion, verbose=True)

    ps_catalog = ps_miner.get_results(None)
    ps_catalog = list(set(ps_catalog))
    ps_catalog = [item for item in ps_catalog if not item.is_empty()]

    print("The catalog consists of:")
    for item in ps_catalog:
        print("\n")
        print(indent(f"{benchmark_problem.repr_ps(item)}"))

    # 3. Sampling new solutions
    print("\nFrom the catalog we can sample new solutions")
    new_solutions_to_produce = 12
    sampler = SA(fitness_function=benchmark_problem.fitness_function,
                   search_space=benchmark_problem.search_space,
                   mutation_operator=SinglePointFSMutation(benchmark_problem.search_space),
                   cooling_coefficient=0.9995)

    solutions = pRef.get_evaluated_FSs()
    solutions = list(set(solutions))
    solutions.sort(reverse=True)


    for index, sample in enumerate(solutions[:6]):
        print(f"[{index}]")
        print(indent(indent(f"{benchmark_problem.repr_fs(sample.full_solution)}, has fitness {sample.fitness:.2f}")))

    # 4. Explainability, at least locally.
    explainer = Explainer(benchmark_problem, ps_catalog, pRef)
    explainer.explanation_loop(solutions)

    print("And that concludes the showcase")

def get_bt_explainer() -> Detector:
    experimental_directory = r"C:\Users\gac8\PycharmProjects\PS-PDF\Experimentation\BT\MartinBT"
    problem = EfficientBTProblem.from_default_files()
    return Detector.from_folder(problem=problem,
                                  folder=experimental_directory,
                                  speciality_threshold=0.2,
                                  verbose=True)

def get_gc_explainer():
    experimental_directory = r"C:\Users\gac8\PycharmProjects\PS-PDF\Experimentation\GC\Dummy"
    problem_file = os.path.join(experimental_directory, "fibre.json")
    problem = GraphColouring.from_file(problem_file)#GraphColouring.random(amount_of_colours=3, amount_of_nodes=5, chance_of_connection=0.40)
    problem.view()
    return Detector.from_folder(folder = experimental_directory,
                                  problem = problem,
                                  speciality_threshold=0.20,
                                  verbose=True)


def get_manual_bt_explainer() -> Detector:
    experimental_directory = r"C:\Users\gac8\PycharmProjects\PS-PDF\Experimentation\BT\TwoTeam"
    amount_of_skills = 12
    problem = get_bad_week_instance(amount_of_skills, workers_per_skill=4)
    #problem = get_start_and_end_instance(amount_of_skills)
    #problem = get_toestepping_instance(amount_of_skills=3)
    #problem = get_unfairness_instance(amount_of_skills=3)
    return Detector.from_folder(problem=problem,
                                folder=experimental_directory,
                                speciality_threshold=0.2,
                                verbose=True)


def get_problem_explainer() -> Detector:
    experimental_directory = r"C:\Users\gac8\PycharmProjects\PS-PDF\Experimentation\GC\Dummy"

    use_gc = False
    if use_gc:
        gc_problem = GraphColouring.make_insular_instance(4)
        gc_problem.view()
        problem = EfficientBTProblem.from_Graph_Colouring(gc_problem)
    else:
        rr_problem = RoyalRoad(3, 4)
        problem = rr_problem
    return Detector.from_folder(problem=problem,
                                folder=experimental_directory,
                                speciality_threshold=0.2,
                                verbose=True)


def test_classic3(pRef: PRef):
    evaluator = Classic3PSEvaluator(pRef)

    pss = []
    for s in range(2, 8):
        pss.extend([PS.random_with_fixed_size(pRef.search_space, s) for _ in range(1000)])
    pss = list(set(pss))

    pss = [EvaluatedPS(ps, evaluator.get_S_MF_A(ps)) for ps in pss]
    def sort_by_metric(metric: Metric):
        print(f"Sorted by {metric}")
        for ps in pss:
            ps.metric_scores[2] = metric.get_single_score(ps)
        best = heapq.nlargest(30, pss, key=lambda x: x.metric_scores[2])
        for ps in best:
            print(ps)


    atomicity_metrics = [
                         ExternalInfluence(),
                         Atomicity(),
                         #BivariateLocalPerturbation(),
                         #Additivity(0),
                         #Additivity(1),
                         #Additivity(2),
                         #Additivity(3),
                         #BivariateANOVALinkage(),
                         #UnivariateLocalPerturbation(),
                         #MeanError()
                         ]

    for metric in atomicity_metrics:
        metric.set_pRef(pRef)
        sort_by_metric(metric)


    print(f"Sorted by all")
    sorted_pss = utils.sort_by_combination_of(pss, key_functions=[lambda x: x.metric_scores[0],
                                                         lambda x: x.metric_scores[1],
                                                         lambda x: x.metric_scores[2]], reverse=True)
    for ps in sorted_pss[:120]:
        print(ps)


def explanation():
    detector = get_problem_explainer()
    detector.generate_files_with_default_settings(100000, 100000)
    #detector.explanation_loop(amount_of_fs_to_propose=2, ps_show_limit=12, show_debug_info=True)


def grid_search():
    # hype = HyperparameterEvaluator(algorithms_to_test=["NSGAII", "NSGAIII", "MOEAD"],
    #                                problems_to_test=["island_3", "island_5", "island_10", "RR_3", "RR_5", "RR_10"],
    #                                pRef_sizes_to_test=[30000],
    #                                population_sizes_to_test=[100, 200],
    #                                pRef_origin_methods = ["SA+uniform", "SA"],
    #                                ps_budget=30000,
    #                                custom_crowding_operators_to_test = [True, False],
    #                                ps_budgets_per_run_to_test=[1000, 3000, 5000])
    hype = HyperparameterEvaluator(algorithms_to_test=["NSGAII"],
                                   problems_to_test=["RR_5"],
                                   pRef_sizes_to_test=[30000],
                                   population_sizes_to_test=[1000],
                                   pRef_origin_methods = ["SA uniform"],
                                   ps_budget=150000,
                                   custom_crowding_operators_to_test = [True],
                                   ps_budgets_per_run_to_test=[15000])
    hype.get_data()



def test_archive_miner():
    problem = RoyalRoad(5, 5)
    print("Generating the pRef")
    pRef = PRefManager.generate_pRef(problem = problem,
                                     which_algorithm= "SA uniform",
                                     sample_size=10000)

    print("running the miner")
    miner = ArchivePSMiner.with_default_settings(pRef)
    miner.run(termination_criteria=PSEvaluationLimit(10000))
    results = miner.get_results(100)
    print("The results are")
    for ps in results:
        print(ps)




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    warnings.showwarning = warn_with_traceback
    #grid_search()
    #explanation()
    # test_archive_miner()
    test_classic3(RoyalRoad(3, 4).get_reference_population(5000))






