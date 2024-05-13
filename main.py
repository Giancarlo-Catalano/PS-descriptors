#!/usr/bin/env python3


""" This is the source code related to the paper 'Mining Potentially Explanatory Patterns via Partial Solutions',
    The authors of the paper are Giancarlo Catalano (me),
      Sandy Brownlee, David Cairns (Stirling Uni)
      John McCall (RGU)
      Russell Ainsle (BT).


    This code is a minimal implementation of the system I proposed in the paper,
    and I hope you can find the answers to any questions that you have!

    I recommend playing around with:
        - The problem being solved
        - The metrics used to search for PSs (find them in PSMiner.with_default_settings)
        - the sample sizes etc...
"""
import os

import utils
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from BenchmarkProblems.GraphColouring import GraphColouring
from BenchmarkProblems.RoyalRoad import RoyalRoad
from BenchmarkProblems.Trapk import Trapk
from Core import TerminationCriteria
from Core.Explainer import Explainer
from Explanation.Detector import Detector
from FSStochasticSearch.Operators import SinglePointFSMutation
from FSStochasticSearch.SA import SA
from PSMiners.DEAP.NSGAPSMiner import NSGAPSMiner
from PSMiners.MOEAD.testing import test_moead_on_problem
from PSMiners.Mining import get_history_pRef
from PSMiners.PyMoo.PSPyMooProblem import test_pymoo
from utils import announce, indent
import pandas as pd


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
    ps_miner = NSGAPSMiner.with_default_settings(pRef)
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
    experimental_directory = r"C:\Users\gac8\PycharmProjects\PS-PDF\Experimentation\BTDetectorLong"
    problem = EfficientBTProblem.from_default_files()
    return Detector.from_folder(problem=problem,
                          folder=experimental_directory,
                          speciality_threshold=0.30,
                          verbose=True)

def get_faulty_bt_explainer():
    experimental_directory = r"C:\Users\gac8\PycharmProjects\PS-PDF\Experimentation\FaultyerBT"
    print("Using the FAULTY problem")
    problem = EfficientBTProblem.from_default_files()
    problem.use_faulty_fitness_function = True
    return Detector.from_folder(problem=problem,
                          folder=experimental_directory,
                          speciality_threshold=0.1,
                          verbose=True)

def get_gc_explainer():
    experimental_directory = r"C:\Users\gac8\PycharmProjects\PS-PDF\Experimentation\GCDetector"
    problem_file = os.path.join(experimental_directory, "islets.json")
    problem = GraphColouring.from_file(problem_file)
    problem.view()
    return Detector.from_folder(folder = experimental_directory,
                                  problem = problem,
                                  speciality_threshold=0.25,
                                  verbose=True)


def get_trapk_explainer():
    experimental_directory = r"C:\Users\gac8\PycharmProjects\PS-PDF\Experimentation\Other"
    problem = RoyalRoad(4, 5)
    return Detector.from_folder(folder = experimental_directory,
                                  problem = problem,
                                  speciality_threshold=0.25,
                                  verbose=True)






if __name__ == '__main__':
    #detector = get_bt_explainer()
    #detector.generate_files_with_default_settings()
    #detector.explanation_loop(amount_of_fs_to_propose=6, ps_show_limit=12)

    # this is another change

    # utils.make_joined_bt_dataset()

    # problem = RoyalRoad(5, 5)
    # test_pymoo(problem)


    #get_bt_explainer().generate_properties_csv_file()
    # get_faulty_bt_explainer().generate_properties_csv_file()

    #problem = RoyalRoad(3, 5)
    #test_moead_on_problem(problem, sample_size=5000)


    algorithm_which_options = ["NSGAII"]  # , "NSGAIII","MOEAD"
    crowding_which_options= ["gc", "cd", "ce", "mnn", "2nn"]  # PCD
    problem = RoyalRoad(3, 5)

    with announce(f"Generating a pRef"):
        pRef = get_history_pRef(benchmark_problem=problem,
                                sample_size=10000,
                                which_algorithm="SA",
                                verbose=True)


    for algorithm in algorithm_which_options:
        for crowding in crowding_which_options:
            test_pymoo(problem,
                       pRef = pRef,
                       which_algorithm=algorithm,
                       which_crowding =crowding)


