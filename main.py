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
from Core.EvaluatedPS import EvaluatedPS
from Core.Explainer import Explainer
from Core.PS import PS
from Core.PSMetric.Classic3 import Classic3PSEvaluator
from Explanation.Detector import Detector
from FSStochasticSearch.Operators import SinglePointFSMutation
from FSStochasticSearch.SA import SA
from PSMiners.AbstractPSMiner import AbstractPSMiner
from PSMiners.DEAP.DEAPPSMiner import DEAPPSMiner, test_DEAP_miner
from PSMiners.MOEAD.testing import test_moead_on_problem
from PSMiners.Mining import get_history_pRef, get_ps_miner
from PSMiners.Platypus.PlatypusPSProblem import test_platypus
from PSMiners.PyMoo.PSPyMooProblem import test_pymoo, sequential_search_pymoo
from PSMiners.PyMoo.SequentialCrowdingMiner import test_sequential_miner
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
    experimental_directory = r"C:\Users\gac8\PycharmProjects\PS-PDF\Experimentation\BTDetectorSequential"
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





def explanation():
    detector = get_bt_explainer()
    #detector.generate_files_with_default_settings()
    detector.explanation_loop(amount_of_fs_to_propose=6, ps_show_limit=12)

    # utils.make_joined_bt_dataset()

    # problem = RoyalRoad(5, 5)
    # test_pymoo(problem)

    # detector.generate_control_pss()
    # detector.generate_properties_csv_file()
    # get_faulty_bt_explainer().generate_properties_csv_file()

    #problem = RoyalRoad(3, 5)
    #test_moead_on_problem(problem, sample_size=5000)

def get_miners_data():
    problem = RoyalRoad(5, 5)
    ps_budget = 10000

    all_results = dict()

    def add_to_results_with_name(miner_name, pss: list[PS]):
        all_results[miner_name] = pss


    with announce(f"Generating a pRef"):
        pRef = get_history_pRef(benchmark_problem=problem,
                                sample_size=10000,
                                which_algorithm="uniform",
                                verbose=True)

    def get_deap_data():
        print("running deap")
        for custom_crowding in [True, False]:
            from_deap = test_DEAP_miner(benchmark_problem=problem,
                                        budget=ps_budget,
                                        pRef = pRef,
                                        custom_crowding=True)

            all_results[f"DEAP_custom_crowding_{custom_crowding}"] = from_deap.copy()


    def get_pymoo_data():
        from_pymoo = test_pymoo(problem,
                                pRef = pRef,
                                which_algorithm="NSGAIII",
                                which_crowding ="mnn")
        all_results[f"pymoo_NSGAIII"] = from_pymoo.copy()

        for algorithm in ["NSGAII"]:
            for crowding in ["gc", "cd", "ce", "mnn", "2nn"]:
                try:
                    from_pymoo = test_pymoo(problem,
                                            pRef = pRef,
                                            which_algorithm=algorithm,
                                            which_crowding =crowding)

                    all_results[f"pymoo_{algorithm}_{crowding}"] = from_pymoo.copy()
                except Exception as e:
                    print(f"The combination {algorithm} + {crowding} caused an error")


    def get_platypus_data():
        jail = ["NSGAIII", "CMAES", "GDE3",  "IBEA", "OMOPSO", "SMPSO", "SPEA2", "EpsMOEA"]
        slow = ["MOEAD"]
        heaven = ["NSGAII"]
        for which_algorithm in heaven:
            with announce(f"Running Platypus's {which_algorithm}"):
                pss = test_platypus(problem, pRef=pRef, budget=ps_budget, which_algorithm=which_algorithm)
                add_to_results_with_name(f"Platypus_{which_algorithm}", pss)




    evaluator = Classic3PSEvaluator(pRef)
    def get_atomicity(ps: PS) -> float:
        s, mf, a = evaluator.get_S_MF_A(ps)
        return a
    def get_sorted_by_atomicity(pss: list[PS]):
        e_pss = [EvaluatedPS(ps.values, aggregated_score=get_atomicity(ps)) for ps in pss]
        e_pss.sort(reverse=True)
        return e_pss

    # get_deap_data()
    # get_pymoo_data()
    # get_platypus_data()
    #
    # for miner_key in all_results:
    #     print(f"\n\n\nFor the miner {miner_key}")
    #     sorted_pss = get_sorted_by_atomicity(all_results[miner_key])
    #     for ps in sorted_pss:
    #         print(problem.repr_ps(ps))



if __name__ == '__main__':
    explanation()

    # problem = EfficientBTProblem.from_default_files()
    # #problem = Trapk(5, 5)
    # ps_budget = 10000





    # with announce(f"Generating a pRef"):
    #     pRef = get_history_pRef(benchmark_problem=problem,
    #                             sample_size=30000,
    #                             which_algorithm="uniform",
    #                             verbose=True)
    #
    # pss = test_sequential_miner(pRef=pRef, total_budget=10000)
    #
    # print("The archive at the end is")
    # for ps in pss:
    #     print(problem.repr_ps(ps))







