#!/usr/bin/env python3
import logging
import os
import warnings

import utils
from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from BenchmarkProblems.GraphColouring import GraphColouring
from Core.FullSolution import FullSolution
from Core.PS import PS
from ExplanationGeneration.Detector import Detector
from ExplanationGeneration.HyperparameterEvaluator import HyperparameterEvaluator


def get_bt_explainer() -> Detector:
    experimental_directory = r"C:\Users\gac8\PycharmProjects\PS-PDF\Experimentation\BT\Dummy"
    problem = EfficientBTProblem.from_default_files()

    return Detector.from_folder(problem=problem,
                                  folder=experimental_directory,
                                  speciality_threshold=0.10,
                                  verbose=True)

def get_gc_explainer():
    experimental_directory = r"C:\Users\gac8\PycharmProjects\PS-PDF\Experimentation\GC\Dummy"
    problem_file = os.path.join(experimental_directory, "bert.json")
    problem = GraphColouring.from_file(problem_file)
    problem.view()
    return Detector.from_folder(folder = experimental_directory,
                                  problem = problem,
                                  speciality_threshold=0.50,
                                  verbose=True)

def explanation():
    detector = get_bt_explainer()
    detector.generate_files_with_default_settings(500, 500)
    #detector.explanation_loop(amount_of_fs_to_propose=2, ps_show_limit=1000, show_debug_info=True)


def grid_search():
    hype = HyperparameterEvaluator(algorithms_to_test=["NSGAII", "NSGAIII", "MOEAD", "SMS-EMOA"],
                                   problems_to_test=["collaboration_5", "insular_5", "RR_5"],
                                   pRef_sizes_to_test=[10000],
                                   population_sizes_to_test=[50, 100, 200],
                                   pRef_origin_methods = ["uniform", "SA", "uniform SA"],
                                   ps_budget=50000,
                                   custom_crowding_operators_to_test = [False, True],
                                   ps_budgets_per_run_to_test=[1000, 2000, 3000, 5000, 10000])


    hype.get_data(ignore_errors=True,
                  verbose=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    warnings.showwarning = utils.warn_with_traceback

    #grid_search()
    explanation()








