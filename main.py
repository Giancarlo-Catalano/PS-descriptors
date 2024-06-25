#!/usr/bin/env python3
import logging
import os
import warnings

import utils
from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from BenchmarkProblems.GraphColouring import GraphColouring
from FirstPaper.FullSolution import FullSolution
from FirstPaper.PS import PS
from ExplanationGeneration.Explainer import Explainer
from ExplanationGeneration.HyperparameterEvaluator import HyperparameterEvaluator


def get_bt_explainer() -> Explainer:
    # this defines the directory where the Partial Solution files will be stored.
    experimental_directory = r"C:\Users\gac8\PycharmProjects\PS-PDF\Experimentation\BT\StaffRosteringProblemCache"

    # loads the problem as defined in some files, it should be resources/BT/MartinsInstance
    problem = EfficientBTProblem.from_default_files()

    return Explainer.from_folder(problem=problem,
                                 folder=experimental_directory,
                                 speciality_threshold=0.10,  # this is the threshold for polarity as discussed in the paper
                                 verbose=True)

def get_gc_explainer():
    experimental_directory = r"C:\Users\gac8\PycharmProjects\PS-PDF\Experimentation\GC\Dummy"
    problem_file = os.path.join(experimental_directory, "bert.json")
    problem = GraphColouring.from_file(problem_file)
    problem.view()
    return Explainer.from_folder(folder = experimental_directory,
                                 problem = problem,
                                 speciality_threshold=0.50,
                                 verbose=True)

def explanation():
    # constructing the explainer object, which determines the problem and the working directory
    explainer = get_bt_explainer()

    # to generate the files containing PSs, properties etc..
    # You should only run this once, since it is quite slow
    #explainer.generate_files_with_default_settings(5000, 5000)

    # this starts the main explanation function, and uses the files generated above
    explainer.explanation_loop(amount_of_fs_to_propose=2, ps_show_limit=3, show_debug_info=True)


def grid_search():
    # construct the set of parameters that will be used in the testing
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

    # the 2 lines below are just to see more detailed errors and logs
    logging.basicConfig(level=logging.INFO)
    warnings.showwarning = utils.warn_with_traceback

    # this line is to run the tests as discussed in the paper
    #grid_search()

    # this line is to run the explainer
    explanation()








