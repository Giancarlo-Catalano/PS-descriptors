#!/usr/bin/env python3
import logging
import os
import warnings

import utils
from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from Core.Explainer import Explainer
from Explanation.HyperparameterEvaluator import HyperparameterEvaluator


def get_bt_explainer() -> Explainer:
    """
    Loads the BT Staff Rostering problem instance
    @return:
    """
    # this defines the directory where the Partial Solution files will be stored.
    use_dummy = False
    ps_directory = os.path.join("ExplanatoryCachedData", "BT", "Dummy" if use_dummy else "StaffRosteringProblemCacheAGAIN")

    # loads the problem as defined in some files, it should be resources/BT/MartinsInstance
    problem = EfficientBTProblem.from_default_files()

    return Explainer.from_folder(problem=problem,
                                 folder=ps_directory,
                                 polarity_threshold=0.10,
                                 # this is the threshold for polarity as discussed in the paper
                                 verbose=True)

# commented out so that you don't have to install extra things
# def get_gc_explainer() -> Explainer:
#     """
#     Constructs a Graph Colouring problem and its explainer.
#     The problem is the one used in the figure in the paper.
#     @return: The Explainer instance
#     """
#     ps_directory = os.path.join("ExplanatoryCachedData", "BT", "Dummy")
#     problem_file = os.path.join(ps_directory, "bert.json")
#     problem = GraphColouring.from_file(problem_file)
#     problem.view()
#     return Explainer.from_folder(folder=ps_directory,
#                                  problem=problem,
#                                  speciality_threshold=0.50,
#                                  verbose=True)


def explanation() -> None:
    """
    Loads the
    @return: Nothing, it just prints things and manages files
    """
    # constructing the explainer object, which determines the problem and the working directory
    explainer = get_bt_explainer()

    # to generate the files containing PSs, properties etc..
    # You should only run this once, since it is quite slow
    # explainer.generate_files_with_default_settings(50000, 50000)

    # this starts the main explanation function, and uses the files generated above
    explainer.explanation_loop(amount_of_fs_to_propose=2, ps_show_limit=3, show_debug_info=True)


def grid_search() -> None:
    """
    This function gathers the data that is discussed in the Results section of the paper.
    It's grid search, so it's very slow!

    The result is just printed to the console as a json, because it works well with Condor Cluster Computing
    """
    # construct the set of parameters that will be used in the testing
    hype = HyperparameterEvaluator(algorithms_to_test=["NSGAII", "NSGAIII", "MOEAD", "SMS-EMOA"],
                                   problems_to_test=["collaboration_5", "insular_5", "RR_5"],
                                   pRef_sizes_to_test=[10000],
                                   population_sizes_to_test=[50, 100, 200],
                                   pRef_origin_methods=["uniform", "SA", "uniform SA"],
                                   ps_budget=50000,
                                   custom_crowding_operators_to_test=[False, True],
                                   ps_budgets_per_run_to_test=[1000, 2000, 3000, 5000, 10000])

    hype.get_data(ignore_errors=True,
                  verbose=True)


if __name__ == '__main__':
    # the 2 lines below are just to see more detailed errors and logs
    logging.basicConfig(level=logging.INFO)
    warnings.showwarning = utils.warn_with_traceback

    # this line is to run the tests as discussed in the paper
    # grid_search()

    # this line is to run the explainer
    explanation()