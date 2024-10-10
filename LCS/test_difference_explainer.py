import random

from BenchmarkProblems.EfficientBTProblem.EfficientBTProblem import EfficientBTProblem
from BenchmarkProblems.GraphColouring import GraphColouring
from BenchmarkProblems.RoyalRoad import RoyalRoad
from BenchmarkProblems.Trapk import Trapk
from LCS.DifferenceExplainer.DifferenceExplainer import DifferenceExplainer


def test_difference_explainer():
    def messing_around():
        # problem = GraphColouring.make_insular_instance(6)
        #problem = RoyalRoad(5, 4)
        problem = Trapk(5, 5)
        problem = EfficientBTProblem.random_subset_of(EfficientBTProblem.from_default_files(),
                                                     quantity_workers_to_keep=30,
                                                    random_state=42)
        #problem = EfficientBTProblem.from_default_files()
        folder = r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\resources\explanations\messing_around"
        return problem, folder

    def bt_problem():
        problem = EfficientBTProblem.from_default_files()
        folder = r"C:\Users\gac8\PycharmProjects\PS-descriptors-LCS\resources\explanations\heavy_BT_explain"
        return problem, folder

    problem, folder = messing_around()

    random.seed(720)

    diff_explainer = DifferenceExplainer.from_folder(problem=problem,
                                                     folder=folder,
                                                     allow_negative_traits=True,
                                                     allow_positive_traits=True,
                                                     verbose=False)

    diff_explainer.explanation_loop(amount_of_fs_to_propose=10, suppress_errors=False)


test_difference_explainer()
